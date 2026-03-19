import os
import numpy as np
import random
import math
import time
import ast
from tqdm import tqdm
import argparse
import json

import torch
from torch import nn
import torch.distributed as dist
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from util_liveact import *

from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from diffusers.utils import export_to_video

from fp8_gemm import FP8GemmOptions, enable_fp8_gemm


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt, image and audio"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="480*832",
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--offload_cache",
        action="store_true",
        default=False,
        help="Whether to place kv cache on CPU.")
    parser.add_argument(
        "--fp8_kv_cache",
        action="store_true",
        default=False,
        help="Whether to store kv cache in FP8 and dequantize to BF16 on use.")
    parser.add_argument(
        "--block_offload",
        action="store_true",
        default=False,
        help="Whether to offload WanModel blocks to CPU between block forwards.")
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="The target fps.")
    parser.add_argument(
        "--audio_cfg",
        type=float,
        default=1.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--dura_print",
        action="store_true",
        default=False,
        help="Whether print duration for every block.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--steam_audio",
        action="store_true",
        default=False,
        help="Whether inference with steaming audio.")
    parser.add_argument(
        "--mean_memory",
        action="store_true",
        default=False,
        help="Whether inference with mean memory strategy.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")

    args = parser.parse_args()

    return args


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    if world_size > 1:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=world_size,
        )

    if world_size > 1:
        from model_liveact.model_memory_sp import WanModel
    else:
        from model_liveact.model_memory import WanModel

    width, height = [int(_) for _ in args.size.split('*')]
    fps = args.fps
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    timesteps = [torch.tensor([_]).to(device, dtype=torch.float32) for _ in [1000.0, 937.5, 833.33333333, 0.0]]
    blksz_lst = [6, 8]
    frame_len = (height // (patch_size[1] * vae_stride[1])) * (width // (patch_size[2] * vae_stride[2]))
    kv_cache_tokens = frame_len * sum(blksz_lst) // world_size
    kv_cache_device = 'cpu' if args.offload_cache else device
    kv_cache_dtype = torch.float8_e4m3fn if args.fp8_kv_cache else torch.bfloat16
    kv_scale_shape = (1, kv_cache_tokens, 40, 1)
    kv_cache = \
        {
            i: {
                layer_id: {
                    'k': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'v': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'k_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'v_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'mean_memory': args.mean_memory,
                    'offload_cache': args.offload_cache,
                    'fp8_kv_cache': args.fp8_kv_cache,
                }
                for layer_id in range(40)
            } for i in range(len(timesteps) - 1)
        }
    if args.audio_cfg > 1.0:
        kv_cache_null_audio = \
            {
                i: {layer_id: {
                    'k': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'v': torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                    'k_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'v_scale': torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if args.fp8_kv_cache else None,
                    'mean_memory': args.mean_memory,
                    'offload_cache': args.offload_cache,
                    'fp8_kv_cache': args.fp8_kv_cache,
                } for layer_id in range(40)} for i in range(len(timesteps) - 1)
            }

    wan_i2v_model = WanModel.from_pretrained(args.ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    wan_i2v_model = wan_i2v_model.to(dtype=torch.bfloat16)
    for n in range(40):
        wan_i2v_model.blocks[n].self_attn.init_kvidx(frame_len, world_size)

    enable_fp8_gemm(wan_i2v_model, options=FP8GemmOptions())
    if args.block_offload:
        for name, child in wan_i2v_model.named_children():
            if name != 'blocks':
                child.to(device)
        wan_i2v_model.enable_block_offload(
            onload_device=torch.device(f"cuda:{device}"),
        )
    else:
        wan_i2v_model = wan_i2v_model.to(device)
    wan_i2v_model.eval()
    wan_i2v_model = torch.compile(wan_i2v_model, mode="max-autotune-no-cudagraphs", backend="inductor", dynamic=False)

    vae = LightVAE(vae_path=os.path.join(args.ckpt_dir, 'Wan2.1_VAE.pth'), dtype=torch.bfloat16, device=device,
                   use_lightvae=False, parallel=(world_size > 1))

    clip = CLIPModel(
        checkpoint_path=os.path.join(args.ckpt_dir, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
        tokenizer_path=os.path.join(args.ckpt_dir, 'xlm-roberta-large'), dtype=torch.bfloat16, device=device)
    clip.model = clip.model.to(device, dtype=torch.bfloat16)

    text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device='cpu' if args.t5_cpu else device,
                                  checkpoint_path=os.path.join(args.ckpt_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
                                  tokenizer_path=os.path.join(args.ckpt_dir, 'google/umt5-xxl'))

    audio_encoder = Wav2Vec2Model.from_pretrained(
        args.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device, dtype=torch.bfloat16).eval()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir, local_files_only=True)

    audio_encoder.feature_extractor._freeze_parameters()
    wan_i2v_model.freqs = wan_i2v_model.freqs.to(device)
    for _model in [wan_i2v_model, clip.model, audio_encoder, vae.model]:
        for name, param in _model.named_parameters():
            param.requires_grad = False

    vae.model.eval()
    vae.encode = torch.compile(vae.encode)
    vae.decode = torch.compile(vae.decode)

    torch_gc()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (height, width))),
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    for data in input_data:
        image_path = data['cond_image']
        audio_path = data['cond_audio']
        out_path = os.path.basename(image_path).split('.')[0] + '_' + os.path.basename(audio_path).split('.')[0] + '.mp4'
        prompt = data['prompt']
        edit_prompts = data.get('edit_prompt', {})

        context = [text_encoder(texts=prompt, device='cpu' if args.t5_cpu else device)[0].to(device, dtype=torch.bfloat16)]
        if edit_prompts:
            edit_prompts = {
                k: text_encoder(texts=v, device='cpu' if args.t5_cpu else device)[0].to(device, dtype=torch.bfloat16)
                for k, v in edit_prompts.items()
            }

        image = Image.open(image_path).convert("RGB")
        cond_image = transform(image).unsqueeze(1).unsqueeze(0).to(device, torch.bfloat16)  # 1 C 1 H W
        clip.model.to(device)
        clip_context = clip.visual(cond_image)  # 1, 257, 1280
        clip.model.cpu()
        torch_gc()

        audio_ori, sr_ori = torchaudio.load(audio_path)  # y: [channels, time]
        def resample_audio(audio, sr, fps):
            rate = 25 / fps
            effects = [["tempo", f"{rate}"], ]
            y, sr = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
            resampler = T.Resample(sr, 16000)
            return resampler(y) * 3.0, 16000

        audio, sr = resample_audio(audio_ori, sr_ori, fps)
        audio_embedding = get_embedding(audio[0], wav2vec_feature_extractor, audio_encoder, device=device)
        audio_len = audio_ori.size(1) / sr_ori

        ref_target_masks = torch.ones(3, height // vae_stride[1], width // vae_stride[2]).to(device, torch.bfloat16)
        frame_num = (sum(blksz_lst) - 1) * 4 + 1
        msk = get_msk(frame_num, cond_image, vae_stride, device)

        def get_y(frame_num):
            video_frames = torch.zeros(
                1, cond_image.shape[1], frame_num - cond_image.shape[2], height, width
            ).to(cond_image.device, cond_image.dtype)
            padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
            y = vae.encode(padding_frames_pixels_values.to(vae.device)).to(wan_i2v_model.device).unsqueeze(0)
            y = torch.concat([msk, y], dim=1)
            return y

        y = get_y(frame_num)

        iter_total_num = int(audio_len / (vae_stride[0] * blksz_lst[-1] / fps)) + 1
        print('----iter_total_num=', iter_total_num)
        gen_video_list = []
        torch.manual_seed(args.seed)
        for _ in range(iter_total_num):
            t1 = time.time()
            audio_start_idx, audio_end_idx = 0, frame_num
            if (_ - 1) * blksz_lst[-1] * vae_stride[0] > 0:
                audio_start_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]
                audio_end_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]

            if not args.steam_audio:
                audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, device)
            else:
                audio, sr = resample_audio(
                    audio_ori[:1, int(sr_ori*(audio_start_idx/fps)):int(sr_ori*((audio_end_idx+2)/fps))], sr_ori, fps
                )
                audio_embedding = get_embedding(audio[0], wav2vec_feature_extractor, audio_encoder, device=device)
                audio_embs = get_audio_emb(audio_embedding, 0, frame_num, device)

            y_cut = y[:, :, :frame_num // 4 + 1, ...]

            _context = context
            if edit_prompts:
                for k, v in edit_prompts.items():
                    if ast.literal_eval(k)[0] <= _ <= ast.literal_eval(k)[1]:
                        _context = [v]
                        break

            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                f = _ if _ <= 1 else 1
                latent = torch.randn(16, blksz_lst[f], height // vae_stride[1], width // vae_stride[2],
                                     dtype=torch.bfloat16, device=device)
                for i in tqdm(range(len(timesteps) - 1)):
                    timestep = timesteps[i]
                    arg_c = {'context': _context, 'clip_fea': clip_context, 'ref_target_masks': ref_target_masks,
                             'audio': audio_embs, 'y': y_cut[:, :, sum(blksz_lst[:f]):sum(blksz_lst[:f + 1])],
                             'start_idx': sum(blksz_lst[:f]) * frame_len, 'end_idx': sum(blksz_lst[:f + 1]) * frame_len,
                             'update_cache': _ > 1}
                    noise_pred = wan_i2v_model([latent.to(device)], t=timestep, kv_cache=kv_cache[i],
                                               skip_audio=False if i in [1, 2] else False, **arg_c)[0]

                    if args.audio_cfg>1.0 and i in [1, 2]:
                        arg_null_audio = \
                            {'context': _context, 'clip_fea': clip_context, 'ref_target_masks': ref_target_masks,
                             'audio': torch.zeros_like(audio_embs), 'y': y_cut[:, :, sum(blksz_lst[:f]):sum(blksz_lst[:f + 1])],
                             'start_idx': sum(blksz_lst[:f]) * frame_len, 'end_idx': sum(blksz_lst[:f + 1]) * frame_len,
                             'update_cache': _ > 1}
                        noise_pred_drop_audio = wan_i2v_model([latent.to(device)], t=timestep, kv_cache=kv_cache_null_audio[i],
                                                              **arg_null_audio)[0]
                        noise_pred = noise_pred_drop_audio + args.audio_cfg * (noise_pred - noise_pred_drop_audio)

                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / 1000
                    latent = latent + (-noise_pred) * dt[0]

                if f == 0:
                    _latent = latent
                    _videos = vae.decode(_latent.squeeze(0))
                else:
                    _latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
                    _videos = vae.decode(_latent.squeeze(0))[:, :, 9:]
                pre_latent = latent
                gen_video_list.append(_videos.cpu())

                if args.dura_print:
                    torch.cuda.synchronize()
                    if rank == 0:
                        t2 = time.time()
                        dura = blksz_lst[f] * vae_stride[0] / fps * 1000
                        print(f"Done Block {_}: duration {dura}ms video cost {(t2 - t1) * 1000:.2f} ms")
        torch.cuda.synchronize()
        # torch_gc()

        videos = (torch.concat(gen_video_list, dim=2).permute((0, 2, 3, 4, 1))[0] + 1.0) / 2
        video_path = 'tmp.mp4'
        export_to_video(videos[:, ...].float().cpu().numpy(), video_path, fps=fps)
        add_audio_to_video(video_path, audio_path, out_path)

        torch.cuda.synchronize()
        # torch_gc()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
