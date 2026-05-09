import os
import argparse
import threading
import time
import socket
import subprocess
import shutil
import json
import gc
import datetime
import torch
import torch.distributed as dist
import torchaudio
import torchaudio.transforms as T
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template_string, send_from_directory, jsonify, request, render_template

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from util_liveact import center_rescale_crop_keep_ratio, get_embedding, get_msk, get_audio_emb
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
from fp4_gemm import FP4GemmOptions, enable_fp4_gemm
import queue
from datetime import timedelta
import errno

# ================= 1. 全局环境与配置 =================

gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
HLS_ROOT = os.path.join(BASE_DIR, "hls_output")
M3U8_NAME = "live.m3u8"
task_queue = queue.Queue()

os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(HLS_ROOT, exist_ok=True)

# 状态变量
streaming_active = False
task_status_map = {}
task_status_lock = threading.Lock()


# ================= 2. 辅助工具函数 =================

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def resample_audio(audio, sr, fps):
    rate = 25 / fps
    # torchaudio dropped sox_effects in 2.11; functional.speed is the documented replacement
    y, _ = torchaudio.functional.speed(audio, sr, factor=rate)
    resampler = T.Resample(sr, 16000).to(audio.device)
    return resampler(y) * 3.0, 16000


def update_task_status(task_id, **kwargs):
    with task_status_lock:
        if task_id not in task_status_map:
            task_status_map[task_id] = {}
        task_status_map[task_id].update(kwargs)
        task_status_map[task_id]["updated_at"] = time.time()


def get_task_status(task_id):
    with task_status_lock:
        data = task_status_map.get(task_id)
        return dict(data) if data is not None else None


# ================= 3. 分布式推理引擎类 =================

class DistributedVideoEngine:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = self.local_rank
        self.width, self.height = [int(x) for x in args.size.split('*')]
        self.use_dist = self.world_size > 1

        self.video_save_root = os.path.abspath(getattr(args, "video_save_path", "./generated_videos"))
        os.makedirs(self.video_save_root, exist_ok=True)

        if not dist.is_initialized() and self.world_size>1:
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend="nccl", init_method="env://", rank=self.rank, world_size=self.world_size)

        # 多卡时触发：发送python消息，防止长时间不操作，nccl超时异常
        self.control_pg = dist.new_group(backend="gloo") if self.use_dist else None

        if self.world_size>1:
            from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
            init_distributed_environment(rank=self.rank, world_size=self.world_size)
            initialize_model_parallel(sequence_parallel_degree=self.world_size, ring_degree=1,
                                      ulysses_degree=self.world_size)

        # 加载核心生成模型 (Wan2.1)
        # --tp: tensor-parallel (head-shard cache + Megatron self-attn/FFN), supports world_size in {1,2,4,8}
        # else: existing path (Ulysses SP for world>1, single-GPU otherwise; SP hardcoded for world<=2)
        if getattr(args, 'tp', False):
            from model_liveact.model_memory_tp import WanModel
        elif self.world_size > 1:
            from model_liveact.model_memory_sp import WanModel
        else:
            from model_liveact.model_memory import WanModel
        self.wan_i2v_model = WanModel.from_pretrained(args.ckpt_dir, torch_dtype=torch.bfloat16,
                                                      low_cpu_mem_usage=False)
        self.wan_i2v_model = self.wan_i2v_model.to(dtype=torch.bfloat16)

        if getattr(args, 'tp', False) and self.world_size > 1:
            # Pre-slice TP-relevant weights so FP8 GEMM can wrap them transparently.
            # MUST run BEFORE enable_fp8_gemm.
            from model_liveact.model_memory_tp import shard_wan_model_for_tp
            shard_wan_model_for_tp(self.wan_i2v_model)

        # Disabled: vllm 0.11.0 (CUDA 12) incompatible with system CUDA 13. Re-enable after upgrading vllm to a cu13 build.
        # When re-enabled, this call must come AFTER shard_wan_model_for_tp above so that FP8Linear
        # wraps the rank-local weight tensors and quantizes only that rank's slice.
        # enable_fp8_gemm(self.wan_i2v_model, options=FP8GemmOptions())
        # FP4 path (Blackwell only). Mutually exclusive with --fp8_gemm; both
        # paths require vllm so the same cu13 caveat above applies. Wrap order
        # is identical: must run AFTER shard_wan_model_for_tp.
        if getattr(args, 'fp4_gemm', False):
            enable_fp4_gemm(self.wan_i2v_model, options=FP4GemmOptions())
        if args.block_offload:
            for name, child in self.wan_i2v_model.named_children():
                if name != 'blocks':
                    child.to(self.device)
            self.wan_i2v_model.enable_block_offload(
                onload_device=torch.device(f"cuda:{self.device}"),
            )
        else:
            self.wan_i2v_model = self.wan_i2v_model.to(self.device)
        self.wan_i2v_model.freqs = self.wan_i2v_model.freqs.to(self.device)
        self.wan_i2v_model.eval()
        self.wan_i2v_model = torch.compile(self.wan_i2v_model, mode="max-autotune-no-cudagraphs", backend="inductor", dynamic=False)

        # 采样参数
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.timesteps = [torch.tensor([_]).to(self.device, dtype=torch.float32) for _ in
                          [1000.0, 937.5, 833.33333333, 0.0]]

        # 加载辅件 (VAE / CLIP / T5 / Audio)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (self.height, self.width))),
            transforms.ToTensor(),
            transforms.Resize((self.height, self.width)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.vae = LightVAE(vae_path=os.path.join(args.ckpt_dir, 'Wan2.1_VAE.pth'), dtype=torch.bfloat16,
                            device=self.device,
                            use_lightvae=False, parallel=(self.world_size > 1))

        self.clip = CLIPModel(
            checkpoint_path=os.path.join(args.ckpt_dir, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
            tokenizer_path=os.path.join(args.ckpt_dir, 'xlm-roberta-large'), dtype=torch.bfloat16, device=self.device)

        self.text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16,
                                           device='cpu' if args.t5_cpu else self.device,
                                           checkpoint_path=os.path.join(args.ckpt_dir,
                                                                        'models_t5_umt5-xxl-enc-bf16.pth'),
                                           tokenizer_path=os.path.join(args.ckpt_dir, 'google/umt5-xxl'))

        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            args.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir,
                                                                                  local_files_only=True)

        torch.cuda.empty_cache()
        # 初始化KV Cache
        self.blksz_lst = [6, 8]
        self.frame_len = (self.height // (self.patch_size[1] * self.vae_stride[1])) * (
                    self.width // (self.patch_size[2] * self.vae_stride[2]))
        if getattr(args, 'tp', False):
            # TP path: cache holds full seq, num_heads sharded across ranks.
            kv_cache_tokens = self.frame_len * sum(self.blksz_lst)
            kv_cache_heads = 40 // max(self.world_size, 1)
            assert 40 % max(self.world_size, 1) == 0, (
                f"--tp requires num_heads (40) divisible by world_size; got world_size={self.world_size}"
            )
        else:
            # SP path: cache holds rank's seq slice, full num_heads.
            kv_cache_tokens = self.frame_len * sum(self.blksz_lst) // self.world_size
            kv_cache_heads = 40
        kv_cache_device = self.device
        kv_cache_dtype = torch.float8_e4m3fn if args.fp8_kv_cache else torch.bfloat16
        kv_scale_shape = (1, kv_cache_tokens, kv_cache_heads, 1)
        self.kv_cache  = \
            {
                i: {
                    layer_id: {
                        'k': torch.zeros([1, kv_cache_tokens, kv_cache_heads, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                        'v': torch.zeros([1, kv_cache_tokens, kv_cache_heads, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                        'k_scale': torch.ones(kv_scale_shape, dtype=torch.float32,
                                              device=kv_cache_device) if args.fp8_kv_cache else None,
                        'v_scale': torch.ones(kv_scale_shape, dtype=torch.float32,
                                              device=kv_cache_device) if args.fp8_kv_cache else None,
                        'mean_memory': False,
                        'offload_cache': False,
                        'fp8_kv_cache': args.fp8_kv_cache,
                    }
                    for layer_id in range(40)
                } for i in range(len(self.timesteps) - 1)
            }
        for n in range(40):
            self.wan_i2v_model.blocks[n].self_attn.init_kvidx(self.frame_len, self.world_size)

        # 编译加速
        self.vae.model.eval()
        # self.vae.encode = torch.compile(self.vae.encode)
        self.vae.decode = torch.compile(self.vae.decode)

        # 预热
        print("开始预热")
        start_time = time.perf_counter()
        self._warmup()
        print(f"Total Warmup time {time.perf_counter() - start_time:.4f}s")

    def _warmup(self):
        print(f"[Warmup][Rank {self.rank}] start", flush=True)

        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        try:
            with torch.no_grad():
                frame_num_init = (sum(self.blksz_lst) - 1) * 4 + 1
                # 1. 准备假图像
                cond_image = torch.randn(
                    1, 3, 1, self.height, self.width,
                    device=self.device, dtype=torch.bfloat16
                ).clamp_(-1, 1)
                # 2. CLIP
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    clip_context = self.clip.visual(cond_image)

                # 3. 假音频
                dummy_audio = torch.randn(16000 * 6)
                audio_embedding = get_embedding(
                    dummy_audio,
                    self.wav2vec_feature_extractor,
                    self.audio_encoder,
                    device=self.device)
                # 4. init y
                ref_target_masks = torch.ones(
                    3,
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    device=self.device,
                    dtype=torch.bfloat16)

                video_frames_placeholder = torch.zeros(
                    1,
                    cond_image.shape[1],
                    frame_num_init - cond_image.shape[2],
                    self.height,
                    self.width,
                    device=self.device,
                    dtype=torch.bfloat16)

                padding_frames = torch.concat([cond_image, video_frames_placeholder], dim=2)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = self.vae.encode(padding_frames).to(self.device).unsqueeze(0)

                msk = get_msk(frame_num_init, cond_image, self.vae_stride, self.device)
                y = torch.concat([msk, y], dim=1)

                # 5. prompt
                context = [
                    self.text_encoder(
                        texts="A person is speaking naturally.",
                        device='cpu' if self.args.t5_cpu else self.device
                    )[0].to(self.device, dtype=torch.bfloat16)
                ]

                # 6. 完全按原逻辑跑，只是 iter_total_num = 2
                iter_total_num = 2
                pre_latent = None

                for iteration in range(iter_total_num):
                    audio_start_idx = 0 if iteration == 0 else (iteration - 1) * self.blksz_lst[-1] * self.vae_stride[0]
                    audio_end_idx = audio_start_idx + frame_num_init

                    audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, self.device)

                    y_cut = y[:, :, :frame_num_init // 4 + 1, ...]
                    f_idx = 0 if iteration == 0 else 1
                    latent = torch.randn(
                        16,
                        self.blksz_lst[f_idx],
                        self.height // self.vae_stride[1],
                        self.width // self.vae_stride[2],
                        dtype=torch.bfloat16,
                        device=self.device
                    )

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for i in range(len(self.timesteps) - 1):
                            timestep = self.timesteps[i]
                            arg_c = {
                                'context': context,
                                'clip_fea': clip_context,
                                'ref_target_masks': ref_target_masks,
                                'audio': audio_embs,
                                'y': y_cut[:, :, sum(self.blksz_lst[:f_idx]):sum(self.blksz_lst[:f_idx + 1])],
                                'start_idx': sum(self.blksz_lst[:f_idx]) * self.frame_len,
                                'end_idx': sum(self.blksz_lst[:f_idx + 1]) * self.frame_len,
                                'update_cache': iteration > 1
                            }

                            noise_pred = self.wan_i2v_model(
                                [latent],
                                t=timestep,
                                kv_cache=self.kv_cache[i],
                                skip_audio=False if i in [1, 2] else True,
                                **arg_c
                            )[0]

                            dt = (self.timesteps[i] - self.timesteps[i + 1]) / 1000
                            latent = latent + (-noise_pred) * dt[0]

                        if iteration == 0:
                            _videos = self.vae.decode(latent)
                        else:
                            combined_latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
                            _videos = self.vae.decode(combined_latent)[:, :, 9:]

                        pre_latent = latent
                    torch.cuda.synchronize(self.device)
                    print(f"[Warmup][Rank {self.rank}] iteration {iteration + 1}/2 done", flush=True)

                del cond_image, clip_context, dummy_audio, audio_embedding
                del ref_target_masks, video_frames_placeholder, padding_frames
                del y, msk, context, audio_embs, y_cut, latent, pre_latent, _videos
                if 'combined_latent' in locals():
                    del combined_latent
                if 'noise_pred' in locals():
                    del noise_pred

                # torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)

            if dist.is_initialized():
                dist.barrier()

            print(f"[Warmup][Rank {self.rank}] done", flush=True)
        except Exception as e:
            print(f"[Warmup][Rank {self.rank}] failed: {e}", flush=True)
            raise

    def generate_and_push(self, params):
        global streaming_active

        prompt_list = params['prompt_list']
        fps = int(params['fps'])
        img_path = params['img_path']
        audio_path = params['audio_path']
        task_id = params['task_id']
        main_prompt = params['main_prompt']
        stream_with_audio = bool(params.get('stream_with_audio', False))

        task_hls_dir = os.path.join(HLS_ROOT, task_id)
        final_video_path = os.path.join(self.video_save_root, f"{task_id}.mp4")

        hls_ffmpeg_process = None
        save_ffmpeg_process = None
        stats = {}

        def close_proc(proc, name="ffmpeg"):
            if proc is None:
                return
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                ret = proc.wait()
                if ret != 0:
                    print(f"[{name}] exited with code {ret}", flush=True)
            except Exception as e:
                print(f"[{name}] wait failed: {e}", flush=True)

        def write_chunk_bytes(proc, chunk_bytes, name="ffmpeg"):
            if proc is None or proc.stdin is None:
                return
            try:
                proc.stdin.write(chunk_bytes)
                proc.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError(f"{name} stdin broken pipe")
            except Exception as e:
                raise RuntimeError(f"write to {name} failed: {e}")

        def tensor_chunk_to_rgb_bytes(video_tensor):
            """
            video_tensor: [1, 3, T, H, W] in [-1, 1]
            return:
                chunk_bytes: 整个 chunk 的连续 rgb24 bytes
                num_frames: 这个 chunk 的帧数
            """
            video_u8 = (
                ((video_tensor.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .contiguous()
                .cpu()
            )  # [T, H, W, C], uint8
            num_frames = video_u8.shape[0]
            chunk_bytes = video_u8.numpy().tobytes()
            return chunk_bytes, num_frames

        try:
            if self.rank == 0:
                update_task_status(
                    task_id,
                    status="running",
                    stage="preparing",
                    message="开始预处理",
                    generated_chunks=0,
                    is_done=False,
                    error=None,
                    stream_ready=False,
                )

            # 1. 音频特征预处理
            if self.rank == 0:
                start_time = time.perf_counter()

            audio_ori, sr_ori = torchaudio.load(audio_path)
            audio_resampled, _ = resample_audio(audio_ori, sr_ori, fps)
            audio_embedding = get_embedding(
                audio_resampled[0],
                self.wav2vec_feature_extractor,
                self.audio_encoder,
                device=self.device
            )
            audio_len_sec = audio_ori.size(1) / sr_ori

            if self.rank == 0:
                stats['audio_proc'] = time.perf_counter() - start_time
                update_task_status(task_id, stage="audio_ready", message="音频加载完成")

            # 2. Rank0 启动 ffmpeg
            if self.rank == 0:
                start_time = time.perf_counter()

                if os.path.exists(task_hls_dir):
                    shutil.rmtree(task_hls_dir)
                os.makedirs(task_hls_dir, exist_ok=True)

                if os.path.exists(final_video_path):
                    os.remove(final_video_path)

                # ---------- HLS ffmpeg ----------
                hls_ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',
                    '-loglevel', 'warning',

                    # rawvideo input
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{self.width}x{self.height}',
                    '-r', str(fps),
                    '-i', 'pipe:0',
                ]

                if stream_with_audio:
                    hls_ffmpeg_cmd += [
                        '-thread_queue_size', '1024',
                        '-i', audio_path,
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-af', 'aresample=async=1:first_pts=0',
                        '-shortest',
                    ]
                else:
                    hls_ffmpeg_cmd += [
                        '-an',
                        '-map', '0:v:0',
                    ]

                hls_ffmpeg_cmd += [
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',

                    # 固定 1 秒一个关键帧，方便 HLS 切片
                    '-g', str(fps),
                    '-keyint_min', str(fps),
                    '-sc_threshold', '0',

                    '-f', 'hls',
                    '-hls_time', '1',
                    '-hls_list_size', '5',
                    '-hls_segment_type', 'mpegts',
                    '-hls_flags', 'delete_segments+append_list+independent_segments',
                    os.path.join(task_hls_dir, M3U8_NAME)
                ]

                print(f"[Generate][{task_id}] hls_ffmpeg_cmd = {' '.join(map(str, hls_ffmpeg_cmd))}", flush=True)
                hls_ffmpeg_process = subprocess.Popen(
                    hls_ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    bufsize=0
                )

                # ---------- 保存 mp4 ffmpeg ----------
                # 直接带音频保存，不再先写 silent.mp4 再二次 mux
                save_ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',
                    '-loglevel', 'warning',

                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'rgb24',
                    '-s', f'{self.width}x{self.height}',
                    '-r', str(fps),
                    '-i', 'pipe:0',

                    '-thread_queue_size', '1024',
                    '-i', audio_path,

                    '-map', '0:v:0',
                    '-map', '1:a:0',

                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-af', 'aresample=async=1:first_pts=0',
                    '-shortest',
                    '-movflags', '+faststart',
                    final_video_path
                ]

                print(f"[Generate][{task_id}] save_ffmpeg_cmd = {' '.join(map(str, save_ffmpeg_cmd))}", flush=True)
                save_ffmpeg_process = subprocess.Popen(
                    save_ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    bufsize=0
                )

                stats['ffmpeg_proc'] = time.perf_counter() - start_time
                update_task_status(task_id, stage="ffmpeg_ready", message="推流器已启动")

            # 3. 图像 / 条件
            if self.rank == 0:
                start_time = time.perf_counter()

            image = Image.open(img_path).convert("RGB")
            cond_image = self.transform(image).unsqueeze(1).unsqueeze(0).to(self.device, torch.bfloat16)

            if self.rank == 0:
                stats['image_proc'] = time.perf_counter() - start_time

            if self.rank == 0:
                start_time = time.perf_counter()

            with torch.no_grad():
                clip_context = self.clip.visual(cond_image)

            if self.rank == 0:
                stats['clip_proc'] = time.perf_counter() - start_time

            if self.rank == 0:
                start_time = time.perf_counter()

            torch.manual_seed(self.args.seed)
            ref_target_masks = torch.ones(
                3,
                self.height // self.vae_stride[1],
                self.width // self.vae_stride[2],
                device=self.device,
                dtype=torch.bfloat16
            )
            frame_num_init = (sum(self.blksz_lst) - 1) * 4 + 1
            msk = get_msk(frame_num_init, cond_image, self.vae_stride, self.device)
            video_frames_placeholder = torch.zeros(
                1,
                cond_image.shape[1],
                frame_num_init - cond_image.shape[2],
                self.height,
                self.width,
                device=self.device,
                dtype=torch.bfloat16
            )
            padding_frames = torch.concat([cond_image, video_frames_placeholder], dim=2)
            y = self.vae.encode(padding_frames).to(self.device).unsqueeze(0)
            y = torch.concat([msk, y], dim=1)

            if self.rank == 0:
                stats['init_y'] = time.perf_counter() - start_time

            if self.rank == 0:
                start_time = time.perf_counter()

            edit_prompts = {}
            if prompt_list:
                for edit_prompt in prompt_list:
                    key = (edit_prompt[0], edit_prompt[1])
                    edit_prompts[key] = [
                        self.text_encoder(
                            texts=edit_prompt[2],
                            device='cpu' if self.args.t5_cpu else self.device
                        )[0].to(self.device, dtype=torch.bfloat16)
                    ]

            context_0 = [
                self.text_encoder(
                    texts=main_prompt,
                    device='cpu' if self.args.t5_cpu else self.device
                )[0].to(self.device, dtype=torch.bfloat16)
            ]

            if self.rank == 0:
                stats['prompt_init'] = time.perf_counter() - start_time

            print("\n" + "=" * 30)
            print(f"Task {task_id} Pre-processing Report:")
            for stage, duration in stats.items():
                print(f" - {stage:20}: {duration:.4f}s")
            print("=" * 30 + "\n")

            # 4. 主循环
            iter_total_num = int(audio_len_sec / (self.vae_stride[0] * self.blksz_lst[-1] / fps)) + 1
            pre_latent = None
            chunk_times = []

            if self.rank == 0:
                update_task_status(
                    task_id,
                    status="running",
                    stage="generating",
                    message=f"计划生成 {iter_total_num} 个 chunk",
                    total_chunks=iter_total_num,
                    generated_chunks=0,
                    is_done=False,
                )

            for iteration in range(iter_total_num):
                if self.rank == 0:
                    start_time = time.perf_counter()

                cached_context = context_0
                if prompt_list:
                    for k, v in edit_prompts.items():
                        if k[0] <= iteration <= k[1]:
                            cached_context = v
                            break

                audio_start_idx = 0 if iteration == 0 else (iteration - 1) * self.blksz_lst[-1] * self.vae_stride[0]
                audio_end_idx = audio_start_idx + frame_num_init
                audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, self.device)

                y_cut = y[:, :, :frame_num_init // 4 + 1, ...]
                f_idx = 0 if iteration == 0 else 1

                latent = torch.randn(
                    16,
                    self.blksz_lst[f_idx],
                    self.height // self.vae_stride[1],
                    self.width // self.vae_stride[2],
                    dtype=torch.bfloat16,
                    device=self.device
                )

                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for i in range(len(self.timesteps) - 1):
                        timestep = self.timesteps[i]
                        arg_c = {
                            'context': cached_context,
                            'clip_fea': clip_context,
                            'ref_target_masks': ref_target_masks,
                            'audio': audio_embs,
                            'y': y_cut[:, :, sum(self.blksz_lst[:f_idx]):sum(self.blksz_lst[:f_idx + 1])],
                            'start_idx': sum(self.blksz_lst[:f_idx]) * self.frame_len,
                            'end_idx': sum(self.blksz_lst[:f_idx + 1]) * self.frame_len,
                            'update_cache': iteration > 1
                        }
                        noise_pred = self.wan_i2v_model(
                            [latent],
                            t=timestep,
                            kv_cache=self.kv_cache[i],
                            skip_audio=False if i in [1, 2] else True,
                            **arg_c
                        )[0]
                        dt = (self.timesteps[i] - self.timesteps[i + 1]) / 1000
                        latent = latent + (-noise_pred) * dt[0]

                    if iteration == 0:
                        _videos = self.vae.decode(latent)
                    else:
                        combined_latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
                        _videos = self.vae.decode(combined_latent)[:, :, 9:]

                    pre_latent = latent

                if self.rank == 0:
                    # 这里改成“整个 chunk 一次写入”
                    chunk_bytes, num_frames_this_chunk = tensor_chunk_to_rgb_bytes(_videos)

                    write_chunk_bytes(hls_ffmpeg_process, chunk_bytes, name="hls_ffmpeg")
                    write_chunk_bytes(save_ffmpeg_process, chunk_bytes, name="save_ffmpeg")

                    m3u8_path = os.path.join(task_hls_dir, M3U8_NAME)
                    chunk_elapsed = time.perf_counter() - start_time
                    chunk_times.append(chunk_elapsed)
                    chunk_fps = num_frames_this_chunk / chunk_elapsed if chunk_elapsed > 0 else 0.0
                    avg_s = sum(chunk_times) / len(chunk_times)
                    min_s = min(chunk_times)
                    max_s = max(chunk_times)
                    update_task_status(
                        task_id,
                        status="running",
                        stage="generating",
                        message=f"已生成 {iteration + 1}/{iter_total_num} 个 chunk",
                        total_chunks=iter_total_num,
                        generated_chunks=iteration + 1,
                        is_done=False,
                        stream_ready=os.path.exists(m3u8_path),
                        last_chunk_s=round(chunk_elapsed, 3),
                        last_chunk_frames=num_frames_this_chunk,
                        last_chunk_fps=round(chunk_fps, 2),
                        avg_chunk_s=round(avg_s, 3),
                        min_chunk_s=round(min_s, 3),
                        max_chunk_s=round(max_s, 3),
                    )

                    print(
                        f"生成完成 {iteration + 1}/{iter_total_num}, "
                        f"frames={num_frames_this_chunk}, "
                        f"一个chunk耗时:{chunk_elapsed:.4f}s "
                        f"({chunk_fps:.2f} FPS, avg {avg_s:.3f}s)",
                        flush=True
                    )

            # 5. 收尾
            if self.rank == 0:
                update_task_status(
                    task_id,
                    status="running",
                    stage="finalizing",
                    message="视频生成完成，正在封装最终文件",
                    is_done=False,
                )

                close_proc(hls_ffmpeg_process, name="hls_ffmpeg")
                close_proc(save_ffmpeg_process, name="save_ffmpeg")

                print(f"[Save] 最终视频已保存到: {final_video_path}", flush=True)

                update_task_status(
                    task_id,
                    status="finished",
                    stage="finished",
                    message="生成完成",
                    total_chunks=iter_total_num,
                    generated_chunks=iter_total_num,
                    is_done=True,
                    stream_ready=True,
                    error=None,
                    final_video_path=final_video_path,
                )

        except Exception as e:
            print(f"[Generate] 生成失败: {e}", flush=True)

            if self.rank == 0:
                try:
                    close_proc(hls_ffmpeg_process, name="hls_ffmpeg")
                    close_proc(save_ffmpeg_process, name="save_ffmpeg")
                except Exception:
                    pass

                update_task_status(
                    task_id,
                    status="failed",
                    stage="failed",
                    message=f"生成失败: {e}",
                    is_done=True,
                    error=str(e),
                )
                streaming_active = False
            raise

        finally:
            if self.rank == 0:
                streaming_active = False


# ================= 4. Flask 路由 (与前端对接) =================
def control_loop_rank0():
    global streaming_active

    while True:
        try:
            params = task_queue.get(timeout=1.0)
        except queue.Empty:
            params = None

        if engine.use_dist:
            payload = [params]
            dist.broadcast_object_list(payload, src=0, group=engine.control_pg)

        if params is None:
            continue

        try:
            update_task_status(
                params['task_id'],
                status="running",
                stage="starting",
                message="任务开始执行"
            )
            engine.generate_and_push(params)
        finally:
            streaming_active = False


def control_loop_rank_other():
    if not engine.use_dist:
        return
    while True:
        payload = [None]
        dist.broadcast_object_list(payload, src=0, group=engine.control_pg)
        params = payload[0]

        if params is None:
            continue

        engine.generate_and_push(params)
        torch.cuda.empty_cache()
        gc.collect()


@app.route('/')
def index():
    return render_template('index.html', stream_resolution=engine.args.size.replace('*', 'x'))


@app.route('/start_stream', methods=['POST'])
def start_stream():
    global streaming_active

    if streaming_active:
        return jsonify({"status": "error", "message": "GPU 任务繁忙，请稍后再试"}), 429

    task_id = request.form.get('task_id')
    main_prompt = (request.form.get('main_prompt') or '').strip()
    prompt_json = request.form.get('prompt_json') or '[]'
    fps = request.form.get('fps')
    prompt_list = json.loads(prompt_json)

    stream_with_audio = str(request.form.get('stream_with_audio', 'false')).lower() in ('1', 'true', 'yes', 'on')
    img_file = request.files.get('img_file')
    audio_file = request.files.get('audio_file')
    if not task_id:
        return jsonify({"status": "error", "message": "缺少 task_id"}), 400
    if not img_file or not audio_file:
        return jsonify({"status": "error", "message": "缺少图片或音频文件"}), 400
    if not fps:
        return jsonify({"status": "error", "message": "缺少 fps"}), 400
    task_upload_dir = os.path.join(UPLOAD_ROOT, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)
    img_path = os.path.join(task_upload_dir, "input.png")
    audio_path = os.path.join(task_upload_dir, "input.wav")
    img_file.save(img_path)
    audio_file.save(audio_path)
    params = {
        'task_id': task_id,
        'prompt_list': prompt_list,
        'main_prompt': main_prompt,
        'fps': int(fps),
        'img_path': img_path,
        'audio_path': audio_path,
        'stream_with_audio': stream_with_audio, }

    update_task_status(
        task_id,
        status="queued",
        stage="queued",
        message="任务已入队，等待执行",
        total_chunks=None,
        generated_chunks=0,
        is_done=False,
        stream_ready=False,
        error=None,
        stream_with_audio=stream_with_audio, )

    streaming_active = True
    task_queue.put(params)
    return jsonify({
        "status": "success",
        "task_id": task_id,
        "stream_with_audio": stream_with_audio
    })


@app.route('/stream/<task_id>/<path:filename>')
def serve_hls(task_id, filename):
    return send_from_directory(os.path.join(HLS_ROOT, task_id), filename)


@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    data = get_task_status(task_id)
    if data is None:
        return jsonify({
            "status": "not_found",
            "message": "task_id 不存在"
        }), 404
    return jsonify(data)


# ================= 5. 分布式启动 =================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--wav2vec_dir", type=str, required=True)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--size", type=str, default="720*416",
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.")
    parser.add_argument("--video_save_path", type=str, default="./generated_videos",
                        help="Directory to save final generated videos.")
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
        "--tp",
        action="store_true",
        default=False,
        help="Use tensor-parallel WanModel (model_liveact/model_memory_tp.py). "
             "Supports world_size in {1,2,4,8}. Default uses Ulysses sequence "
             "parallelism (model_memory_sp.py), which only supports world_size <= 2.")
    parser.add_argument(
        "--fp4_gemm",
        action="store_true",
        default=False,
        help="Wrap nn.Linear with NVFP4 W4A4 GEMM (Blackwell SM100+ only — "
             "B100/B200/RTX 5090/6000). Mutually exclusive with --fp8_gemm. "
             "Falls back to BF16 silently on unsupported GPUs unless "
             "VLLM_USE_NVFP4_CT_EMULATIONS=1 is set for software emulation.")
    args = parser.parse_args()

    try:
        engine = DistributedVideoEngine(args)
        if engine.rank == 0:
            threading.Thread(target=control_loop_rank0, daemon=True).start()

            ip = get_local_ip()
            print(f"\n🚀 LiveAct 服务启动!")
            print(f"访问地址: http://{ip}:{args.port}\n")
            app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
        else:
            print(f"节点 Rank {engine.rank} 等待指令...")
            control_loop_rank_other()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()