<div align="center">

<img src="./assets/logo.png" alt="LiveAct Logo" width="30%">

# SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV Memory

[Dingcheng Zhen*<sup>&#9993;</sup>](https://scholar.google.com/citations?user=jSLx3CcAAAAJ) · [Xu Zheng*](https://scholar.google.com/citations?user=Ii1c51QAAAAJ) · [Ruixin Zhang*](https://openreview.net/profile?id=~Ruixin_Zhang5) · [Zhiqi Jiang*](https://openreview.net/profile?id=~Zhiqi_Jiang3)

[Yichao Yan]() · [Ming Tao]() · [Shunshun Yin]()

</div>

**SoulX-LiveAct** presents a novel framework that enables **lifelike, multimodal-controlled, high-fidelity** human animation video generation for real-time streaming interactions.

(I) We identify diffusion-step-aligned neighbor latents as a key inductive bias for AR diffusion, providing a principled and theoretically grounded **Neighbor Forcing** for step-consistent AR video generation.

(II) We introduce **ConvKV Memory**, a lightweight plug-in compression mechanism that enables constant-memory hour-scale video generation with negligible overhead.

(III) We develop an optimized real-time system that achieves **20 FPS using only two H100/H200 GPUs** with end-end adaptive FP8 precision, sequence parallelism, and operator fusion at 720×416 or 512×512 resolution.


<div align="center">
  <a href='http://arxiv.org/abs/2603.11746'><img src='https://img.shields.io/badge/Technical-Report-red'></a>
  <a href='https://soul-ailab.github.io/soulx-liveact/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
  <a href='https://github.com/Soul-AILab/SoulX-LiveAct'><img src='https://img.shields.io/badge/Github-Home-blue'></a>
  <a href='https://huggingface.co/Soul-AILab/LiveAct'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow'></a>
</div>


## 🔥🔥🔥 News

* 📢 Mar 18, 2026: We now support consumer GPUs (e.g., RTX 4090, RTX 5090) with FP8 KV cache and CPU model offloading. In our tests, the 18B model (14B Wan2.1 + 4B audio module) achieves a throughput of 6 FPS on a single RTX 5090.
* 👋 Mar 16, 2026: We release the inference code and model weights of SoulX-LiveAct.


## 🎥 Demo

[//]: # (**Note:** Due to GitHub limitations, the videos are heavily compressed. Please refer to the [demo page]&#40;https://demopagedemo.github.io/LiveAct/&#41; for the original results.)

### 👫 Podcast
<table>
  <tr>
    <td><video controls playsinline width="666" src="https://github.com/user-attachments/assets/7d50441c-2a90-48c7-a557-c375936f2b65"></video></td>
  </tr>
</table>


### 🎤 Music & Talk Show
<table>
  <tr>
    <td><video controls playsinline width="360" src="https://github.com/user-attachments/assets/9fd4fbcf-3e76-48ca-a8e0-2a46da18da5c"></video></td>
    <td><video controls playsinline width="360" src="https://github.com/user-attachments/assets/9ac3ad4b-db6a-470b-9f4f-6ab9d1c8d998"></video></td>
  </tr>
</table>

### 📱 FaceTime
<table>
  <tr>
    <td><video controls playsinline width="360" src="https://github.com/user-attachments/assets/143bb565-078a-48ba-8daa-f2fb56616189"></video></td>
    <td><video controls playsinline width="360" src="https://github.com/user-attachments/assets/5619381e-bd8c-4aac-a1d6-2a1fdfe9d673"></video></td>
  </tr>
</table>


## 📑 Open-source Plan

  - [x] Release inference code and checkpoints
  - [x] GUI demo Support
  - [x] End-end adaptive FP8 precision
  - [x] Support model offloading for consumer GPUs (e.g., RTX 4090, RTX 5090) to reduce memory usage
  - [ ] Support FP4 precision for B-series GPUs (e.g., RTX 5090, B100, B200)
  - [ ] Release training code

## ▶️ Quick Start

### 🛠️ Dependencies and Installation

#### Step 1: Install Basic Dependencies

```bash
conda create -n liveact python=3.10
conda activate liveact
pip install -r requirements.txt
conda install conda-forge::sox -y
```

#### Step 2: Install SageAttention
To enable fp8 attention kernel, you need to install SageAttention:
* Install SageAttention:
  ```bash
  git clone https://github.com/thu-ml/SageAttention.git
  cd SageAttention
  git checkout v2.2.0
  python setup.py install
  ```

* (Optional) Install the modified version of SageAttention: 
  To enable SageAttention for QKV's operator fusion, you need to install it by the following command:

  ```bash
  git clone https://github.com/ZhiqiJiang/SageAttentionFusion.git
  cd SageAttentionFusion
  python setup.py install
  ```

#### Step 3: Install vllm:
  To enable fp8 gemm kernel, you need to install vllm:
  ```bash
  pip install vllm==0.11.0
  ```

#### Step 4 Install LightVAE:：

  ```bash
  git clone https://github.com/ModelTC/LightX2V
  cd LightX2V
  python setup_vae.py install
  ```


### 🤗 Download Checkpoints

### Model Cards
| ModelName             | Download                                                                       |
|-----------------------|--------------------------------------------------------------------------------| 
| SoulX-LiveAct         | [🤗 Huggingface](https://huggingface.co/Soul-AILab/LiveAct)                    |
| chinese-wav2vec2-base | [🤗 Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |


### 🔑 Inference

#### Usage of LiveAct

#### 1. Run real-time streaming inference on two H100/H200 GPUs

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=$(shuf -n 1 -i 10000-65535)  \
    generate.py \
    --size 416*720 \
    --ckpt_dir MODEL_PATH \
    --wav2vec_dir chinese-wav2vec2-base \
    --fps 20 \
    --dura_print \
    --input_json examples/example.json \
    --steam_audio
```

#### 2. Run with the best performance settings

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=$(shuf -n 1 -i 10000-65535)  \
    generate.py \
    --size 480*832 \
    --ckpt_dir MODEL_PATH \
    --wav2vec_dir chinese-wav2vec2-base \
    --fps 24 \
    --input_json examples/example.json
```

#### 3. Run with action or emotion editing

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=$(shuf -n 1 -i 10000-65535)  \
    generate.py \
    --size 512*512 \
    --ckpt_dir MODEL_PATH \
    --wav2vec_dir chinese-wav2vec2-base \
    --fps 24 \
    --input_json examples/example_edit.json
```

#### 4. Run on RTX 4090/RTX 5090 GPUs
**Note:** FP8 KV cache may slightly affect generation quality.
```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0 \
python generate.py \
    --size 416*720 \
    --ckpt_dir MODEL_PATH \
    --wav2vec_dir chinese-wav2vec2-base \
    --fps 24 \
    --input_json examples/example.json \
    --fp8_kv_cache \
    --block_offload \
    --t5_cpu
```

#### 5. Run with single GPU for Eval

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0 \
python generate.py \
    --size 480*832 \
    --ckpt_dir MODEL_PATH \
    --wav2vec_dir chinese-wav2vec2-base \
    --fps 24 \
    --input_json examples/example.json \
    --audio_cfg 1.7 \
    --t5_cpu
```


### Command Line Arguments

| Argument          | Type  | Required | Default | Description                                                                                   |
|-------------------|-------|----------|---------|-----------------------------------------------------------------------------------------------|
| `--size`          | str   | Yes      | -       | The width and height of the generated video.                                                  |
| `--t5_cpu`        | bool  | No       | false   | Whether to place T5 model on CPU.                                                             |
| `--offload_cache` | bool  | No       | -       | Whether to place kv cache on CPU.                                                             |
| `--fps`           | int   | Yes      | -       | The target fps  of the generated video.                                                       |
| `--audio_cfg`     | float | No       | 1.0     | Classifier free guidance scale for audio control.                                             |
| `--dura_print`    | bool  | No       | no      | Whether print duration for every block.                                                       |
| `--input_json`    | str   | Yes      | _       | The condition json file path to generate the video.                                           |
| `--seed`          | int   | No       | 42      | The seed to use for generating the image or video.                                            |
| `--steam_audio`   | bool  | No       | false   | Whether inference with steaming audio.                                                        |
| `--mean_memory`   | bool  | No       | false   | Whether to use the mean memory strategy during inference for further performance improvement. |
| `--fp8_kv_cache`   | bool  | No       | false   | Whether to store kv cache in FP8 and dequantize to BF16 on use. FP8 KV cache may slightly affect generation quality.|
| `--block_offload`   | bool  | No       | false   | Whether to offload model blocks to CPU between block forwards.|


### 💻 GUI demo
Run SoulX-LiveAct inference on the GUI demo and evaluate real-time performance.

<div>
  <video controls playsInline src="https://github.com/user-attachments/assets/7150345d-693f-4250-af07-e94daa6ef6ed" width="50%"></video>
</div>

**Note:** The first few blocks during the initial run require warm-up. Normal performance will be observed from the second run onward.

#### 1. Run real-time streaming inference on two H100/H200 GPUs

```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=$(shuf -n 1 -i 10000-65535) \
  demo.py \
  --ckpt_dir MODEL_PATH \
  --wav2vec_dir chinese-wav2vec2-base \
  --size 416*720 \
  --video_save_path ./generated_videos
```

#### 2. Run on RTX 4090/RTX 5090 GPUs
```bash
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 --master_port=$(shuf -n 1 -i 10000-65535) \
  demo.py \
  --ckpt_dir MODEL_PATH \
  --wav2vec_dir chinese-wav2vec2-base \
  --size 416*720 \
  --fp8_kv_cache \
  --block_offload \
  --t5_cpu \
  --video_save_path ./generated_videos
```

## 📚 Citation

```bibtex
@misc{zhen2026soulxliveacthourscalerealtimehuman,
      title={SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV Memory}, 
      author={Dingcheng Zhen and Xu Zheng and Ruixin Zhang and Zhiqi Jiang and Yichao Yan and Ming Tao and Shunshun Yin},
      year={2026},
      eprint={2603.11746},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.11746}, 
}
```
## 📮 Contact Us
If you are interested in leaving a message to our work, feel free to email dingchengzhen@soulapp.cn.

You’re welcome to join our WeChat group or Soul group for technical discussions.
<p align="center">
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/QRCode_WX.jpg" width="200" alt="WeChat Group QR Code"/>
  </span>
  <span style="display: inline-block;">
    <img src="assets/QRCode_Soul.png" width="300" alt="WeChat QR Code"/>
  </span>
</p>