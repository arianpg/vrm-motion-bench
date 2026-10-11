# HunyuanVideo 1.5 アダプター（動画生成）
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/hyvideo/docker/Dockerfile -t vmb-adapter-hyvideo .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する
ARG HYVIDEO_REPO=https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git
ARG HYVIDEO_COMMIT=60783e704160023913bee78f0b47036d393d4dfa
RUN git clone ${HYVIDEO_REPO} /opt/hyvideo && git -C /opt/hyvideo checkout ${HYVIDEO_COMMIT} && rm -rf /opt/hyvideo/.git

# 推論に必要なものだけを入れる（量子化・クラウドのプロンプト書き直し・学習の依存は入れない）。
# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt \
      "diffusers==0.35.0" "transformers[accelerate]==4.57.1" "peft==0.17.0" "huggingface-hub==0.34.0" \
      "einops==0.8.0" "loguru==0.7.3" "numpy==1.26.4" "pillow==11.3.0" "imageio==2.37.0" "imageio-ffmpeg==0.6.0" \
      "omegaconf>=2.3.0" "safetensors>=0.5.3" "qwen-vl-utils==0.0.8" psutil webcolors openai tiktoken

# flash-attn（任意。なければ PyTorch 標準の注意機構で動く）。torch 2.8 / CUDA 12 のビルド済みを使う
RUN pip install -c /tmp/constraints.txt \
      https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
    || echo "flash-attn unavailable; falling back to torch attention"

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/hyvideo/server /opt/vmb/adapter

ENV HYVIDEO_DIR=/opt/hyvideo \
    HYVIDEO_CKPTS=/models/hyvideo/ckpts \
    HF_HOME=/models/hyvideo/hf \
    HF_HUB_OFFLINE=1 \
    PORT=8000
EXPOSE 8000
CMD ["python", "/opt/vmb/adapter/hyvideo_adapter.py"]
