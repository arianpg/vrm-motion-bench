# HY-Motion 1.0 アダプター
# 公式は torch 2.5.1 固定だが、RTX 5090（Blackwell）では CUDA 12.8 版が必要なため torch 2.8 に置き換える。
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/hymotion/docker/Dockerfile -t vmb-adapter-hymotion .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する。木製人形モデルや統計量は Git LFS で管理されている
ARG HYMOTION_REPO=https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git
ARG HYMOTION_COMMIT=4e426f5a1021cbcf7f375458c37b840ee7225229
RUN git lfs install \
 && git clone ${HYMOTION_REPO} /opt/hymotion \
 && git -C /opt/hymotion checkout ${HYMOTION_COMMIT} \
 && git -C /opt/hymotion lfs pull \
 && rm -rf /opt/hymotion/.git /opt/hymotion/assets

# 推論に必要なものだけを入れる（FBX 書き出し・プロンプト書き直し・可視化の依存は入れない）。
# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt \
      "transformers==4.53.3" "accelerate==0.30.1" "huggingface_hub==0.30.0" "safetensors==0.5.3" \
      "torchdiffeq==0.2.5" "einops==0.8.1" "numpy>=1.24,<2" "scipy>=1.10" \
      "PyYAML==6.0.1" "omegaconf==2.3.0"

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/hymotion/server /opt/vmb/adapter

# テキストエンコーダ（CLIP-L、Qwen3-8B）は Hugging Face のキャッシュ（/models/hymotion/hf）から読む
ENV HYMOTION_DIR=/opt/hymotion \
    HYMOTION_CKPTS=/models/hymotion/ckpts \
    HF_HOME=/models/hymotion/hf \
    HF_HUB_OFFLINE=1 \
    USE_HF_MODELS=1 \
    PORT=8000
EXPOSE 8000
CMD ["python", "/opt/vmb/adapter/hymotion_adapter.py"]
