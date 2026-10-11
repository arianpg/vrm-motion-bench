# MoMask アダプター（CPU 実行）
# 公式は PyTorch 1.7/1.12 + CUDA 11 前提だが、CPU で動かすため新しい PyTorch（CPU 版）に置き換える。
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/momask/docker/Dockerfile -t vmb-adapter-momask .
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する
ARG MOMASK_REPO=https://github.com/EricGuo5513/momask-codes.git
ARG MOMASK_COMMIT=94a6636c9c463b7a9414c3401a6f1b67e6c51824
RUN git clone ${MOMASK_REPO} /opt/momask && git -C /opt/momask checkout ${MOMASK_COMMIT}

# 推論に必要なものだけを入れる（学習・評価用の依存は入れない）。numpy は付属コードに合わせて 1.23 系
# CLIP の依存（torchvision）が CUDA 版の torch を引き込まないよう、CPU 版を constraints で固定する
RUN --mount=type=cache,target=/root/.cache/pip \
    printf 'torch==2.8.0+cpu\ntorchvision==0.23.0+cpu\n' > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install -c /tmp/constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu \
      "numpy<1.24" scipy "matplotlib<3.8" einops tqdm ftfy regex \
      "vector-quantize-pytorch==1.6.30" "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /opt/vmb/packages/sourcemotion \
 && pip install --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30" "pydantic>=2"

COPY adapters/momask/server /opt/vmb/adapter

ENV MOMASK_DIR=/opt/momask \
    MOMASK_CHECKPOINTS=/models/momask/checkpoints \
    MOMASK_CLIP_DIR=/models/momask/clip \
    PORT=8000
EXPOSE 8000
CMD ["python", "/opt/vmb/adapter/momask_adapter.py"]
