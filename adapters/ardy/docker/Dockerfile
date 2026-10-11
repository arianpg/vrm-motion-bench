# ARDY アダプター（RTX 5090 / Blackwell 対応のため CUDA 12.8 系の PyTorch を使う）
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/ardy/docker/Dockerfile -t vmb-adapter-ardy .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する
ARG ARDY_REPO=https://github.com/nv-tlabs/ardy.git
ARG ARDY_COMMIT=693f74d13b3d04a0a22ce127ee79c929dd89756b
RUN git clone ${ARDY_REPO} /opt/ardy && git -C /opt/ardy checkout ${ARDY_COMMIT}

# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/constraints.txt \
 && cd /opt/ardy \
 && pip install -c /tmp/constraints.txt .  # setup.py が MotionCorrection（C++ 拡張）も一緒にビルドする

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/ardy/server /opt/vmb/adapter
WORKDIR /opt/vmb/adapter

# 重みは /models/ardy/hf（ダウンロード済みならオフラインで使う）
ENV HF_HOME=/models/ardy/hf \
    TEXT_ENCODER_MODE=local \
    LOCAL_CACHE=true \
    PORT=8000
EXPOSE 8000
CMD ["python", "ardy_adapter.py"]
