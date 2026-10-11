# Kimodo アダプター（RTX 5090 / Blackwell 対応のため CUDA 12.8 系の PyTorch を使う）
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/kimodo/docker/Dockerfile -t vmb-adapter-kimodo .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する
ARG KIMODO_REPO=https://github.com/nv-tlabs/kimodo.git
ARG KIMODO_COMMIT=58e781898b3d7e328a676a75d3e338c45dce3ad9
RUN git clone ${KIMODO_REPO} /opt/kimodo && git -C /opt/kimodo checkout ${KIMODO_COMMIT}

# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/constraints.txt \
 && cd /opt/kimodo \
 && SKIP_MOTION_CORRECTION_IN_SETUP=1 pip install -c /tmp/constraints.txt ./MotionCorrection \
 && SKIP_MOTION_CORRECTION_IN_SETUP=1 pip install -c /tmp/constraints.txt -e . \
 && pip install -c /tmp/constraints.txt "py-soma-x @ git+https://github.com/NVlabs/SOMA-X.git"

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/kimodo/server /opt/vmb/adapter
WORKDIR /opt/vmb/adapter

# 重みは /models/kimodo/hf（読み取り専用でも動くよう、ダウンロード済みならオフラインで使う）
ENV HF_HOME=/models/kimodo/hf \
    TEXT_ENCODER_MODE=local \
    LOCAL_CACHE=true \
    PORT=8000
EXPOSE 8000
CMD ["python", "kimodo_adapter.py"]
