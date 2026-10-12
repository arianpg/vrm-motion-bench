# EMAGE アダプター（音声 → SMPL-X＋FLAME の全身の身振り）
# 公式の環境は Python 3.9・torch 2.0 だが、RTX 5090（Blackwell）では CUDA 12.8 版が必要なため torch 2.8 を使う。
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/emage/docker/Dockerfile -t vmb-adapter-emage .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する。推論に使うのは models/emage_audio だけ
ARG PANTOMATRIX_REPO=https://github.com/PantoMatrix/PantoMatrix.git
ARG PANTOMATRIX_COMMIT=c7356f35f8e39e469e510ccd1bf37e44adf8ec0e
RUN git clone ${PANTOMATRIX_REPO} /opt/pantomatrix \
 && git -C /opt/pantomatrix checkout ${PANTOMATRIX_COMMIT} \
 && rm -rf /opt/pantomatrix/.git /opt/pantomatrix/assets /opt/pantomatrix/examples

# EMAGE のモデルは torch・transformers・omegaconf だけで動く（smplx・librosa・描画の依存は入れない）。
# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt \
      "transformers==4.53.3" "huggingface_hub==0.34.4" "safetensors==0.5.3" "omegaconf==2.3.0" "numpy>=1.24,<2"

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/emage/server /opt/vmb/adapter

# 重みは Hugging Face のキャッシュ（/models/emage/hf）、SMPL-X のモデルは /models/emage/evaltools から読む
ENV PANTOMATRIX_DIR=/opt/pantomatrix \
    EMAGE_SMPLX=/models/emage/evaltools/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz \
    HF_HOME=/models/emage/hf \
    HF_HUB_OFFLINE=1 \
    PORT=8000
EXPOSE 8000
CMD ["python", "/opt/vmb/adapter/emage_adapter.py"]
