# MotionStreamer アダプター（テキスト → SMPL-X の体 22 関節。272 次元の表現から回転を直接取り出す）
# 公式の環境は Python 3.8・CUDA 10.1 だが、RTX 5090（Blackwell）では CUDA 12.8 版が必要なため torch 2.8 を使う。
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/motionstreamer/docker/Dockerfile -t vmb-adapter-motionstreamer .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する。推論に使うのは models/ と utils/ と終わりの判定用の潜在ベクトル
ARG MOTIONSTREAMER_REPO=https://github.com/zju3dv/MotionStreamer.git
ARG MOTIONSTREAMER_COMMIT=8aace3f6f5f564ae1518aa77d925ae1380070c80
RUN git clone ${MOTIONSTREAMER_REPO} /opt/motionstreamer \
 && git -C /opt/motionstreamer checkout ${MOTIONSTREAMER_COMMIT} \
 && rm -rf /opt/motionstreamer/.git /opt/motionstreamer/assets

# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt \
      "transformers==4.46.3" "sentence-transformers==3.2.1" "sentencepiece==0.2.0" "timm==1.0.12" \
      "numpy>=1.24,<2" "typing_extensions"

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/motionstreamer/server /opt/vmb/adapter

# 重み（Causal TAE・生成モデル・正規化の統計量）は /models/motionstreamer/ckpts、
# テキストエンコーダ（sentence-T5-XXL）は /models/motionstreamer/sentencet5-xxl から読む
ENV MOTIONSTREAMER_DIR=/opt/motionstreamer \
    MOTIONSTREAMER_CKPTS=/models/motionstreamer/ckpts \
    MOTIONSTREAMER_T5=/models/motionstreamer/sentencet5-xxl \
    HF_HUB_OFFLINE=1 \
    PORT=8000
EXPOSE 8000
CMD ["python", "/opt/vmb/adapter/motionstreamer_adapter.py"]
