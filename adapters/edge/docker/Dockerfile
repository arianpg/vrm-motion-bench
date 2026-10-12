# EDGE アダプター（音楽 → SMPL 24 関節のダンス）
# 公式の環境は Python 3.7・torch 1.12.1 だが、RTX 5090（Blackwell）では CUDA 12.8 版が必要なため torch 2.8 を使う。
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/edge/docker/Dockerfile -t vmb-adapter-edge .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリと、音楽の特徴抽出（Jukebox）のライブラリを検証済みのコミットに固定する
ARG EDGE_REPO=https://github.com/Stanford-TML/EDGE.git
ARG EDGE_COMMIT=17c3428669ed6733edd9d8c66f7dc62060b8e46d
ARG JUKEBOX_REPO=https://github.com/rodrigo-castellon/jukebox.git
ARG JUKEBOX_COMMIT=6b4486164e150286d9d6944797a14b19f31912f0
ARG JUKEMIRLIB_REPO=https://github.com/rodrigo-castellon/jukemirlib.git
ARG JUKEMIRLIB_COMMIT=a91d87fcae0dd89085752421e794ea7e1b300735
RUN git clone ${EDGE_REPO} /opt/edge && git -C /opt/edge checkout ${EDGE_COMMIT} \
 && rm -rf /opt/edge/.git /opt/edge/media /opt/edge/SMPL-to-FBX \
 && git clone ${JUKEBOX_REPO} /opt/jukebox && git -C /opt/jukebox checkout ${JUKEBOX_COMMIT} \
 && git clone ${JUKEMIRLIB_REPO} /opt/jukemirlib && git -C /opt/jukemirlib checkout ${JUKEMIRLIB_COMMIT}

# pytorch3d は回転の変換（transforms）だけを使う。純 Python の部分だけを入れ、C++/CUDA 拡張はビルドしない
ARG PYTORCH3D_VERSION=v0.7.9
RUN git clone --depth 1 --branch ${PYTORCH3D_VERSION} https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d \
 && SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
 && mkdir -p $SITE/pytorch3d \
 && cp -r /tmp/pytorch3d/pytorch3d/__init__.py /tmp/pytorch3d/pytorch3d/common /tmp/pytorch3d/pytorch3d/transforms $SITE/pytorch3d/ \
 && rm -rf /tmp/pytorch3d

# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）。
# jukebox・jukemirlib は依存の版が古く固定されているため --no-deps で入れ、必要なものを別に入れる
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/constraints.txt \
 && pip install -c /tmp/constraints.txt \
      "accelerate==1.10.1" "einops==0.8.1" "p_tqdm==1.4.2" "librosa==0.11.0" "soundfile==0.13.1" \
      "matplotlib==3.10.6" "tqdm" "fire==0.7.1" "unidecode==1.4.0" "wget==3.2" "numpy>=1.24,<2" "wandb==0.21.4" \
 && pip install -c /tmp/constraints.txt --no-deps /opt/jukebox /opt/jukemirlib

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/edge/server /opt/vmb/adapter

# EDGE のチェックポイントは /models/edge/checkpoint.pt、Jukebox の重みは /models/edge/jukemirlib から読む。
# どちらも torch.save の pickle（EDGE は正規化の係数のオブジェクトを含む）なので、torch 2.6 以降の weights_only を無効にする
ENV EDGE_DIR=/opt/edge \
    EDGE_CHECKPOINT=/models/edge/checkpoint.pt \
    JUKEBOX_CACHE=/models/edge/jukemirlib \
    WANDB_MODE=disabled \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PORT=8000
EXPOSE 8000
CMD ["python", "/opt/vmb/adapter/edge_adapter.py"]
