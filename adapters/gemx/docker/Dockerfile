# GEM-X アダプター（動画 → SOMA77 モーション）
# ビルドコンテキストはリポジトリのルート:
#   docker build -f adapters/gemx/docker/Dockerfile -t vmb-adapter-gemx .
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

# PyTorch 2.10 のイメージはシステムの Python への pip を禁止している（PEP 668）。コンテナ専用なので許可する
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs ca-certificates ffmpeg build-essential libgl1 libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# 公式リポジトリの検証済みコミットに固定する。サブモジュールは SOMA と SAM-3D-Body だけ使う
# （soma-retargeter は SSH が必要で、ロボットへのリターゲット用なので使わない）
ARG GEMX_REPO=https://github.com/NVlabs/GEM-X.git
ARG GEMX_COMMIT=32992550dba114c62243fb55e361311972dce8f9
RUN git lfs install \
 && git clone ${GEMX_REPO} /opt/gemx && cd /opt/gemx && git checkout ${GEMX_COMMIT} \
 && git submodule update --init third_party/soma third_party/sam-3d-body \
 && cd third_party/soma && git lfs pull

# torch はベースイメージのものを使う（pip が CUDA 版を差し替えないよう constraints で固定）
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import torch, torchvision; print(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}')" > /tmp/constraints.txt \
 && cd /opt/gemx \
 && pip install -c /tmp/constraints.txt -e third_party/soma \
 && pip install -c /tmp/constraints.txt -e . \
 && pip install -c /tmp/constraints.txt cloudpickle fvcore iopath pycocotools braceexpand roma "setuptools<75" "onnxruntime-gpu==1.22.0" \
 && TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 pip install -c /tmp/constraints.txt --no-build-isolation --no-deps \
      "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9"  # RTX 5090（sm_120）向けだけにビルドする

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt /opt/vmb/packages/sourcemotion \
 && pip install -c /tmp/constraints.txt --no-deps /opt/vmb/packages/adapter_kit \
 && pip install -c /tmp/constraints.txt "fastapi>=0.115" "uvicorn>=0.30"

COPY adapters/gemx/server /opt/vmb/adapter

# inputs/: 重みはマウントした /models/gemx/inputs を参照する。SOMA のアセットはサブモジュール同梱版（Git LFS）を使う
# （inputs/soma_assets がないと Hugging Face から最新版を取得するが、固定したコミットのコードと版が合わない）
RUN mkdir -p /opt/gemx/inputs \
 && for d in pretrained checkpoints mhr_data soma_data; do ln -s /models/gemx/inputs/$d /opt/gemx/inputs/$d; done \
 && ln -s /opt/gemx/third_party/soma/assets /opt/gemx/inputs/soma_assets

# YOLOX（rtmlib）や DINOv3（torch.hub）は初回にダウンロードされるので、保存先をマウントした /models/gemx に置く
ENV GEMX_DIR=/opt/gemx \
    HF_HOME=/models/gemx/hf \
    TORCH_HOME=/models/gemx/cache/torch \
    GEMX_HOME=/models/gemx/home \
    PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=surfaceless \
    PORT=8000
EXPOSE 8000
CMD ["python", "/opt/vmb/adapter/gemx_adapter.py"]
