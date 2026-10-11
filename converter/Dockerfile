# 共通コンバーター: Blender 公式ビルド（numpy 同梱）＋ VRM Add-on for Blender をバージョン固定で入れる
# ビルドコンテキストはリポジトリのルート:
#   docker build -f converter/Dockerfile -t vmb-converter .
FROM ubuntu:24.04

ARG BLENDER_VERSION=5.0.1
ARG BLENDER_SERIES=5.0
ARG VRM_ADDON_VERSION=4.7.2

ENV DEBIAN_FRONTEND=noninteractive
# GPU なしでの描画は Mesa（llvmpipe）の EGL を使う
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl xz-utils ffmpeg \
      libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglx-mesa0 libgbm1 \
      libx11-6 libxext6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 \
      libwayland-client0 libwayland-egl1 libdecor-0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
      | tar -xJ -C /opt \
 && mv /opt/blender-${BLENDER_VERSION}-linux-x64 /opt/blender \
 && ln -s /opt/blender/blender /usr/local/bin/blender

# ユーザー設定（拡張機能）は実行ユーザーに依らない場所に置く
ENV BLENDER_USER_RESOURCES=/opt/blender-user
RUN curl -fsSL -o /tmp/vrm.zip \
      https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/v${VRM_ADDON_VERSION}/VRM_Addon_for_Blender-Extension-$(echo ${VRM_ADDON_VERSION} | tr . _).zip \
 && blender -b --factory-startup --command extension install-file -r user_default -e /tmp/vrm.zip \
 && rm /tmp/vrm.zip \
 && chmod -R a+rwX /opt/blender-user

COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
RUN PY=$(ls /opt/blender/${BLENDER_SERIES}/python/bin/python3*| head -1) \
 && $PY -m ensurepip \
 && $PY -m pip install --no-deps --no-cache-dir /opt/vmb/packages/sourcemotion

COPY converter/worker /opt/vmb/converter
COPY assets/stickman/stickman.vrm /opt/vmb/assets/stickman.vrm

ENV HOME=/tmp
EXPOSE 8100
CMD ["blender", "-b", "--python", "/opt/vmb/converter/blender_worker.py", "--", \
     "--stickman", "/opt/vmb/assets/stickman.vrm", "--port", "8100"]
