"""共通コンバーター（Blender 常駐コンテナ）の起動と呼び出し。GPU は使わず、常時起動しておく。"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import docker
import httpx
from docker.errors import NotFound

from .config import Settings
from .containers import LABEL_ROLE, ContainerError

log = logging.getLogger(__name__)

IMAGE = os.environ.get("VMB_CONVERTER_IMAGE", "vmb-converter")
NAME = "vmb-converter"
PORT = "8100/tcp"


class ConverterService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = docker.from_env()
        # 開発時にホストの Blender で動かすワーカーを使う場合（パスはホストのものをそのまま渡す）
        self.base_url: str | None = os.environ.get("VMB_CONVERTER_URL")
        self.external = self.base_url is not None
        self._lock = threading.Lock()  # Blender は 1 件ずつ処理する

    def to_container_path(self, host_path: Path) -> str:
        if self.external:
            return str(Path(host_path).resolve())
        rel = Path(host_path).resolve().relative_to(self.settings.data_dir)
        return f"/data/{rel.as_posix()}"

    def start(self) -> None:
        if self.external:
            return
        try:
            self.client.containers.get(NAME).remove(force=True)
        except NotFound:
            pass
        try:
            container = self.client.containers.run(
                IMAGE,
                name=NAME,
                detach=True,
                labels={LABEL_ROLE: "converter"},
                user=f"{os.getuid()}:{os.getgid()}",
                volumes={str(self.settings.data_dir): {"bind": "/data", "mode": "rw"}},
                ports={PORT: ("127.0.0.1", None)},
            )
        except docker.errors.ImageNotFound:
            raise ContainerError("missing_asset", f"Docker イメージ {IMAGE} がない。ビルドが必要") from None
        container.reload()
        self.base_url = f"http://127.0.0.1:{container.ports[PORT][0]['HostPort']}"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            container.reload()
            if container.status in ("exited", "dead"):
                raise ContainerError("internal", container.logs(tail=40).decode(errors="replace"))
            try:
                if httpx.get(f"{self.base_url}/health", timeout=3).json().get("status") == "ready":
                    log.info("converter ready at %s", self.base_url)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        raise ContainerError("internal", "converter did not become ready")

    def stop(self) -> None:
        if self.external:
            return
        try:
            self.client.containers.get(NAME).remove(force=True)
        except NotFound:
            pass

    def convert(self, source_motion: Path, bone_map: dict[str, str], out_dir: Path, *, preview: bool = True) -> dict:
        if not self.base_url:
            self.start()
        payload = {
            "source_motion": self.to_container_path(source_motion),
            "bone_map": bone_map,
            "out_dir": self.to_container_path(out_dir),
            "preview": preview,
            "bvh": True,
        }
        with self._lock:
            try:
                r = httpx.post(f"{self.base_url}/convert", json=payload, timeout=self.settings.timeouts.convert)
            except httpx.TimeoutException:
                raise ContainerError("internal", "convert timed out") from None
        body = r.json()
        if r.status_code != 200:
            raise ContainerError(body.get("error_code", "internal"), body.get("message", r.text))
        return body

    def preview_group(self, items: list[tuple[Path, dict[str, str], str]], out: Path) -> dict:
        """items: (SourceMotion のパス, ボーン対応表, 色)"""
        if not self.base_url:
            self.start()
        payload = {
            "items": [
                {"source_motion": self.to_container_path(p), "bone_map": bm, "color": color} for p, bm, color in items
            ],
            "out": self.to_container_path(out),
        }
        with self._lock:
            try:
                r = httpx.post(f"{self.base_url}/preview_group", json=payload, timeout=self.settings.timeouts.convert * 2)
            except httpx.TimeoutException:
                raise ContainerError("internal", "group preview timed out") from None
        body = r.json()
        if r.status_code != 200:
            raise ContainerError(body.get("error_code", "internal"), body.get("message", r.text))
        return body
