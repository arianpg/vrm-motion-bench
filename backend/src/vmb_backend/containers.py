"""アダプターコンテナの起動・停止（＝ロード・アンロード）と GPU メモリ解放の確認。"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import docker
import httpx
from docker.errors import NotFound
from docker.types import DeviceRequest

from .config import Settings
from .manifest import Manifest

log = logging.getLogger(__name__)

LABEL_ROLE = "vmb.role"
LABEL_ADAPTER = "vmb.adapter"
CONTAINER_PORT = "8000/tcp"


class ContainerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ServiceSpec:
    """GPU キューで入れ替えるサービス（アダプター、LLM）の起動方法。"""

    key: str  # キューの model_key と同じ（アダプター ID、または "llm:<名前>"）
    image: str
    role: str  # "adapter" | "llm"
    device: str = "gpu"
    command: list[str] | None = None
    environment: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    port: str = CONTAINER_PORT
    health: str = "adapter"  # "adapter": {"status": "ready"} を返す / "llama": llama-server の /health
    startup_timeout: float = 600.0
    shm_size: str | None = None


@dataclass
class RunningService:
    key: str
    container_id: str
    base_url: str
    load_seconds: float
    gpu_baseline_mib: int | None


class GpuMonitor:
    """NVML で GPU メモリの使用量を読む。NVML が使えない環境では None を返す。"""

    def __init__(self, index: int) -> None:
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as e:  # noqa: BLE001
            log.warning("NVML unavailable: %s", e)

    def used_mib(self) -> int | None:
        if self._handle is None:
            return None
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used // (1024 * 1024))

    def total_mib(self) -> int | None:
        if self._handle is None:
            return None
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).total // (1024 * 1024))


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    path = Path.home() / ".cache/huggingface/token"
    return path.read_text().strip() if path.exists() else None


class ContainerManager:
    """GPU アダプターは同時に 1 つだけ動かす。CPU アダプターはこのクラスの外で別に扱う。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = docker.from_env()
        self.gpu = GpuMonitor(settings.gpu_index)
        self.current: RunningService | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------ 起動時の掃除

    def cleanup_orphans(self) -> list[str]:
        """前回の実行で残ったアダプターコンテナを止める（GPU メモリを掴んだまま残さない）。"""
        removed = []
        for c in self.client.containers.list(all=True, filters={"label": f"{LABEL_ROLE}=adapter"}):
            log.info("removing orphan adapter container %s", c.name)
            try:
                c.stop(timeout=self.settings.timeouts.stop_grace)
            except Exception:  # noqa: BLE001
                pass
            try:
                c.remove(force=True)
            except NotFound:
                pass
            removed.append(c.name)
        return removed

    # ------------------------------------------------------------ ロード

    def adapter_spec(self, manifest: Manifest) -> ServiceSpec:
        s = self.settings
        model_dir = s.models_dir / manifest.id
        model_dir.mkdir(parents=True, exist_ok=True)
        env = {"HOME": "/tmp"}
        if token := _hf_token():
            env["HF_TOKEN"] = token
        timeout = manifest.resources.timeouts.get("startup") or (
            manifest.resources.load_time_s * s.timeouts.startup_factor
            if manifest.resources.load_time_s
            else s.timeouts.startup_default
        )
        return ServiceSpec(
            key=manifest.id,
            image=manifest.resources.container_image,
            role="adapter",
            device=manifest.device,
            environment=env,
            volumes={
                str(s.data_dir): {"bind": "/data", "mode": "rw"},
                str(model_dir): {"bind": f"/models/{manifest.id}", "mode": "rw"},
            },
            startup_timeout=timeout,
            shm_size="8g",
        )

    def ensure(self, spec: ServiceSpec, on_status: Callable[[str], None] | None = None) -> RunningService:
        """spec のサービスを ready 状態にする。別のサービスが動いていれば先に止める（＝アンロード）。"""
        with self._lock:
            if self.current and self.current.key == spec.key:
                if self._is_healthy(self.current, spec):
                    return self.current
                log.warning("service %s is not healthy; restarting", spec.key)
            if self.current:
                self.stop_current()
            self.current = self._start(spec, on_status or (lambda _s: None))
            return self.current

    def _health(self, base_url: str, spec: ServiceSpec) -> tuple[str, str | None]:
        """("ready" | "loading" | "error", エラー内容)"""
        r = httpx.get(f"{base_url}/health", timeout=5)
        if spec.health == "llama":
            # llama-server: 読み込み中は 503、完了すると 200 {"status": "ok"}
            return ("ready", None) if r.status_code == 200 else ("loading", None)
        body = r.json()
        status = body.get("status")
        if status == "ready":
            return "ready", None
        if status == "error":
            return "error", str(body.get("error") or "")
        return "loading", None

    def _is_healthy(self, running: RunningService, spec: ServiceSpec) -> bool:
        try:
            return self._health(running.base_url, spec)[0] == "ready"
        except (httpx.HTTPError, ValueError):
            return False

    def _start(self, spec: ServiceSpec, on_status: Callable[[str], None]) -> RunningService:
        s = self.settings
        name = f"vmb-{spec.role}-{spec.key.replace(':', '-')}"
        try:
            self.client.containers.get(name).remove(force=True)
        except NotFound:
            pass
        kwargs: dict = {}
        if spec.device == "gpu":
            kwargs["device_requests"] = [DeviceRequest(device_ids=[str(s.gpu_index)], capabilities=[["gpu"]])]
        if spec.command:
            kwargs["command"] = spec.command
        if spec.shm_size:
            kwargs["shm_size"] = spec.shm_size

        baseline = self.gpu.used_mib() if spec.device == "gpu" else None
        on_status("loading")
        started = time.monotonic()
        try:
            container = self.client.containers.run(
                spec.image,
                name=name,
                detach=True,
                # 役割に依らず "adapter" ラベルを付け、起動時の掃除の対象にする
                labels={LABEL_ROLE: "adapter", LABEL_ADAPTER: spec.key},
                environment=spec.environment,
                user=f"{os.getuid()}:{os.getgid()}",
                volumes=spec.volumes,
                ports={spec.port: ("127.0.0.1", None)},
                **kwargs,
            )
        except docker.errors.ImageNotFound:
            raise ContainerError("missing_asset", f"Docker イメージ {spec.image} がない。ビルドが必要") from None

        container.reload()
        host_port = container.ports[spec.port][0]["HostPort"]
        running = RunningService(spec.key, container.id, f"http://127.0.0.1:{host_port}", 0.0, baseline)

        deadline = started + spec.startup_timeout
        while True:
            container.reload()
            if container.status in ("exited", "dead"):
                logs = container.logs(tail=40).decode(errors="replace")
                self._remove(container)
                raise ContainerError("internal", f"container exited during startup:\n{logs}")
            try:
                status, err = self._health(running.base_url, spec)
                if status == "ready":
                    break
                if status == "error":
                    self._remove(container)
                    code = err.split(":", 1)[0] if err and ":" in err else "internal"
                    raise ContainerError(code, f"{spec.key} failed to load: {err}")
            except (httpx.HTTPError, ValueError):
                pass  # サーバー起動前
            if time.monotonic() > deadline:
                logs = container.logs(tail=40).decode(errors="replace")
                self._remove(container)
                raise ContainerError("internal", f"startup timed out after {spec.startup_timeout:.0f}s:\n{logs}")
            time.sleep(s.timeouts.health_interval)

        running.load_seconds = time.monotonic() - started
        log.info("service %s ready in %.1fs (%s)", spec.key, running.load_seconds, running.base_url)
        return running

    # ------------------------------------------------------------ アンロード

    def stop_current(self) -> None:
        with self._lock:
            running, self.current = self.current, None
        if running is None:
            return
        try:
            container = self.client.containers.get(running.container_id)
            self._remove(container)
        except NotFound:
            pass
        if running.gpu_baseline_mib is not None:
            self.wait_gpu_released(running.gpu_baseline_mib)

    def _remove(self, container) -> None:
        try:
            container.stop(timeout=self.settings.timeouts.stop_grace)
        except Exception:  # noqa: BLE001
            pass
        try:
            container.remove(force=True)
        except NotFound:
            pass

    def wait_gpu_released(self, baseline_mib: int) -> bool:
        """使用量が起動前の水準（＋余裕）に戻るまで待つ。戻らなければ警告して続行する。"""
        limit = baseline_mib + self.settings.gpu_release_margin_mib
        deadline = time.monotonic() + self.settings.timeouts.gpu_release
        while True:
            used = self.gpu.used_mib()
            if used is None or used <= limit:
                log.info("GPU memory released: %s MiB (limit %s MiB)", used, limit)
                return True
            if time.monotonic() > deadline:
                log.warning("GPU memory not released: %s MiB > %s MiB", used, limit)
                return False
            time.sleep(1.0)

    # ------------------------------------------------------------ 推論

    def generate(self, running: RunningService, payload: dict, timeout: float, on_progress: Callable[[float], None]) -> dict:
        stop = threading.Event()

        def poll() -> None:
            while not stop.wait(2.0):
                try:
                    p = httpx.get(f"{running.base_url}/progress", timeout=5).json().get("progress")
                    if p is not None:
                        on_progress(float(p))
                except (httpx.HTTPError, ValueError):
                    pass

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        try:
            r = httpx.post(f"{running.base_url}/generate", json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise ContainerError("internal", f"generate timed out after {timeout:.0f}s") from None
        finally:
            stop.set()
        body = r.json()
        if r.status_code != 200:
            raise ContainerError(body.get("error_code", "internal"), body.get("message", r.text))
        return body


class CpuServicePool:
    """CPU で動くアダプター。GPU キューとは別に、起動したままにして並行して使う。"""

    def __init__(self, manager: ContainerManager) -> None:
        self.manager = manager
        self.running: dict[str, RunningService] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def ensure(self, spec: ServiceSpec, on_status: Callable[[str], None] | None = None) -> RunningService:
        with self._lock:
            lock = self._locks.setdefault(spec.key, threading.Lock())
        with lock:
            current = self.running.get(spec.key)
            if current and self.manager._is_healthy(current, spec):
                return current
            if current:
                self._stop(current)
            started = self.manager._start(spec, on_status or (lambda _s: None))
            self.running[spec.key] = started
            return started

    def _stop(self, running: RunningService) -> None:
        try:
            self.manager._remove(self.manager.client.containers.get(running.container_id))
        except NotFound:
            pass

    def stop_all(self) -> None:
        for running in list(self.running.values()):
            self._stop(running)
        self.running.clear()
