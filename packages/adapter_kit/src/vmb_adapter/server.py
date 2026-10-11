"""アダプター共通の HTTP サーバー（仕様: docs/spec.md「アダプターAPI」）。

各アダプターは `Adapter` を実装し、`create_app(adapter)` を uvicorn で起動する。

    GET  /health    -> {"status": "starting" | "loading" | "ready" | "error", ...}
    POST /generate  -> 完了まで応答しない。同時に 1 件だけ実行する
    GET  /progress  -> {"job_id": ..., "progress": 0.0-1.0 | null}
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("vmb_adapter")

ErrorCode = Literal["missing_asset", "invalid_input", "oom", "busy", "not_ready", "internal"]
_STATUS_BY_CODE = {
    "missing_asset": 409,
    "invalid_input": 422,
    "oom": 507,
    "busy": 409,
    "not_ready": 503,
    "internal": 500,
}


class AdapterError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class GenerateRequest(BaseModel):
    job_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    output_dir: str  # コンテナ内のパス（/data/jobs/{job_id}/output）


class GenerateResult(BaseModel):
    # モーションのアダプターは SourceMotion を返す。動画生成のアダプター（kind: video）は None で、
    # 生成した動画を outputs["video"] に入れる
    source_motion: str | None = None  # output_dir からの相対パス
    raw_outputs: list[str]  # output_dir からの相対パス
    outputs: dict[str, str] = Field(default_factory=dict)  # 種類 → output_dir からの相対パス
    fps: float
    num_frames: int
    duration: float
    seed: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


ProgressCallback = Callable[[float], None]


class Adapter(ABC):
    """モデル固有の処理。load() はサーバー起動直後にバックグラウンドで 1 回だけ呼ばれる。"""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult: ...

    def is_oom(self, error: BaseException) -> bool:
        """GPU のメモリ不足かどうか。torch を使うアダプターは上書きする。"""
        return "out of memory" in str(error).lower()


class _State:
    def __init__(self) -> None:
        self.status = "starting"
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.lock = threading.Lock()
        self.job_id: str | None = None
        self.progress: float | None = None


def create_app(adapter: Adapter) -> FastAPI:
    state = _State()

    def load() -> None:
        state.status = "loading"
        started = time.monotonic()
        try:
            adapter.load()
        except AdapterError as e:
            state.status, state.error = "error", f"{e.code}: {e.message}"
            log.error("load failed: %s", state.error)
            return
        except Exception as e:  # noqa: BLE001
            state.status, state.error = "error", f"internal: {e}"
            log.error("load failed\n%s", traceback.format_exc())
            return
        state.load_seconds = time.monotonic() - started
        state.status = "ready"
        log.info("model loaded in %.1fs", state.load_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        threading.Thread(target=load, name="adapter-load", daemon=True).start()
        yield

    app = FastAPI(title="VRM Motion Bench adapter", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": state.status, "error": state.error, "load_seconds": state.load_seconds}

    @app.get("/progress")
    def progress() -> dict[str, Any]:
        return {"job_id": state.job_id, "progress": state.progress}

    def error(code: ErrorCode, message: str) -> JSONResponse:
        return JSONResponse(status_code=_STATUS_BY_CODE[code], content={"error_code": code, "message": message})

    @app.post("/generate")
    def generate(request: GenerateRequest) -> Any:
        # def（非 async）なので FastAPI のスレッドプールで実行される
        if state.status != "ready":
            return error("not_ready", f"adapter is {state.status}" + (f": {state.error}" if state.error else ""))
        if not state.lock.acquire(blocking=False):
            return error("busy", f"another job is running: {state.job_id}")
        try:
            state.job_id, state.progress = request.job_id, 0.0
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            def report(value: float) -> None:
                state.progress = max(0.0, min(1.0, float(value)))

            started = time.monotonic()
            result = adapter.generate(request, output_dir, report)
            result.extra.setdefault("generate_seconds", time.monotonic() - started)
            return result
        except AdapterError as e:
            return error(e.code, e.message)
        except Exception as e:  # noqa: BLE001
            log.error("generate failed\n%s", traceback.format_exc())
            if adapter.is_oom(e):
                return error("oom", str(e))
            return error("internal", f"{type(e).__name__}: {e}")
        finally:
            state.job_id, state.progress = None, None
            state.lock.release()

    return app


def serve(adapter: Adapter, host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(adapter), host=host, port=port, log_level="info")
