"""オーケストレーターの起動。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from .api import build_router
from .config import Settings, load_settings
from .containers import ContainerManager
from .converter import ConverterService
from .db import Database
from .jobs import EventBus, JobQueue
from .llm import PromptService, load_llm_config
from .manifest import Manifest, load_manifests
from .pipeline import Pipeline

log = logging.getLogger("vmb_backend")


@dataclass
class AppState:
    settings: Settings
    db: Database
    bus: EventBus
    queue: JobQueue
    manifests: dict[str, Manifest]
    containers: ContainerManager
    converter: ConverterService
    pipeline: Pipeline
    prompts: PromptService


def create_app() -> FastAPI:
    settings = load_settings()
    db = Database(settings.db_path)
    bus = EventBus()
    manifests = load_manifests(settings.adapters_dir)
    containers = ContainerManager(settings)
    converter = ConverterService(settings)
    queue = JobQueue(
        bus, loaded_model=lambda: containers.current.key if containers.current else None
    )
    prompts = PromptService(settings, load_llm_config(settings.llm_config), containers)
    pipeline = Pipeline(settings, db, bus, queue, manifests, containers, converter, prompts)
    state = AppState(settings, db, bus, queue, manifests, containers, converter, pipeline, prompts)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        interrupted = db.mark_interrupted()
        removed = containers.cleanup_orphans()
        log.info("startup: %d interrupted results, removed containers %s", interrupted, removed)
        try:
            converter.start()
        except Exception as e:  # noqa: BLE001
            log.error("converter failed to start: %s", e)  # 変換時に再試行する
        queue.start()
        yield
        queue.stop()
        containers.stop_current()
        pipeline.cpu_services.stop_all()
        converter.stop()

    app = FastAPI(title="VRM Motion Bench", lifespan=lifespan)
    app.state.vmb = state
    app.include_router(build_router(state))
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    uvicorn.run(create_app(), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
