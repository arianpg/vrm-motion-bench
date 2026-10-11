"""ジョブキュー（仕様: docs/spec.md「ジョブ・キュー・モデル切り替え」）。

- GPU ジョブは 1 本のワーカーで直列に実行する。待機中のジョブは、いまロード中のモデルと同じものを優先し、
  入れ替え回数を減らす。依存関係が満たされていないジョブは選ばない。
- CPU ジョブ（変換、プレビュー、CPU アダプター）は別のワーカー群で並行に実行する。
- キューは永続化しない。
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

JobType = Literal["llm_prompt", "motion", "convert", "group_preview"]
JobStatus = Literal["queued", "loading", "running", "done", "failed"]
Resource = Literal["gpu", "cpu"]

_seq = itertools.count()


@dataclass
class Job:
    id: str
    type: JobType
    resource: Resource
    model_key: str | None = None  # GPU で使うモデル（アダプター ID や LLM 名）
    result_id: str | None = None
    group_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = "queued"
    progress: float | None = None
    error: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    seq: int = field(default_factory=lambda: next(_seq))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "resource": self.resource,
            "model_key": self.model_key,
            "result_id": self.result_id,
            "group_id": self.group_id,
            "depends_on": self.depends_on,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
        }


class EventBus:
    """スレッドから asyncio の購読者（SSE）へイベントを配る。"""

    def __init__(self) -> None:
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.add((asyncio.get_running_loop(), queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = {(l, q) for l, q in self._subscribers if q is not queue}

    def publish(self, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(_put_nowait, queue, {"event": event, "data": data})
            except RuntimeError:
                pass  # ループが閉じている


def _put_nowait(queue: asyncio.Queue, item: dict) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


Handler = Callable[[Job], None]


class JobQueue:
    def __init__(self, bus: EventBus, cpu_workers: int = 2, loaded_model: Callable[[], str | None] = lambda: None):
        self.bus = bus
        self.jobs: dict[str, Job] = {}
        self.handlers: dict[JobType, Handler] = {}
        self._cond = threading.Condition()
        self._loaded_model = loaded_model
        self._running = False
        self._threads: list[threading.Thread] = []
        self._cpu_workers = cpu_workers

    def register(self, job_type: JobType, handler: Handler) -> None:
        self.handlers[job_type] = handler

    def start(self) -> None:
        self._running = True
        specs = [("gpu", 0)] + [("cpu", i) for i in range(self._cpu_workers)]
        for resource, i in specs:
            t = threading.Thread(target=self._worker, args=(resource,), name=f"{resource}-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()

    # ------------------------------------------------------------ 投入・更新

    def submit(self, job: Job) -> Job:
        with self._cond:
            self.jobs[job.id] = job
            self._cond.notify_all()
        self._publish(job)
        return job

    def update(self, job: Job, **fields: Any) -> None:
        with self._cond:
            for key, value in fields.items():
                setattr(job, key, value)
        self._publish(job)

    def _publish(self, job: Job) -> None:
        self.bus.publish("job", job.to_dict())

    def list(self, include_finished: bool = True) -> list[Job]:
        with self._cond:
            jobs = sorted(self.jobs.values(), key=lambda j: j.seq)
        return [j for j in jobs if include_finished or j.status not in ("done", "failed")]

    # ------------------------------------------------------------ 選択

    def _deps_state(self, job: Job) -> Literal["ready", "waiting", "failed"]:
        for dep_id in job.depends_on:
            dep = self.jobs.get(dep_id)
            if dep is None or dep.status == "failed":
                return "failed"
            if dep.status != "done":
                return "waiting"
        return "ready"

    def _pick(self, resource: Resource) -> Job | None:
        """呼び出し側で self._cond を保持していること。"""
        candidates: list[Job] = []
        for job in sorted(self.jobs.values(), key=lambda j: j.seq):
            if job.resource != resource or job.status != "queued":
                continue
            state = self._deps_state(job)
            if state == "failed":
                job.status, job.error, job.finished = "failed", "dependency failed", time.time()
                self._publish(job)
                continue
            if state == "ready":
                candidates.append(job)
        if not candidates:
            return None
        if resource == "gpu":
            loaded = self._loaded_model()
            same = [j for j in candidates if loaded is not None and j.model_key == loaded]
            if same:
                return same[0]
        return candidates[0]

    def _worker(self, resource: Resource) -> None:
        while True:
            with self._cond:
                job = None
                while self._running and (job := self._pick(resource)) is None:
                    self._cond.wait(timeout=1.0)
                if not self._running:
                    return
                job.status, job.started = "running", time.time()
            self._publish(job)
            handler = self.handlers.get(job.type)
            try:
                if handler is None:
                    raise RuntimeError(f"no handler for job type {job.type}")
                handler(job)
                self.update(job, status="done", progress=1.0, finished=time.time())
            except Exception as e:  # noqa: BLE001
                log.error("job %s (%s) failed\n%s", job.id, job.type, traceback.format_exc())
                self.update(job, status="failed", error=str(e), finished=time.time())
            with self._cond:
                self._cond.notify_all()
