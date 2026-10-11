"""ジョブの中身: motion（推論）→ convert（VRMA 変換とプレビュー）。

結果 1 件ごとの表示上の状態: queued → loading → running → converting → done／failed
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .containers import ContainerError, ContainerManager, CpuServicePool
from .converter import ConverterService
from .db import Database, new_id, now_iso
from .jobs import EventBus, Job, JobQueue
from .llm import PromptService, generate_prompt, llm_spec, model_key
from .manifest import Manifest, check_assets, load_bone_map
from .uploads import UploadError, resolve_upload

log = logging.getLogger(__name__)

# グループプレビューの色（フロントエンドの比較ビューと同じ順）
GROUP_COLORS = ["#5b9cf5", "#ef6b6b", "#4cc38a", "#e5b450", "#b07cf5", "#4cc3c3", "#f58fc8", "#a0a8b5"]


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        bus: EventBus,
        queue: JobQueue,
        manifests: dict[str, Manifest],
        containers: ContainerManager,
        converter: ConverterService,
        prompts: PromptService,
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.queue = queue
        self.manifests = manifests
        self.containers = containers
        self.converter = converter
        self.prompts = prompts
        self.cpu_services = CpuServicePool(containers)
        queue.register("llm_prompt", self._guard(self.run_llm_prompt))
        queue.register("motion", self._guard(self.run_motion))
        queue.register("convert", self._guard(self.run_convert))
        queue.register("group_preview", self.run_group_preview)

    # ------------------------------------------------------------ 投入

    def submit_generation(
        self,
        *,
        adapter_id: str,
        inputs: dict[str, Any],
        params: dict[str, Any],
        group_id: str | None = None,
        request_text: str = "",
        llm_model: str | None = None,
        llm_prompt: str | None = None,
    ) -> tuple[str, Job]:
        manifest = self.manifests.get(adapter_id)
        if manifest is None:
            raise KeyError(f"unknown adapter: {adapter_id}")
        missing = [a for a in check_assets(manifest, self.settings.models_dir) if not a.ok]
        if missing:
            raise ContainerError("missing_asset", "; ".join(f"{a.id}: {a.detail}（{a.how_to}）" for a in missing))
        params = {**{k: v for k, v in manifest.defaults.items()}, **params}
        for key in self._file_inputs(manifest):
            if inputs.get(key):
                try:
                    resolve_upload(self.settings, str(inputs[key]))
                except UploadError as e:
                    raise ContainerError("invalid_input", str(e)) from None
            elif key in manifest.inputs.get("required", []):
                raise ContainerError("invalid_input", f"{key} のファイルが必要")
        final_prompt = inputs.get("text")
        rid = self.db.create_result(
            adapter_id=adapter_id,
            adapter_version=manifest.version,
            group_id=group_id,
            request_text=request_text,
            inputs=inputs,
            params=params,
            final_prompt=final_prompt,
            llm_model=llm_model,
            llm_prompt=llm_prompt,
        )
        job = Job(
            id=new_id(),
            type="motion",
            resource=manifest.device,
            model_key=adapter_id,
            result_id=rid,
            group_id=group_id,
            payload={"adapter_id": adapter_id},
        )
        self.queue.submit(job)
        self._publish_result(rid)
        return rid, job

    def submit_drafts(self, *, request_text: str, adapter_ids: list[str], llm_model: str | None) -> tuple[str, list[str]]:
        """依頼文から、テキスト入力のアダプターごとにプロンプト案を作るジョブを投入する。"""
        name = llm_model or self.prompts.config.default
        if name not in self.prompts.config.models:
            raise KeyError(f"unknown LLM: {name}")
        for adapter_id in adapter_ids:
            manifest = self.manifests.get(adapter_id)
            if manifest is None:
                raise KeyError(f"unknown adapter: {adapter_id}")
            if not manifest.prompt.llm_assist:
                raise ContainerError("invalid_input", f"{adapter_id} は LLM によるプロンプト生成の対象外")
        gid = self.db.create_group(request_text)
        draft_ids = []
        for adapter_id in adapter_ids:
            did = self.db.create_draft(group_id=gid, adapter_id=adapter_id, llm_model=name, request_text=request_text)
            draft_ids.append(did)
            self.queue.submit(
                Job(
                    id=new_id(),
                    type="llm_prompt",
                    resource="gpu",
                    model_key=model_key(name),
                    group_id=gid,
                    payload={"draft_id": did},
                )
            )
            self._publish_draft(did)
        return gid, draft_ids

    def confirm_group(self, group_id: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
        """確定したプロンプトで motion ジョブを投入する。items: adapter_id, prompt, duration, seed, draft_id"""
        group = self.db.get_group(group_id)
        if group is None:
            raise KeyError(f"unknown group: {group_id}")
        out = []
        for item in items:
            draft = self.db.get_draft(item["draft_id"]) if item.get("draft_id") else None
            params: dict[str, Any] = {}
            if item.get("duration") is not None:
                params["duration"] = float(item["duration"])
            if item.get("seed") is not None:
                params["seed"] = int(item["seed"])
            rid, job = self.submit_generation(
                adapter_id=item["adapter_id"],
                inputs={"text": item["prompt"]},
                params=params,
                group_id=group_id,
                request_text=group["request_text"],
                llm_model=draft["llm_model"] if draft else None,
                llm_prompt=draft["prompt"] if draft else None,
            )
            if draft:
                self.db.update_draft(draft["id"], result_id=rid)
                self._publish_draft(draft["id"])
            out.append({"result_id": rid, "job_id": job.id})
        return out

    def retry_result(self, rid: str) -> tuple[str, Job]:
        """同じ条件（入力・パラメータ・seed）で生成し直す。新しい結果として同じグループに入る。"""
        r = self.db.get_result(rid)
        if r is None:
            raise KeyError(f"unknown result: {rid}")
        return self.submit_generation(
            adapter_id=r["adapter_id"],
            inputs=r["inputs"],
            params={**r["params"], **({"seed": r["seed"]} if r["seed"] is not None else {})},
            group_id=r["group_id"],
            request_text=r["request_text"],
            llm_model=r["llm_model"],
            llm_prompt=r["llm_prompt"],
        )

    @staticmethod
    def _file_inputs(manifest: Manifest) -> list[str]:
        props = manifest.inputs.get("properties", {})
        return [k for k, v in props.items() if v.get("x-input") == "file"]

    # ------------------------------------------------------------ 共通

    def _publish_draft(self, did: str) -> None:
        draft = self.db.get_draft(did)
        if draft:
            self.bus.publish("draft", draft)

    def _set_draft(self, did: str, **fields: Any) -> None:
        self.db.update_draft(did, **fields)
        self._publish_draft(did)

    def _publish_result(self, rid: str) -> None:
        result = self.db.get_result(rid)
        if result:
            self.bus.publish("result", result)

    def _set_result(self, rid: str, **fields: Any) -> None:
        self.db.update_result(rid, **fields)
        self._publish_result(rid)

    def _guard(self, fn):
        def run(job: Job) -> None:
            try:
                fn(job)
            except Exception as e:
                if job.payload.get("draft_id"):
                    code = getattr(e, "code", None)
                    message = getattr(e, "message", None) or str(e)
                    self._set_draft(job.payload["draft_id"], status="failed", error=f"{code}: {message}" if code else message)
                if job.result_id:
                    code = getattr(e, "code", None)
                    message = getattr(e, "message", None) or str(e)
                    self._set_result(
                        job.result_id,
                        status="failed",
                        error=f"{code}: {message}" if code else message,
                        finished_at=now_iso(),
                    )
                raise

        return run

    def result_dir(self, rid: str) -> Path:
        d = self.settings.results_dir / rid
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ llm_prompt

    def run_llm_prompt(self, job: Job) -> None:
        did = job.payload["draft_id"]
        draft = self.db.get_draft(did)
        manifest = self.manifests[draft["adapter_id"]]
        name = draft["llm_model"]
        before = self.containers.current

        def on_status(status: str) -> None:
            self.queue.update(job, status=status)
            self._set_draft(did, status=status)

        running = self.containers.ensure(llm_spec(self.prompts.config, name, self.settings), on_status=on_status)
        load_seconds = running.load_seconds if running is not before else None
        self.queue.update(job, status="running")
        self._set_draft(did, status="running", load_seconds=load_seconds)
        result = generate_prompt(
            running.base_url, self.prompts.config, name, manifest, draft["request_text"], self.settings.timeouts.llm
        )
        self._set_draft(
            did,
            status="done",
            prompt=result.prompt,
            duration=result.duration,
            notes=result.notes,
            violations=result.violations,
            attempts=result.attempts,
            reasoning=result.reasoning,
            llm_seconds=result.seconds,
            error=None,
        )

    # ------------------------------------------------------------ motion

    def run_motion(self, job: Job) -> None:
        manifest = self.manifests[job.payload["adapter_id"]]
        rid = job.result_id
        result = self.db.get_result(rid)
        pool = self.containers if manifest.device == "gpu" else self.cpu_services
        before = self.containers.current if manifest.device == "gpu" else self.cpu_services.running.get(manifest.id)

        def on_status(status: str) -> None:
            self.queue.update(job, status=status)
            self._set_result(rid, status=status)

        running = pool.ensure(self.containers.adapter_spec(manifest), on_status=on_status)
        load_seconds = running.load_seconds if running is not before else None
        self.queue.update(job, status="running", progress=0.0)
        self._set_result(rid, status="running", load_seconds=load_seconds)

        job_out = self.settings.jobs_dir / job.id / "output"
        job_out.mkdir(parents=True, exist_ok=True)
        # アップロードしたファイルをジョブの入力ディレクトリに置き、コンテナ内のパスで渡す
        inputs = dict(result["inputs"])
        job_in = self.settings.jobs_dir / job.id / "input"
        for key in self._file_inputs(manifest):
            if inputs.get(key):
                src = resolve_upload(self.settings, str(inputs[key]))
                job_in.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, job_in / src.name)
                inputs[key] = f"/data/{(job_in / src.name).relative_to(self.settings.data_dir).as_posix()}"
        payload = {
            "job_id": job.id,
            "inputs": inputs,
            "params": result["params"],
            "output_dir": f"/data/{job_out.relative_to(self.settings.data_dir).as_posix()}",
        }
        timeout = manifest.resources.timeouts.get("generate", self.settings.timeouts.generate)
        body = self.containers.generate(
            running, payload, timeout, lambda p: self.queue.update(job, progress=round(p, 3))
        )

        # 元出力と SourceMotion を結果ディレクトリへ移す
        out = self.result_dir(rid)
        files: dict[str, str] = {}
        for name in [body["source_motion"], *body.get("raw_outputs", [])]:
            src = job_out / name
            if src.exists():
                shutil.move(str(src), out / name)
        files["source_motion"] = body["source_motion"]
        for name in body.get("raw_outputs", []):
            files[f"raw:{name}"] = name
        shutil.rmtree(self.settings.jobs_dir / job.id, ignore_errors=True)
        self.db.add_result_files(rid, files)
        self._set_result(
            rid,
            seed=body.get("seed"),
            generate_seconds=body.get("extra", {}).get("generate_seconds"),
        )
        job.output = body

        convert = Job(
            id=new_id(),
            type="convert",
            resource="cpu",
            result_id=rid,
            group_id=job.group_id,
            depends_on=[job.id],
            payload={"adapter_id": manifest.id},
        )
        self.queue.submit(convert)

    # ------------------------------------------------------------ convert

    def run_convert(self, job: Job) -> None:
        manifest = self.manifests[job.payload["adapter_id"]]
        rid = job.result_id
        self._set_result(rid, status="converting")
        out = self.result_dir(rid)
        result = self.db.get_result(rid)
        started = time.monotonic()
        body = self.converter.convert(out / result["files"]["source_motion"], load_bone_map(manifest), out)
        self.db.add_result_files(rid, body["files"])
        self._set_result(
            rid,
            status="done",
            convert_seconds=time.monotonic() - started,
            duration=body.get("duration"),
            fps=body.get("fps"),
            finished_at=now_iso(),
            error=None,
        )
        if job.group_id:
            self.maybe_group_preview(job.group_id)

    # ------------------------------------------------------------ group_preview

    def group_dir(self, gid: str) -> Path:
        d = self.settings.data_dir / "groups" / gid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def maybe_group_preview(self, gid: str) -> None:
        """グループの結果がすべて終わり、2 件以上が完了していればプレビューを作る。"""
        results = self.db.list_results(group_id=gid)
        if any(r["status"] not in ("done", "failed") for r in results):
            return
        if sum(r["status"] == "done" for r in results) < 2:
            return
        # 変換ジョブが同時に終わったときに二重に作らない
        if any(j.type == "group_preview" and j.group_id == gid for j in self.queue.list(include_finished=False)):
            return
        self.queue.submit(Job(id=new_id(), type="group_preview", resource="cpu", group_id=gid))

    def run_group_preview(self, job: Job) -> None:
        gid = job.group_id
        done = [r for r in self.db.list_results(group_id=gid) if r["status"] == "done"]
        # 作成時刻（秒単位）→ ID の順。フロントエンドの比較ビューの色の順と揃える
        done.sort(key=lambda r: (r["created_at"], r["id"]))
        items = [
            (
                self.settings.results_dir / r["id"] / r["files"]["source_motion"],
                load_bone_map(self.manifests[r["adapter_id"]]),
                GROUP_COLORS[i % len(GROUP_COLORS)],
            )
            for i, r in enumerate(done)
        ]
        out = self.group_dir(gid) / "group_preview.webp"
        self.converter.preview_group(items, out)
        self.db.set_group_preview(gid, out.name)
        group = self.db.get_group(gid)
        if group:
            self.bus.publish("group", {**group, "result_ids": [r["id"] for r in done]})
