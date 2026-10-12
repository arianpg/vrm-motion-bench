"""HTTP API（ブラウザ向け）。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .containers import ContainerError
from .exports import ExportError, ExportManager
from .manifest import check_assets
from .report import ReportError, build_report
from .uploads import UploadError, save_upload


class GenerateBody(BaseModel):
    adapter_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    request_text: str = ""


class GroupItem(BaseModel):
    adapter_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class GroupBody(BaseModel):
    request_text: str = ""
    items: list[GroupItem]


class DraftsBody(BaseModel):
    request_text: str
    adapter_ids: list[str]
    llm_model: str | None = None


class ConfirmItem(BaseModel):
    adapter_id: str
    prompt: str
    duration: float | None = None
    seed: int | None = None
    draft_id: str | None = None


class ConfirmBody(BaseModel):
    items: list[ConfirmItem]


class ExportBody(BaseModel):
    width: int
    height: int
    frames: int
    audio_result_id: str | None = None  # この結果の入力音声を入れる


def build_router(app_state) -> APIRouter:
    r = APIRouter(prefix="/api")
    s = app_state
    exports = ExportManager(s.settings.exports_dir)

    def submit(item: GroupItem | GenerateBody, group_id: str | None, request_text: str) -> dict[str, str]:
        try:
            rid, job = s.pipeline.submit_generation(
                adapter_id=item.adapter_id,
                inputs=item.inputs,
                params=item.params,
                group_id=group_id,
                request_text=request_text,
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ContainerError as e:
            raise HTTPException(409, {"error_code": e.code, "message": e.message}) from None
        return {"result_id": rid, "job_id": job.id}

    @r.get("/health")
    def health() -> dict[str, Any]:
        current = s.containers.current
        return {
            "status": "ok",
            "loaded_adapter": current.key if current else None,
            "gpu_used_mib": s.containers.gpu.used_mib(),
            "gpu_total_mib": s.containers.gpu.total_mib(),
        }

    @r.get("/adapters")
    def adapters() -> list[dict[str, Any]]:
        out = []
        for m in s.manifests.values():
            data = m.model_dump(mode="json")
            data["asset_status"] = [a.model_dump() for a in check_assets(m, s.settings.models_dir)]
            out.append(data)
        return out

    @r.post("/uploads")
    def upload(kind: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            return save_upload(s.settings, kind, file.filename or "upload", file.file)
        except UploadError as e:
            raise HTTPException(422, str(e)) from None

    @r.post("/generate")
    def generate(body: GenerateBody) -> dict[str, str]:
        return submit(body, None, body.request_text)

    @r.post("/groups")
    def create_group(body: GroupBody) -> dict[str, Any]:
        if not body.items:
            raise HTTPException(422, "items is empty")
        gid = s.db.create_group(body.request_text)
        return {"group_id": gid, "items": [submit(i, gid, body.request_text) for i in body.items]}

    @r.get("/llms")
    def llms() -> list[dict[str, Any]]:
        return s.prompts.list_models()

    @r.post("/drafts")
    def create_drafts(body: DraftsBody) -> dict[str, Any]:
        if not body.request_text.strip() or not body.adapter_ids:
            raise HTTPException(422, "request_text and adapter_ids are required")
        try:
            gid, draft_ids = s.pipeline.submit_drafts(
                request_text=body.request_text.strip(), adapter_ids=body.adapter_ids, llm_model=body.llm_model
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ContainerError as e:
            raise HTTPException(409, {"error_code": e.code, "message": e.message}) from None
        return {"group_id": gid, "drafts": [s.db.get_draft(d) for d in draft_ids]}

    @r.post("/groups/{gid}/confirm")
    def confirm_group(gid: str, body: ConfirmBody) -> dict[str, Any]:
        if not body.items:
            raise HTTPException(422, "items is empty")
        try:
            items = s.pipeline.confirm_group(gid, [i.model_dump() for i in body.items])
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ContainerError as e:
            raise HTTPException(409, {"error_code": e.code, "message": e.message}) from None
        return {"group_id": gid, "items": items}

    @r.get("/groups")
    def list_groups(limit: int = 50) -> list[dict[str, Any]]:
        return s.db.list_groups(limit)

    @r.get("/groups/{gid}")
    def get_group(gid: str) -> dict[str, Any]:
        group = s.db.get_group(gid)
        if group is None:
            raise HTTPException(404, "group not found")
        return {**group, "results": s.db.list_results(group_id=gid), "drafts": s.db.list_drafts(gid)}

    @r.get("/groups/{gid}/preview")
    def group_preview(gid: str) -> FileResponse:
        group = s.db.get_group(gid)
        if group is None or not group.get("preview"):
            raise HTTPException(404, "preview not found")
        return FileResponse(s.settings.data_dir / "groups" / gid / group["preview"])

    # ------------------------------------------------------------ 比較動画（MP4）

    @r.post("/exports")
    def start_export(body: ExportBody) -> dict[str, Any]:
        audio = None
        if body.audio_result_id:
            result = s.db.get_result(body.audio_result_id)
            name = (result or {}).get("files", {}).get("audio")
            if not name:
                raise HTTPException(404, "audio not found")
            audio = s.settings.results_dir / body.audio_result_id / name
        try:
            session = exports.start(width=body.width, height=body.height, frames=body.frames, audio=audio)
        except ExportError as e:
            raise HTTPException(422, str(e)) from None
        return {"id": session.id, "frames": session.frames}

    @r.post("/exports/{sid}/frames")
    async def add_export_frame(sid: str, request: Request) -> dict[str, Any]:
        """本文は JPEG 1 枚。順番どおりに送る。"""
        jpeg = await request.body()
        try:
            received = await asyncio.to_thread(exports.add_frame, sid, jpeg)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ExportError as e:
            raise HTTPException(422, str(e)) from None
        return {"received": received}

    @r.post("/exports/{sid}/finish")
    async def finish_export(sid: str) -> dict[str, Any]:
        try:
            session = await asyncio.to_thread(exports.finish, sid)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ExportError as e:
            raise HTTPException(500, str(e)) from None
        return {"id": session.id, "size": session.path.stat().st_size}

    @r.delete("/exports/{sid}")
    def cancel_export(sid: str) -> dict[str, Any]:
        try:
            exports.cancel(sid)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        return {"id": sid}

    @r.get("/exports/{sid}/file")
    def export_file(sid: str, download: str = "vmb_compare.mp4") -> FileResponse:
        if not re.fullmatch(r"[0-9a-f]{12}", sid):
            raise HTTPException(404, "export not found")
        path = s.settings.exports_dir / f"{sid}.mp4"
        if not path.exists():
            raise HTTPException(404, "export not found")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,150}\.mp4", download):
            raise HTTPException(422, "invalid download name")
        return FileResponse(path, media_type="video/mp4", filename=download)

    @r.get("/report")
    def report(ids: str) -> HTMLResponse:
        """ids: カンマ区切りの結果 ID。比較ビューと同じ順で並べる。"""
        try:
            html, name = build_report(
                s.db,
                s.settings.results_dir,
                s.settings.assets_dir / "stickman/stickman.vrm",
                s.settings.report_template,
                [i for i in ids.split(",") if i],
            )
        except ReportError as e:
            raise HTTPException(409, str(e)) from None
        return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @r.get("/results")
    def list_results(group_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return s.db.list_results(group_id=group_id, limit=limit)

    @r.get("/results/{rid}")
    def get_result(rid: str) -> dict[str, Any]:
        result = s.db.get_result(rid)
        if result is None:
            raise HTTPException(404, "result not found")
        return result

    @r.post("/results/{rid}/retry")
    def retry(rid: str) -> dict[str, str]:
        try:
            new_rid, job = s.pipeline.retry_result(rid)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ContainerError as e:
            raise HTTPException(409, {"error_code": e.code, "message": e.message}) from None
        return {"result_id": new_rid, "job_id": job.id}

    @r.get("/results/{rid}/files/{name}")
    def get_result_file(rid: str, name: str, download: str | None = None) -> FileResponse:
        """download を指定すると、その名前で保存させる（VRMA の保存など。拡張子は元のファイルと同じにする）。"""
        result = s.db.get_result(rid)
        if result is None or name not in result["files"].values():
            raise HTTPException(404, "file not found")
        path = s.settings.results_dir / rid / name
        if not path.exists():
            raise HTTPException(404, "file not found")
        if download is not None:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,150}", download) or Path(download).suffix != path.suffix:
                raise HTTPException(422, "invalid download name")
            return FileResponse(path, filename=download)
        return FileResponse(path, filename=name)

    @r.get("/jobs")
    def jobs(include_finished: bool = False) -> list[dict[str, Any]]:
        return [j.to_dict() for j in s.queue.list(include_finished)]

    @r.get("/assets/stickman.vrm")
    def stickman() -> FileResponse:
        return FileResponse(s.settings.assets_dir / "stickman/stickman.vrm", filename="stickman.vrm")

    @r.get("/events")
    async def events(request: Request) -> StreamingResponse:
        queue = s.bus.subscribe()

        async def stream():
            try:
                snapshot = {"jobs": [j.to_dict() for j in s.queue.list(False)]}
                yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                while not await request.is_disconnected():
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
            finally:
                s.bus.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return r
