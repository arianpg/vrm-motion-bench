"""動画・音声のアップロード。アダプターの入力（x-input: file）で使う。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

from .config import Settings
from .db import new_id


class UploadError(ValueError):
    pass


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "upload"
    return stem[:100]


def probe_duration(path: Path) -> float | None:
    """ffprobe で長さ（秒）を調べる。ffprobe がなければ None。"""
    if shutil.which("ffprobe") is None:
        return None
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        raise UploadError("動画・音声として読めないファイル")
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (KeyError, ValueError):
        return None


def save_upload(settings: Settings, kind: str, filename: str, stream: BinaryIO) -> dict:
    limits = settings.upload_limits.get(kind)
    if limits is None:
        raise UploadError(f"未対応の種類: {kind}")
    ext = Path(filename).suffix.lower()
    if ext not in limits["ext"]:
        raise UploadError(f"{kind} に使えない拡張子: {ext}（{', '.join(limits['ext'])}）")
    uid = new_id()
    d = settings.uploads_dir / uid
    d.mkdir(parents=True)
    path = d / _safe_name(filename)
    max_bytes = limits["max_mb"] * 1024 * 1024
    size = 0
    with open(path, "wb") as f:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise UploadError(f"ファイルが大きすぎる（上限 {limits['max_mb']}MB）")
            f.write(chunk)
    try:
        duration = probe_duration(path)
    except UploadError:
        shutil.rmtree(d, ignore_errors=True)
        raise
    if duration is not None and duration > limits["max_seconds"]:
        shutil.rmtree(d, ignore_errors=True)
        raise UploadError(f"長すぎる（{duration:.1f} 秒、上限 {limits['max_seconds']} 秒）")
    meta = {"upload_id": uid, "kind": kind, "name": path.name, "size": size, "duration": duration}
    (d / "upload.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def resolve_upload(settings: Settings, upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", upload_id or ""):
        raise UploadError(f"不正なアップロード ID: {upload_id}")
    meta_path = settings.uploads_dir / upload_id / "upload.json"
    if not meta_path.exists():
        raise UploadError(f"アップロードが見つからない: {upload_id}")
    return meta_path.parent / json.loads(meta_path.read_text())["name"]
