"""比較動画の MP4 書き出し（POST /api/exports → frames → finish → GET file）。ffmpeg を実際に動かす。"""

import json
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vmb_backend.api import build_router
from vmb_backend.config import Settings
from vmb_backend.db import Database

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")


def jpeg(width: int, height: int, color: str) -> bytes:
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s={width}x{height}", "-frames:v", "1",
           "-f", "mjpeg", "-"]  # fmt: skip
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def probe(path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,pix_fmt",
           "-of", "json", str(path)]  # fmt: skip
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


@pytest.fixture()
def env(tmp_path):
    settings = replace(Settings(), data_dir=tmp_path)
    db = Database(tmp_path / "db.sqlite3")
    rid = db.create_result(adapter_id="edge", adapter_version="0.1.0", group_id=None, request_text="", inputs={}, params={})
    (settings.results_dir / rid).mkdir(parents=True)
    # 映像（2 秒）より短い 1 秒の音声。無音で埋めて映像の長さに合わせる
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=1",
                    str(settings.results_dir / rid / "input_audio.wav")], check=True)  # fmt: skip
    db.add_result_files(rid, {"audio": "input_audio.wav"})
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(settings=settings, db=db)))
    return TestClient(app), settings, rid


@pytest.mark.parametrize("size,with_audio", [((1920, 1080), True), ((1080, 1920), False)])
def test_export_flow(env, size, with_audio):
    client, settings, rid = env
    w, h = size
    body = {"width": w, "height": h, "frames": 60, **({"audio_result_id": rid} if with_audio else {})}
    sid = client.post("/api/exports", json=body).json()["id"]
    frames = [jpeg(w, h, "red"), jpeg(w, h, "blue")]
    for i in range(60):
        assert client.post(f"/api/exports/{sid}/frames", content=frames[i // 30]).status_code == 200
    assert client.post(f"/api/exports/{sid}/finish").status_code == 200
    res = client.get(f"/api/exports/{sid}/file", params={"download": "vmb_compare_side.mp4"})
    assert res.status_code == 200 and 'filename="vmb_compare_side.mp4"' in res.headers["content-disposition"]
    info = probe(settings.exports_dir / f"{sid}.mp4")
    streams = {s["codec_type"]: s for s in info["streams"]}
    assert streams["video"]["codec_name"] == "h264" and streams["video"]["pix_fmt"] == "yuv420p"
    assert (streams["video"]["width"], streams["video"]["height"]) == (w, h)
    assert ("audio" in streams) == with_audio
    if with_audio:
        assert streams["audio"]["codec_name"] == "aac"
    assert abs(float(info["format"]["duration"]) - 2.0) < 0.1


def test_rejects_bad_requests(env):
    client, _, _ = env
    assert client.post("/api/exports", json={"width": 640, "height": 480, "frames": 10}).status_code == 422
    assert client.post("/api/exports", json={"width": 1920, "height": 1080, "frames": 60 * 30 + 1}).status_code == 422
    assert client.post("/api/exports", json={"width": 1920, "height": 1080, "frames": 10, "audio_result_id": "nope"}).status_code == 404
    sid = client.post("/api/exports", json={"width": 1920, "height": 1080, "frames": 2}).json()["id"]
    assert client.post(f"/api/exports/{sid}/frames", content=b"not a jpeg").status_code == 422
    assert client.post(f"/api/exports/{sid}/frames", content=jpeg(1920, 1080, "red")).status_code == 200
    assert client.post(f"/api/exports/{sid}/finish").status_code == 500  # コマが足りない
    assert client.delete(f"/api/exports/{sid}").status_code == 200
    assert client.get(f"/api/exports/{sid}/file").status_code == 404
    assert client.get("/api/exports/../../etc/file").status_code == 404
