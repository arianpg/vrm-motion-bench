"""動画入力のアダプター（GEM-X）に、テキストから動画生成を前段に挟むジョブの組み立て。"""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vmb_backend.config import Settings
from vmb_backend.containers import ContainerError
from vmb_backend.db import Database
from vmb_backend.jobs import EventBus, JobQueue
from vmb_backend.manifest import load_manifests
from vmb_backend.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    settings = replace(Settings(), data_dir=tmp_path, models_dir=tmp_path / "models")
    for d in (settings.jobs_dir, settings.results_dir, settings.uploads_dir, settings.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    manifests = load_manifests(REPO / "adapters")
    # テストではアセットの配置を確認しない
    monkeypatch.setattr("vmb_backend.pipeline.check_assets", lambda m, d: [])
    queue = JobQueue(EventBus())
    return Pipeline(settings, Database(tmp_path / "db.sqlite3"), EventBus(), queue, manifests, MagicMock(), MagicMock(), MagicMock())


def test_text_input_inserts_video_generation(pipeline):
    rid, motion = pipeline.submit_generation(adapter_id="gemx", inputs={"text": "A man waves."}, params={"seed": 3})
    jobs = {j.id: j for j in pipeline.queue.list()}
    video = jobs[motion.depends_on[0]]
    assert video.type == "video" and video.model_key == "hyvideo" and video.resource == "gpu"
    assert video.result_id == rid == motion.result_id
    assert motion.model_key == "gemx"
    result = pipeline.db.get_result(rid)
    assert result["params"]["duration"] == 5.0 and result["params"]["seed"] == 3


def test_uploaded_video_skips_generation(pipeline, tmp_path, monkeypatch):
    from vmb_backend.uploads import save_upload

    monkeypatch.setattr("vmb_backend.uploads.probe_duration", lambda path: 2.0)  # ダミーの動画なので長さは固定
    mp4 = tmp_path / "in.mp4"
    mp4.write_bytes(b"\x00" * 10)
    meta = save_upload(pipeline.settings, "video", "in.mp4", mp4.open("rb"))
    _, motion = pipeline.submit_generation(adapter_id="gemx", inputs={"video": meta["upload_id"]}, params={})
    assert motion.depends_on == []
    assert [j.type for j in pipeline.queue.list()] == ["motion"]


def test_requires_text_or_video(pipeline):
    with pytest.raises(ContainerError):
        pipeline.submit_generation(adapter_id="gemx", inputs={}, params={})


def test_video_adapter_is_not_a_motion_adapter(pipeline):
    with pytest.raises(KeyError):
        pipeline.submit_generation(adapter_id="hyvideo", inputs={"text": "x"}, params={})
