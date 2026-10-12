"""音声入力のアダプター（EMAGE・EDGE）: アップロードした音声を渡し、アダプターが返した音声を結果のファイルにする。"""

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
from vmb_backend.uploads import save_upload

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    settings = replace(Settings(), data_dir=tmp_path, models_dir=tmp_path / "models")
    for d in (settings.jobs_dir, settings.results_dir, settings.uploads_dir, settings.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("vmb_backend.pipeline.check_assets", lambda m, d: [])
    monkeypatch.setattr("vmb_backend.uploads.probe_duration", lambda path: 5.0)  # ダミーの音声なので長さは固定
    queue = JobQueue(EventBus())
    return Pipeline(settings, Database(tmp_path / "db.sqlite3"), EventBus(), queue, load_manifests(REPO / "adapters"), MagicMock(), MagicMock(), MagicMock())


def test_emage_manifest():
    m = load_manifests(REPO / "adapters")["emage"]
    assert m.prompt.llm_assist is False  # 依頼からの一括生成の対象にしない
    assert m.output.expressions is True
    assert m.inputs["properties"]["audio"]["x-accept"] == "audio"


def test_edge_manifest():
    m = load_manifests(REPO / "adapters")["edge"]
    assert m.prompt.llm_assist is False
    assert m.output.foot_contacts is True
    assert m.inputs["properties"]["audio"]["x-accept"] == "audio"
    assert m.inputs["properties"]["seed"]["x-input"] == "seed"


@pytest.mark.parametrize("adapter_id", ["emage", "edge"])
def test_requires_audio(pipeline, adapter_id):
    with pytest.raises(ContainerError):
        pipeline.submit_generation(adapter_id=adapter_id, inputs={}, params={})


def test_audio_output_is_registered(pipeline, tmp_path):
    src = tmp_path / "talk.mp3"
    src.write_bytes(b"\x00" * 10)
    meta = save_upload(pipeline.settings, "audio", "talk.mp3", src.open("rb"))
    rid, job = pipeline.submit_generation(adapter_id="emage", inputs={"audio": meta["upload_id"]}, params={})

    def generate(running, payload, timeout, progress):
        assert payload["inputs"]["audio"].startswith("/data/jobs/") and payload["inputs"]["audio"].endswith("talk.mp3")
        out = pipeline.settings.data_dir / payload["output_dir"].removeprefix("/data/")
        for name in ("source_motion.npz", "emage_output.npz", "input_audio.mp3"):
            (out / name).write_bytes(b"x")
        return {
            "source_motion": "source_motion.npz",
            "raw_outputs": ["emage_output.npz", "input_audio.mp3"],
            "outputs": {"audio": "input_audio.mp3"},
            "seed": None,
        }

    pipeline.containers.generate.side_effect = generate
    pipeline.containers.ensure.return_value = MagicMock(load_seconds=1.0)
    pipeline.run_motion(job)
    files = pipeline.db.get_result(rid)["files"]
    assert files["audio"] == "input_audio.mp3" == files["raw:input_audio.mp3"]
    assert (pipeline.result_dir(rid) / "input_audio.mp3").exists()
