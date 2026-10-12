"""seed 違いの比較: 同じアダプター・同じ入力で seed だけ変えた項目を、1 つの比較グループとして投入する（POST /api/groups）。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vmb_backend.api import build_router
from vmb_backend.config import Settings
from vmb_backend.db import Database
from vmb_backend.jobs import EventBus, JobQueue
from vmb_backend.manifest import load_manifests
from vmb_backend.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[2]


def test_seed_variants_group(tmp_path, monkeypatch):
    settings = replace(Settings(), data_dir=tmp_path, models_dir=tmp_path / "models")
    for d in (settings.jobs_dir, settings.results_dir, settings.uploads_dir, settings.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("vmb_backend.pipeline.check_assets", lambda m, d: [])
    db = Database(tmp_path / "db.sqlite3")
    queue = JobQueue(EventBus())
    pipeline = Pipeline(settings, db, EventBus(), queue, load_manifests(REPO / "adapters"), MagicMock(), MagicMock(), MagicMock())
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(settings=settings, db=db, pipeline=pipeline)))

    item = {"adapter_id": "momask", "inputs": {"text": "A person waves."}, "params": {"duration": 4.0}}
    body = {"request_text": "seed 比較（MoMask）: A person waves.", "items": [{**item, "params": {**item["params"], "seed": s}} for s in (10, 11, 12)]}
    res = TestClient(app).post("/api/groups", json=body)
    assert res.status_code == 200
    gid = res.json()["group_id"]
    results = db.list_results(group_id=gid)
    assert sorted(r["params"]["seed"] for r in results) == [10, 11, 12]
    assert {r["adapter_id"] for r in results} == {"momask"}
    assert all(r["inputs"] == item["inputs"] and r["params"]["duration"] == 4.0 for r in results)
    assert db.get_group(gid)["request_text"] == body["request_text"]
    assert [j.type for j in queue.list()] == ["motion"] * 3


def test_group_order_sorts_same_second_by_seed():
    from vmb_backend.pipeline import group_order

    same = "2026-10-12T04:00:00+00:00"
    results = [
        {"id": "c", "created_at": same, "seed": 101},
        {"id": "a", "created_at": same, "seed": None},
        {"id": "b", "created_at": same, "seed": 100},
        {"id": "z", "created_at": "2026-10-12T03:59:59+00:00", "seed": 999},
    ]
    assert [r["id"] for r in sorted(results, key=group_order)] == ["z", "b", "c", "a"]
