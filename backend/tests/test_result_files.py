"""結果のファイルの取得（GET /api/results/{rid}/files/{name}）。download で保存名を指定できる。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vmb_backend.api import build_router
from vmb_backend.config import Settings
from vmb_backend.db import Database


@pytest.fixture()
def client_and_rid(tmp_path):
    settings = replace(Settings(), data_dir=tmp_path)
    db = Database(tmp_path / "db.sqlite3")
    rid = db.create_result(adapter_id="edge", adapter_version="0.1.0", group_id=None, request_text="", inputs={}, params={})
    (settings.results_dir / rid).mkdir(parents=True)
    (settings.results_dir / rid / "motion.vrma").write_bytes(b"glTF")
    db.add_result_files(rid, {"vrma": "motion.vrma"})
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(settings=settings, db=db)))
    return TestClient(app), rid


def test_default_name(client_and_rid):
    client, rid = client_and_rid
    res = client.get(f"/api/results/{rid}/files/motion.vrma")
    assert res.status_code == 200 and res.content == b"glTF"
    assert 'filename="motion.vrma"' in res.headers["content-disposition"]


def test_download_name(client_and_rid):
    client, rid = client_and_rid
    res = client.get(f"/api/results/{rid}/files/motion.vrma", params={"download": f"edge_seed7_{rid}.vrma"})
    assert res.status_code == 200
    assert f'filename="edge_seed7_{rid}.vrma"' in res.headers["content-disposition"]


@pytest.mark.parametrize("name", ["../x.vrma", "a/b.vrma", "motion.html", ".hidden.vrma", "x.vrma\r\nX: y"])
def test_rejects_bad_download_name(client_and_rid, name):
    client, rid = client_and_rid
    assert client.get(f"/api/results/{rid}/files/motion.vrma", params={"download": name}).status_code == 422
