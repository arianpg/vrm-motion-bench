import base64
import json

import pytest

from vmb_backend.db import Database
from vmb_backend.report import PLACEHOLDER, ReportError, build_report


def test_build_report_embeds_data(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    gid = db.create_group("手を振る")
    rid = db.create_result(adapter_id="kimodo", adapter_version="0.1", group_id=gid, request_text="手を振る",
                           inputs={"text": "A person waves </script>"}, params={"seed": 1}, final_prompt="A person waves </script>")
    (tmp_path / "results" / rid).mkdir(parents=True)
    (tmp_path / "results" / rid / "motion.vrma").write_bytes(b"vrma-bytes")
    db.add_result_files(rid, {"vrma": "motion.vrma"})
    db.update_result(rid, status="done")
    stickman = tmp_path / "stickman.vrm"
    stickman.write_bytes(b"vrm-bytes")
    template = tmp_path / "report.html"
    template.write_text(f"<html><body>{PLACEHOLDER}<script>if (x === '__VMB_REPORT_DATA__') {{}}</script></body></html>")

    html, name = build_report(db, tmp_path / "results", stickman, template, [rid])
    assert name == f"vmb_report_{gid}.html"
    # 文字列比較用の目印（JS 側）は残り、データ用の script だけが置き換わる
    assert html.count("__VMB_REPORT_DATA__") == 1
    start = html.index('<script id="vmb-data" type="application/json">') + len('<script id="vmb-data" type="application/json">')
    payload = html[start : html.index("</script>", start)]
    data = json.loads(payload.replace("<\\/", "</"))
    assert data["request_text"] == "手を振る"
    assert base64.b64decode(data["results"][0]["vrma"]) == b"vrma-bytes"


def test_build_report_rejects_unfinished(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    rid = db.create_result(adapter_id="kimodo", adapter_version="0.1", group_id=None, request_text="",
                           inputs={}, params={})
    template = tmp_path / "report.html"
    template.write_text(PLACEHOLDER)
    with pytest.raises(ReportError):
        build_report(db, tmp_path, tmp_path / "s.vrm", template, [rid])
