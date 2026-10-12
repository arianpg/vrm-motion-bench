"""比較結果を 1 枚の HTML（ネット接続なしで開ける）にする（仕様: docs/spec.md「レポートHTML出力」）。

テンプレートはフロントエンドの `npm run build:report` で作る（frontend/dist-report/report.html）。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from .credits import build_credits
from .db import Database, now_iso
from .manifest import Manifest

PLACEHOLDER = '<script id="vmb-data" type="application/json">__VMB_REPORT_DATA__</script>'
COLORS = ["#5b9cf5", "#ef6b6b", "#4cc38a", "#e5b450", "#b07cf5", "#4cc3c3", "#f58fc8", "#a0a8b5"]


class ReportError(RuntimeError):
    pass


def build_report(
    db: Database,
    results_dir: Path,
    stickman: Path,
    template: Path,
    result_ids: list[str],
    manifests: dict[str, Manifest] | None = None,
) -> tuple[str, str]:
    """(HTML, ファイル名) を返す。"""
    if not template.exists():
        raise ReportError(f"レポートのテンプレートがない: {template}（frontend で npm run build:report を実行する）")
    html = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise ReportError("テンプレートにデータの差し込み位置がない")

    results: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []  # クレジット用（ファイル一覧を含む元の結果）
    for rid in result_ids:
        r = db.get_result(rid)
        if r is None or r["status"] != "done" or "vrma" not in r["files"]:
            raise ReportError(f"完了していない結果は含められない: {rid}")
        vrma = (results_dir / rid / r["files"]["vrma"]).read_bytes()
        contacts_name = r["files"].get("contacts")
        contacts = json.loads((results_dir / rid / contacts_name).read_text()) if contacts_name else None
        sources.append(r)
        i = len(results)
        results.append(
            {
                **r,
                "files": {},  # レポート内では使わない
                "label": f"{r['adapter_id']} · seed {r['seed'] if r['seed'] is not None else '–'}",
                "color": COLORS[i % len(COLORS)],
                "vrma": base64.b64encode(vrma).decode("ascii"),
                "contacts": contacts,
            }
        )
    if not results:
        raise ReportError("結果が選ばれていない")

    groups = {r["group_id"] for r in results}
    request_text = results[0]["request_text"] if len(groups) == 1 else ""
    data = {
        "title": "VRM Motion Bench 比較レポート",
        "request_text": request_text,
        "created_at": now_iso(),
        "stickman": base64.b64encode(stickman.read_bytes()).decode("ascii"),
        "results": results,
        "credits": build_credits(manifests or {}, sources),
    }
    # </script> で JSON が途切れないようにする
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace(PLACEHOLDER, f'<script id="vmb-data" type="application/json">{payload}</script>', 1)
    group_id = next(iter(groups)) if len(groups) == 1 and None not in groups else None
    name = f"vmb_report_{group_id or results[0]['id']}.html"
    return html, name
