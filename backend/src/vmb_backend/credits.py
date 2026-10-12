"""比較動画（MP4）とレポート HTML に出すクレジット（docs/licenses.md「出力を公開するときの条件」）。"""

from __future__ import annotations

from typing import Any

from .manifest import Manifest

AI_NOTICE = "AI 生成のモーション"


def build_credits(manifests: dict[str, Manifest], results: list[dict[str, Any]]) -> dict[str, Any]:
    """比較中の結果に使ったモデルのクレジット。

    - 動画生成を前段に使った結果（GEM-X のテキスト入力）は、動画生成のアダプターも含める
    - line: MP4 の下部に出す 1 行（「AI 生成」の表示を常に先頭に置く）
    """
    ids: list[str] = []
    for r in results:
        ids.append(r["adapter_id"])
        source = manifests.get(r["adapter_id"])
        if source and source.video_source and "video_meta" in r.get("files", {}):
            ids.append(source.video_source.adapter)
    items = []
    for aid in dict.fromkeys(ids):  # 順序を保って重複を除く
        m = manifests.get(aid)
        if m is None or m.credit is None:
            continue
        items.append(
            {
                "adapter_id": aid,
                "name": m.name,
                "text": m.credit.text,
                "noncommercial": m.credit.noncommercial,
                "notes": list(m.credit.notes),
                "license_note": m.license_note,
            }
        )
    return {
        "notice": AI_NOTICE,
        "line": " · ".join([AI_NOTICE, *(i["text"] for i in items)]),
        "noncommercial": any(i["noncommercial"] for i in items),
        "items": items,
    }
