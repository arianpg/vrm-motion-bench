"""比較動画・レポートのクレジット。"""

from pathlib import Path

from vmb_backend.credits import AI_NOTICE, build_credits
from vmb_backend.manifest import load_manifests

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = load_manifests(REPO / "adapters")


def result(adapter_id: str, files: dict | None = None) -> dict:
    return {"adapter_id": adapter_id, "files": files or {}}


def test_every_motion_adapter_has_credit():
    for m in MANIFESTS.values():
        assert m.credit is not None and m.credit.text, m.id


def test_line_starts_with_ai_notice_and_dedupes():
    c = build_credits(MANIFESTS, [result("kimodo"), result("edge"), result("kimodo")])
    assert c["line"] == f"{AI_NOTICE} · Kimodo (NVIDIA) · EDGE (Stanford) + OpenAI Jukebox"
    assert c["noncommercial"] is True
    assert [i["adapter_id"] for i in c["items"]] == ["kimodo", "edge"]


def test_generated_video_adds_video_model():
    uploaded = build_credits(MANIFESTS, [result("gemx", {"video": "input_video.mp4"})])
    generated = build_credits(MANIFESTS, [result("gemx", {"video": "input_video.mp4", "video_meta": "video_meta.json"})])
    assert [i["adapter_id"] for i in uploaded["items"]] == ["gemx"]
    assert [i["adapter_id"] for i in generated["items"]] == ["gemx", "hyvideo"]
    assert any("機械生成" in n for n in generated["items"][1]["notes"])
    assert generated["noncommercial"] is False
