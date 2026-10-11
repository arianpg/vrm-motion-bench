"""アダプターのマニフェスト（adapters/<id>/manifest.yaml）の読み込みと、アセットの事前確認。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DurationSpec(BaseModel):
    min: float
    max: float
    step: float = 0.1
    auto_supported: bool = False


class PromptRules(BaseModel):
    model_config = ConfigDict(extra="allow")
    language: str | None = None
    starts_with: str | None = None
    max_words: int | None = None
    forbidden_terms: list[str] = Field(default_factory=list)


class PromptSpec(BaseModel):
    llm_assist: bool = False
    guide: str = ""
    rules: PromptRules = Field(default_factory=PromptRules)


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    skeleton: str
    fps: float
    up: str
    forward: str
    unit: str = "m"
    rest_pose: str = "tpose"
    foot_contacts: bool = False
    expressions: bool = False


class AssetSpec(BaseModel):
    id: str
    kind: Literal["hf_gated", "manual_download", "auto"]
    path: str | None = None  # models_dir からの相対パス
    repo: str | None = None
    how_to: str = ""
    license: str | None = None


class VideoSourceSpec(BaseModel):
    """動画入力のアダプターで、テキストから動画を生成して入力にするための設定。"""

    adapter: str  # kind: video のアダプター ID
    input: str = "video"  # 生成した動画を渡す入力欄
    start_image: str  # assets ディレクトリからの相対パス（I2V の開始画像）
    params: dict[str, Any] = Field(default_factory=dict)  # 動画生成に渡す追加のパラメータ


class ResourceSpec(BaseModel):
    vram_gb: float = 0
    load_time_s: float | None = None
    container_image: str
    timeouts: dict[str, float] = Field(default_factory=dict)


class Manifest(BaseModel):
    id: str
    name: str
    version: str
    device: Literal["gpu", "cpu"]
    # motion: モーションを返すアダプター。video: 動画を返すアダプター（ほかのアダプターの前段で使う）
    kind: Literal["motion", "video"] = "motion"
    inputs: dict[str, Any]
    duration: DurationSpec | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    options: dict[str, Any] = Field(default_factory=dict)
    output: OutputSpec | None = None  # kind: video では不要
    assets: list[AssetSpec] = Field(default_factory=list)
    resources: ResourceSpec
    video_source: VideoSourceSpec | None = None
    license_note: str = ""

    # 読み込み元（API には出さない）
    directory: Path = Field(exclude=True)

    @property
    def bone_map_path(self) -> Path:
        return self.directory / "bone_map.yaml"


class AssetStatus(BaseModel):
    id: str
    kind: str
    ok: bool
    how_to: str = ""
    detail: str = ""


def load_manifests(adapters_dir: Path) -> dict[str, Manifest]:
    manifests: dict[str, Manifest] = {}
    for path in sorted(adapters_dir.glob("*/manifest.yaml")):
        data = yaml.safe_load(path.read_text())
        manifest = Manifest(**data, directory=path.parent)
        if manifest.id != path.parent.name:
            raise ValueError(f"{path}: id '{manifest.id}' must match directory name")
        manifests[manifest.id] = manifest
    return manifests


def load_bone_map(manifest: Manifest) -> dict[str, str]:
    return dict(yaml.safe_load(manifest.bone_map_path.read_text()))


def check_assets(manifest: Manifest, models_dir: Path) -> list[AssetStatus]:
    """コンテナを起動せずに確認できる範囲でアセットの配置を確かめる。"""
    statuses: list[AssetStatus] = []
    for asset in manifest.assets:
        ok, detail = True, ""
        if asset.kind == "hf_gated":
            if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
                ok, detail = False, "HF_TOKEN が設定されていない"
            elif asset.path and asset.repo:
                repo_dir = models_dir / asset.path / "hub" / ("models--" + asset.repo.replace("/", "--"))
                if not repo_dir.exists():
                    detail = "未ダウンロード（初回起動時にダウンロードされる）"
        elif asset.kind == "manual_download":
            if not asset.path or not (models_dir / asset.path).exists():
                ok, detail = False, f"{models_dir / (asset.path or '')} が見つからない"
        statuses.append(AssetStatus(id=asset.id, kind=asset.kind, ok=ok, how_to=asset.how_to, detail=detail))
    return statuses
