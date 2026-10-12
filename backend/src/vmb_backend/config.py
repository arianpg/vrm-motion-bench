"""オーケストレーターの設定。環境変数で上書きできる。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default))).resolve()


@dataclass(frozen=True)
class Timeouts:
    """仕様書「タイムアウトの既定値」（秒）。マニフェストの resources.timeouts で上書きできる。"""

    startup_default: float = 600.0
    startup_factor: float = 3.0  # ロード時間の目安 × この係数
    health_interval: float = 2.0
    generate: float = 600.0
    convert: float = 120.0
    llm: float = 300.0
    stop_grace: int = 30
    gpu_release: float = 60.0


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO_ROOT
    adapters_dir: Path = field(default_factory=lambda: _path("VMB_ADAPTERS_DIR", REPO_ROOT / "adapters"))
    data_dir: Path = field(default_factory=lambda: _path("VMB_DATA_DIR", REPO_ROOT / "data"))
    models_dir: Path = field(default_factory=lambda: _path("VMB_MODELS_DIR", REPO_ROOT / "models"))
    assets_dir: Path = field(default_factory=lambda: _path("VMB_ASSETS_DIR", REPO_ROOT / "assets"))
    llm_config: Path = field(default_factory=lambda: _path("VMB_LLM_CONFIG", REPO_ROOT / "config/llm.yaml"))
    report_template: Path = field(
        default_factory=lambda: _path("VMB_REPORT_TEMPLATE", REPO_ROOT / "frontend/dist-report/report.html")
    )
    host: str = os.environ.get("VMB_HOST", "0.0.0.0")
    port: int = int(os.environ.get("VMB_PORT", "8800"))
    gpu_index: int = int(os.environ.get("VMB_GPU_INDEX", "0"))
    # GPU メモリが「解放された」とみなす使用量の上限（MiB）。起動前の使用量 + この余裕
    gpu_release_margin_mib: int = int(os.environ.get("VMB_GPU_RELEASE_MARGIN_MIB", "1024"))
    timeouts: Timeouts = field(default_factory=Timeouts)
    # アップロードの上限（暫定。仕様書「運用・設計上の未決事項」）
    upload_limits: dict = field(
        default_factory=lambda: {
            "video": {"max_mb": 200, "max_seconds": 60, "ext": [".mp4", ".mov", ".webm", ".mkv", ".avi"]},
            "audio": {"max_mb": 50, "max_seconds": 60, "ext": [".wav", ".mp3", ".flac", ".ogg", ".m4a"]},
        }
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vmb.sqlite3"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"


def load_settings() -> Settings:
    settings = Settings()
    for d in (settings.data_dir, settings.jobs_dir, settings.results_dir, settings.uploads_dir, settings.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return settings
