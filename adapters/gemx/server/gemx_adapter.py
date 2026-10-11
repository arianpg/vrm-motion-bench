"""GEM-X アダプター（動画 → SOMA77 モーション）。

公式のデモ（scripts/demo/demo_soma.py）の前処理（YOLOX + ByteTrack → ViTPose → SAM-3D-Body の特徴）と
推論をそのまま使い、出力（SOMA の各関節の軸角度・体型・移動量）を SourceMotion にする。
- カメラは固定として扱う（公開版は Visual Odometry を含まず、生成動画も固定カメラで作る）
- 骨格は SOMA の体型係数からレスト時の関節位置を求めて使う（全フレームの平均の体型）
- 世界座標は、足裏を地面（y=0）に合わせ、最初のフレームで +Z を向き原点に立つように正規化する
"""

from __future__ import annotations

import functools
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sourcemotion import SourceMotion, quat
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

ADAPTER_ID = "gemx"
ADAPTER_VERSION = "0.1.0"
GEMX_DIR = Path(os.environ.get("GEMX_DIR", "/opt/gemx"))
INPUTS = GEMX_DIR / "inputs"
REQUIRED = [
    INPUTS / "pretrained" / "gem_soma.ckpt",
    INPUTS / "checkpoints" / "vitpose" / "vitpose.pth",
    INPUTS / "checkpoints" / "sam-3d-body-dinov3" / "sam3d_body.ckpt",
    INPUTS / "mhr_data" / "mhr_model.pt",
    INPUTS / "soma_data" / "scale_mean.pth",
]
# 向きの正規化に使う関節（SOMA77 の名前）
LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER = "LeftLeg", "RightLeg", "LeftArm", "RightArm"


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """軸角度 [..., 3] → クォータニオン [..., 4]（xyzw）。"""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, aa / np.maximum(angle, 1e-12), np.array([1.0, 0.0, 0.0]))
    return quat.from_axis_angle(axis, angle[..., 0])


def cached(factory):
    """前処理のモデル（ViTPose、SAM-3D-Body、YOLOX）を毎回読み込まないよう、最初のインスタンスを使い回す。"""
    instances: dict = {}

    @functools.wraps(factory)
    def make(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in instances:
            instances[key] = factory(*args, **kwargs)
        return instances[key]

    return make


class GemXAdapter(Adapter):
    def __init__(self) -> None:
        self.device = "cuda:0"

    def load(self) -> None:
        missing = [str(p) for p in REQUIRED if not p.exists()]
        if missing:
            raise AdapterError("missing_asset", "missing: " + ", ".join(missing))
        if not torch.cuda.is_available():
            raise AdapterError("internal", "CUDA is not available")
        # rtmlib（YOLOX）は ~/.cache に重みを置くので、ホームを永続化した場所に向ける（オーケストレーターは HOME=/tmp で起動する）
        home = os.environ.get("GEMX_HOME")
        if home:
            Path(home).mkdir(parents=True, exist_ok=True)
            os.environ["HOME"] = home
        os.chdir(GEMX_DIR)  # 公式のコードは inputs/ などを相対パスで読む
        for p in (GEMX_DIR, GEMX_DIR / "scripts" / "demo"):
            sys.path.insert(0, str(p))

        import demo_soma as demo
        import gem.utils.yolox_detector as yolox
        import hydra
        from gem.utils.soma_utils.soma_layer import SomaLayer

        demo.VitPoseExtractor = cached(demo.VitPoseExtractor)
        demo.SAM3DBExtractor = cached(demo.SAM3DBExtractor)
        yolox.YOLOXDetector = cached(yolox.YOLOXDetector)
        self.demo = demo

        cfg = self._cfg(GEMX_DIR / "README.md", "load", static_cam=True)  # モデルの設定は動画に依らない
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(str(INPUTS / "pretrained" / "gem_soma.ckpt"))
        self.model = model.eval().cuda()
        self.soma = SomaLayer(
            data_root="inputs/soma_assets", low_lod=True, device=self.device, identity_model_type="mhr", mode="warp"
        )
        self.joint_names = self._joint_names()

    def _joint_names(self) -> list[str]:
        candidates = []
        for owner in (self.soma, getattr(self.soma, "soma", None)):
            if owner is None:
                continue
            candidates.append(getattr(owner, "joint_names", None))
            rig = getattr(owner, "rig_data", None)
            if rig is not None and "joint_names" in rig:
                # 親子関係（parents）は rig_data の先頭（ワールドの根）を除いて作られているので、名前もそろえる
                names = list(rig["joint_names"])
                candidates.append(names[1:] if len(names) == 78 else names)
        for names in candidates:
            if names is not None and len(names) == 77:
                return [str(n) for n in names]
        raise AdapterError("internal", "SOMA の関節名が取得できない")

    def _cfg(self, video: Path, name: str, static_cam: bool, output_root: Path | None = None):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        overrides = [
            "exp=gem_soma_regression",
            f"video_name={name}",
            f"video_path={video}",
            f"output_root={output_root or '/tmp/gemx'}",
            f"static_cam={str(static_cam).lower()}",
            "verbose=false",
            "render_mhr=false",
            "use_wandb=false",
            "task=test",
        ]
        with initialize_config_dir(version_base="1.3", config_dir=str(GEMX_DIR / "configs")):
            cfg = compose(config_name="demo_soma", overrides=overrides)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        import cv2
        from gem.utils.net_utils import detach_to_cpu, to_cuda

        video = request.inputs.get("video")
        if not video or not Path(video).exists():
            raise AdapterError("invalid_input", f"video not found: {video}")
        static_cam = bool(request.params.get("static_cam", True))
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)

        work = output_dir / "gemx_work"
        cfg = self._cfg(Path(video), "input", static_cam, output_root=work)
        # 公式のデモは動画を 30fps の器に詰め直すが、元の fps を保つためそのまま使う
        fps = float(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)) or 30.0

        started = time.monotonic()
        progress(0.05)
        self.demo.run_preprocess(cfg)
        progress(0.6)
        data = self.demo.load_data_dict(cfg)
        with torch.no_grad():
            pred = self.model.predict(data, static_cam=static_cam, postproc=True)
        pred = detach_to_cpu(pred)
        params = {k: v.float() for k, v in self.demo._get_body_params(pred, "body_params_global").items()}
        progress(0.85)

        motion = self.to_source_motion(params, fps, to_cuda)
        motion.meta.update(
            {
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "video": Path(video).name,
                "static_cam": static_cam,
                "seed": seed,
                "raw_outputs": ["gemx_body_params.pt", "gemx_kp2d.pt"],
            }
        )
        torch.save(params, output_dir / "gemx_body_params.pt")
        shutil.copy2(cfg.paths.vitpose, output_dir / "gemx_kp2d.pt")
        shutil.rmtree(work, ignore_errors=True)
        motion.save(output_dir / "source_motion.npz")
        progress(1.0)
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=["gemx_body_params.pt", "gemx_kp2d.pt"],
            fps=fps,
            num_frames=motion.num_frames,
            duration=motion.duration,
            seed=seed,
            extra={"gemx_seconds": time.monotonic() - started},
        )

    def to_source_motion(self, params: dict, fps: float, to_cuda) -> SourceMotion:
        names = self.joint_names
        parents = np.asarray(self.soma.parents.cpu().numpy() if hasattr(self.soma.parents, "cpu") else self.soma.parents)
        parents = parents.astype(np.int32).copy()
        parents[0] = -1

        with torch.no_grad():
            out = self.soma(**to_cuda(params))
            identity = params["identity_coeffs"].mean(0, keepdim=True)
            scale = params["scale_params"].mean(0, keepdim=True) if params.get("scale_params") is not None else None
            rest = self.soma.get_skeleton(to_cuda(identity), to_cuda(scale) if scale is not None else None)[0]
        joints = out["joints"].cpu().numpy().astype(np.float64)  # [L, 77, 3]
        verts_min_y = float(out["vertices"][..., 1].min())
        rest = rest.cpu().numpy().astype(np.float64)

        poses = np.concatenate(
            [params["global_orient"].numpy()[:, None], params["body_pose"].numpy().reshape(-1, 76, 3)], axis=1
        )
        local = axis_angle_to_quat(poses.astype(np.float64))  # [L, 77, 4]

        offsets = np.zeros_like(rest)
        offsets[1:] = rest[1:] - rest[parents[1:]]
        offsets[0] = [0.0, rest[0, 1] - rest[:, 1].min(), 0.0]

        # 検証: 平均の体型の骨格と回転で計算した関節位置が、SOMA の層の関節位置と一致するか（正規化の前に比べる）
        check = SourceMotion(
            joint_names=names, parents=parents, offsets=offsets.astype(np.float32),
            rest_rotations=np.tile(quat.IDENTITY, (len(names), 1)), local_rotations=quat.normalize(local).astype(np.float32),
            root_translation=joints[:, 0].astype(np.float32), fps=fps, up_axis="+Y", forward_axis="+Z", rest_pose_kind="tpose",
        )  # fmt: skip
        fk_error = float(np.linalg.norm(check.global_positions() - joints, axis=-1).max())

        # 正規化: 足裏を y=0 に、最初のフレームで +Z を向き、水平位置を原点に
        root = joints[:, 0].copy()
        root[:, 1] -= verts_min_y
        idx = {n: i for i, n in enumerate(names)}
        side = (joints[0, idx[LEFT_HIP]] - joints[0, idx[RIGHT_HIP]]) + (
            joints[0, idx[LEFT_SHOULDER]] - joints[0, idx[RIGHT_SHOULDER]]
        )
        # キャラクターの左（腰・肩の右→左）を +X に向ける Y 軸まわりの回転角。
        # Y 軸まわりに θ 回すと z' = -x sinθ + z cosθ なので、z' = 0・x' > 0 になる θ = atan2(z, x)
        heading = np.arctan2(side[2], side[0])
        yaw = quat.from_axis_angle(np.array([0.0, 1.0, 0.0]), np.array(heading))
        root = quat.rotate(yaw, root)
        root[:, [0, 2]] -= root[0, [0, 2]]
        local[:, 0] = quat.multiply(yaw[None], local[:, 0])

        return SourceMotion(
            joint_names=names,
            parents=parents,
            offsets=offsets.astype(np.float32),
            rest_rotations=np.tile(quat.IDENTITY, (len(names), 1)),
            local_rotations=quat.make_continuous(quat.normalize(local)).astype(np.float32),
            root_translation=root.astype(np.float32),
            fps=fps,
            up_axis="+Y",
            forward_axis="+Z",
            rest_pose_kind="tpose",
            meta={"fk_max_error_m": fk_error},  # SOMA の層の関節位置との最大誤差（体型の平均化と warp の影響）
        )


if __name__ == "__main__":
    serve(GemXAdapter(), port=int(os.environ.get("PORT", "8000")))
