"""HY-Motion 1.0 アダプター。

出力（体 22 関節の 6D 回転、30fps）を、同梱の木製人形モデルの骨格（関節テンプレートと親子関係）で
SourceMotion にする。内蔵のプロンプト書き直し・尺推定（Text2MotionPrompter）は使わない。
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from sourcemotion import SourceMotion, quat
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

ADAPTER_ID = "hymotion"
ADAPTER_VERSION = "0.1.0"
HYMOTION_DIR = Path(os.environ.get("HYMOTION_DIR", "/opt/hymotion"))
VARIANT = os.environ.get("HYMOTION_VARIANT", "HY-Motion-1.0")
CKPT_DIR = Path(os.environ.get("HYMOTION_CKPTS", "/models/hymotion/ckpts")) / "tencent" / VARIANT
BODY_MODEL_DIR = HYMOTION_DIR / "scripts/gradio/static/assets/dump_wooden"
NUM_JOINTS = 22
MIN_FRAMES, MAX_FRAMES = 20, 360


def load_body_skeleton() -> tuple[list[str], np.ndarray, np.ndarray]:
    """同梱の木製人形モデルから、体 22 関節の名前・親・レスト位置を読む。"""
    sys.path.insert(0, str(HYMOTION_DIR))
    from hymotion.pipeline.body_model import load_model_data

    data = load_model_data(str(BODY_MODEL_DIR))
    names = list(data["joint_names"])[:NUM_JOINTS]
    parents = np.array(data["parents"][:NUM_JOINTS], dtype=np.int32)  # frombuffer の配列は読み取り専用なのでコピーする
    rest = np.asarray(data["j_template"][:NUM_JOINTS], dtype=np.float64)
    if parents[0] >= 0 or any(p >= i for i, p in enumerate(parents[1:], 1)):
        raise AdapterError("internal", f"unexpected kintree: {parents.tolist()}")
    parents[0] = -1
    return names, parents, rest


class HyMotionAdapter(Adapter):
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def load(self) -> None:
        config = CKPT_DIR / "config.yml"
        ckpt = CKPT_DIR / "latest.ckpt"
        # 公式の読み込みは、チェックポイントがないと警告だけ出してランダムな重みのまま動くため、先に確かめる
        for path in (config, ckpt):
            if not path.exists():
                raise AdapterError("missing_asset", f"{path} がない")
        if self.device.type == "cpu" and os.environ.get("ALLOW_CPU") != "1":
            raise AdapterError("internal", "CUDA is not available")
        os.chdir(HYMOTION_DIR)  # 統計量（stats/）や木製人形モデルを相対パスで読むため
        sys.path.insert(0, str(HYMOTION_DIR))

        import yaml
        from hymotion.utils.loaders import load_object

        with open(config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        pipeline = load_object(
            cfg["train_pipeline"],
            cfg["train_pipeline_args"],
            network_module=cfg["network_module"],
            network_module_args=cfg["network_module_args"],
        )
        pipeline.load_in_demo(str(ckpt), build_text_encoder=True)
        pipeline.to(self.device)
        pipeline.eval()
        self.pipeline = pipeline
        self.fps = float(pipeline.output_mesh_fps)
        self.names, self.parents, self.rest = load_body_skeleton()

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        from hymotion.utils.geometry import rot6d_to_rotation_matrix

        text = str(request.inputs.get("text", "")).strip()
        if not text:
            raise AdapterError("invalid_input", "inputs.text is required")
        duration = float(request.params.get("duration", 5.0))
        frames = int(round(duration * self.fps))
        if not MIN_FRAMES <= frames <= MAX_FRAMES:
            raise AdapterError("invalid_input", f"duration must be within {MIN_FRAMES / self.fps:.2f}-{MAX_FRAMES / self.fps:.0f} s")
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        cfg_scale = float(request.params.get("cfg_scale", 5.0))

        progress(0.05)
        with torch.no_grad():
            out = self.pipeline.generate(text, [seed], duration, cfg_scale=cfg_scale)
        progress(0.9)

        rot6d = out["rot6d"][0, :, :NUM_JOINTS].float()  # [T, 22, 6]
        rotmats = rot6d_to_rotation_matrix(rot6d).cpu().numpy()  # [T, 22, 3, 3]
        # keypoints3d は移動量を含まない関節位置。腰のワールド位置は「移動量（地面に合わせ済み）＋ レスト時の腰の位置」
        transl = out["transl"][0].cpu().numpy()
        root = transl + self.rest[0]
        raw_name = "hymotion_raw.npz"
        np.savez(
            output_dir / raw_name,
            rot6d=out["rot6d"][0].cpu().numpy(),
            transl=transl,
            keypoints3d=out["keypoints3d"][0].cpu().numpy(),
            fps=np.asarray(self.fps),
            text=np.asarray(text),
        )

        offsets = np.zeros_like(self.rest)
        offsets[1:] = self.rest[1:] - self.rest[self.parents[1:]]
        # offsets[0] はレスト時のルート位置（足裏が地面に接する高さ）
        offsets[0] = [0.0, self.rest[0, 1] - self.rest[:, 1].min(), 0.0]
        meta = {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "model": VARIANT,
            "prompt": text,
            "seed": seed,
            "duration": duration,
            "cfg_scale": cfg_scale,
            "native_prompt_rewrite": False,
            "raw_outputs": [raw_name],
        }
        motion = SourceMotion(
            joint_names=self.names,
            parents=self.parents,
            offsets=offsets.astype(np.float32),
            rest_rotations=np.tile(quat.IDENTITY, (NUM_JOINTS, 1)),
            local_rotations=quat.make_continuous(quat.from_matrix(rotmats)).astype(np.float32),
            root_translation=root.astype(np.float32),
            fps=self.fps,
            up_axis="+Y",
            forward_axis="+Z",
            rest_pose_kind="tpose",
            meta=meta,
        )
        motion.save(output_dir / "source_motion.npz")
        progress(1.0)
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=[raw_name],
            fps=self.fps,
            num_frames=motion.num_frames,
            duration=motion.duration,
            seed=seed,
        )


if __name__ == "__main__":
    serve(HyMotionAdapter(), port=int(os.environ.get("PORT", "8000")))
