"""Kimodo アダプター。Kimodo の出力（SOMA77）を SourceMotion NPZ に変換して返す。"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from sourcemotion import SourceMotion, quat
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

ADAPTER_ID = "kimodo"
ADAPTER_VERSION = "0.1.0"
MODEL_NAME = os.environ.get("KIMODO_MODEL", "kimodo-soma-rp")
MAX_DURATION = 10.0


class KimodoAdapter(Adapter):
    def __init__(self) -> None:
        self.model = None
        self.skeleton = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        from kimodo import load_model

        if self.device == "cpu" and os.environ.get("ALLOW_CPU") != "1":
            raise AdapterError("internal", "CUDA is not available")
        self.model = load_model(MODEL_NAME, device=self.device, default_family="Kimodo")
        skel = self.model.skeleton
        # SOMA30 で推論し、出力は SOMA77 に変換される
        self.skeleton = skel.somaskel77 if hasattr(skel, "somaskel77") else skel
        self.contact_skeleton = skel

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.tools import seed_everything

        text = str(request.inputs.get("text", "")).strip()
        if not text:
            raise AdapterError("invalid_input", "inputs.text is required")
        duration = float(request.params.get("duration", 5.0))
        if not 0.5 <= duration <= MAX_DURATION:
            raise AdapterError("invalid_input", f"duration must be within 0.5-{MAX_DURATION} s")
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        steps = int(request.params.get("diffusion_steps", 100))

        fps = float(self.model.fps)
        num_frames = int(round(duration * fps))
        seed_everything(seed)
        progress(0.05)
        with torch.inference_mode():
            output = self.model(
                [text],
                [num_frames],
                constraint_lst=[],
                num_denoising_steps=steps,
                num_samples=1,
                multi_prompt=True,
                num_transition_frames=5,
                post_processing=True,
                return_numpy=True,
            )
        progress(0.9)

        single = {k: (v[0] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1 else v) for k, v in output.items()}
        raw_name = "kimodo_raw.npz"
        save_kimodo_npz(str(output_dir / raw_name), single)

        meta = {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "model": MODEL_NAME,
            "prompt": text,
            "seed": seed,
            "duration": duration,
            "diffusion_steps": steps,
            "raw_outputs": [raw_name],
        }
        motion = to_source_motion(single, self.skeleton, self.contact_skeleton, fps, meta)
        motion.save(output_dir / "source_motion.npz")
        progress(1.0)
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=[raw_name],
            fps=fps,
            num_frames=motion.num_frames,
            duration=motion.duration,
            seed=seed,
        )


def to_source_motion(output: dict, skeleton, contact_skeleton, fps: float, meta: dict) -> SourceMotion:
    """Kimodo の出力（1 サンプル分）を SourceMotion に変換する。

    Kimodo の SOMA は Y-up・+Z 前方・メートル。回転ゼロの姿勢が内部の標準 T ポーズで、
    そのときの関節位置が skeleton.neutral_joints（ルートは原点）。
    """
    names = list(skeleton.bone_order_names)
    parents = np.asarray(skeleton.joint_parents.cpu().numpy(), dtype=np.int32)
    neutral = skeleton.neutral_joints.cpu().numpy().astype(np.float64)  # [J, 3]
    if skeleton.root_idx != 0:
        raise AdapterError("internal", f"unexpected root index {skeleton.root_idx}")

    offsets = np.zeros_like(neutral)
    offsets[1:] = neutral[1:] - neutral[parents[1:]]
    # offsets[0] はレスト時のルート位置。足裏が y=0 になる高さに置く
    offsets[0] = [0.0, -neutral[:, 1].min(), 0.0]

    if "local_rot_mats" in output:
        local_mats = np.asarray(output["local_rot_mats"])
    else:
        global_mats = np.asarray(output["global_rot_mats"])
        local_mats = np.einsum("tjba,tjbc->tjac", global_mats[:, parents.clip(0)], global_mats)
        local_mats[:, 0] = global_mats[:, 0]
    local = quat.make_continuous(quat.from_matrix(local_mats))

    root = np.asarray(output["root_positions"] if "root_positions" in output else output["posed_joints"][:, 0])

    contacts = output.get("foot_contacts")
    contact_names = None
    if contacts is not None:
        contacts = np.asarray(contacts).astype(bool)
        # 出力（SOMA77）の足の関節（左右 3 つずつ）に対応する。推論用の骨格の定義も候補にする
        for skel in (skeleton, contact_skeleton):
            candidates = list(getattr(skel, "foot_joint_names", []) or [])
            if len(candidates) == contacts.shape[1] and all(n in names for n in candidates):
                contact_names = candidates
                break
        else:
            contacts = None  # 対応関節が特定できない場合は付けない

    motion = SourceMotion(
        joint_names=names,
        parents=parents,
        offsets=offsets.astype(np.float32),
        rest_rotations=np.tile(quat.IDENTITY, (len(names), 1)),
        local_rotations=local.astype(np.float32),
        root_translation=root.astype(np.float32),
        fps=fps,
        up_axis="+Y",
        forward_axis="+Z",
        rest_pose_kind="tpose",
        meta=meta,
        foot_contacts=contacts,
        contact_joint_names=contact_names,
    )
    motion.validate()
    return motion


if __name__ == "__main__":
    serve(KimodoAdapter(), port=int(os.environ.get("PORT", "8000")))
