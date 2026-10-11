"""ARDY アダプター。ARDY の出力（Core 骨格、20fps）を SourceMotion NPZ に変換して返す。

ARDY はストリーミング（自己回帰）で動くモデルだが、ここでは指定した尺を 1 回の呼び出しで生成し、
1 本のクリップとして返す（仕様書「リアルタイム性」）。
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve
from vmb_adapter.nv_motion import to_source_motion

ADAPTER_ID = "ardy"
ADAPTER_VERSION = "0.1.0"
MODEL_NAME = os.environ.get("ARDY_MODEL", "core40")
MAX_DURATION = 10.0


def default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    """学習時の窓（10 秒）に収まる最長の履歴（scripts/generate.py と同じ計算）。"""
    max_window_len = (int(10 * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token


class ArdyAdapter(Adapter):
    def __init__(self) -> None:
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.resolved = MODEL_NAME

    def load(self) -> None:
        from ardy.model import load_model
        from ardy.model.registry import resolve_model_name

        if self.device == "cpu" and os.environ.get("ALLOW_CPU") != "1":
            raise AdapterError("internal", "CUDA is not available")
        self.resolved = resolve_model_name(MODEL_NAME)
        self.model = load_model(self.resolved, device=self.device)

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        from ardy.motion_rep.tools import length_to_mask
        from ardy.postprocess import post_process_motion
        from ardy.tools import seed_everything, to_numpy

        text = str(request.inputs.get("text", "")).strip()
        if not text:
            raise AdapterError("invalid_input", "inputs.text is required")
        duration = float(request.params.get("duration", 5.0))
        if not 0.5 <= duration <= MAX_DURATION:
            raise AdapterError("invalid_input", f"duration must be within 0.5-{MAX_DURATION} s")
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)

        model = self.model
        fps = float(model.motion_rep.fps)
        num_frames = int(round(duration * fps))
        num_base_steps = int(model.diffusion.num_base_steps)
        steps = int(request.params.get("diffusion_steps", num_base_steps))
        if not 1 <= steps <= num_base_steps:
            raise AdapterError("invalid_input", f"diffusion_steps must be within 1-{num_base_steps}")
        history = default_history_frames(fps, model.gen_horizon_len, model.num_frames_per_token)

        seed_everything(seed)
        progress(0.05)
        lengths = torch.tensor([num_frames], device=self.device)
        with torch.no_grad():
            motion = model(
                [text],
                num_frames,
                num_denoising_steps=steps,
                pad_mask=length_to_mask(lengths),
                first_heading_angle=torch.zeros(1, device=self.device),  # +Z を向いて始める
                motion_mask=None,
                observed_motion=None,
                cfg_weight=(2.0, 2.0),  # scripts/generate.py の既定値（テキスト、拘束）
                crop_history_length=history,
            )
            output = model.motion_rep.inverse(motion, is_normalized=True)
            # 足の滑りを減らす後処理（scripts/generate.py と同じ。G1 以外で有効）
            output.update(
                post_process_motion(
                    output["local_rot_mats"], output["root_positions"], output["foot_contacts"], model.skeleton
                )
            )
        progress(0.9)

        output = to_numpy(output)
        single = {k: (v[0] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1 else v) for k, v in output.items()}
        raw_name = "ardy_raw.npz"
        np.savez(output_dir / raw_name, fps=np.asarray(fps), text=np.asarray(text), **single)

        meta = {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "model": self.resolved,
            "prompt": text,
            "seed": seed,
            "duration": duration,
            "diffusion_steps": steps,
            "history_frames": history,
            "raw_outputs": [raw_name],
        }
        source = to_source_motion(single, model.skeleton, fps, meta)
        source.save(output_dir / "source_motion.npz")
        progress(1.0)
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=[raw_name],
            fps=fps,
            num_frames=source.num_frames,
            duration=source.duration,
            seed=seed,
        )


if __name__ == "__main__":
    serve(ArdyAdapter(), port=int(os.environ.get("PORT", "8000")))
