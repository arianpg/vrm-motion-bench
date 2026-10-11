"""Kimodo アダプター。Kimodo の出力（SOMA77）を SourceMotion NPZ に変換して返す。"""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch

from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve
from vmb_adapter.nv_motion import to_source_motion

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
        motion = to_source_motion(single, self.skeleton, fps, meta, contact_skeletons=(self.contact_skeleton,))
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


if __name__ == "__main__":
    serve(KimodoAdapter(), port=int(os.environ.get("PORT", "8000")))
