"""パイプライン検証用: Z-up・-Y 前方・A ポーズの小さな骨格で、歩くような合成モーションを返す。"""

import os
import random
import time
from pathlib import Path

import numpy as np

from sourcemotion import SourceMotion, quat
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

# (名前, 親, 親からのオフセット) Z-up・-Y 前方。キャラクターの左は +X（上 × 前 = +X）
JOINTS = [
    ("Pelvis", None, (0, 0, 1.0)),
    ("Spine", "Pelvis", (0, 0, 0.12)),
    ("Chest", "Spine", (0, 0, 0.22)),
    ("Neck", "Chest", (0, 0, 0.22)),
    ("Head", "Neck", (0, 0, 0.1)),
    ("L_Arm", "Chest", (0.2, 0, 0.18)),
    ("L_Elbow", "L_Arm", (0.3, 0, 0)),
    ("L_Wrist", "L_Elbow", (0.27, 0, 0)),
    ("R_Arm", "Chest", (-0.2, 0, 0.18)),
    ("R_Elbow", "R_Arm", (-0.3, 0, 0)),
    ("R_Wrist", "R_Elbow", (-0.27, 0, 0)),
    ("L_Hip", "Pelvis", (0.1, 0, -0.05)),
    ("L_Knee", "L_Hip", (0, 0, -0.45)),
    ("L_Ankle", "L_Knee", (0, 0, -0.42)),
    ("R_Hip", "Pelvis", (-0.1, 0, -0.05)),
    ("R_Knee", "R_Hip", (0, 0, -0.45)),
    ("R_Ankle", "R_Knee", (0, 0, -0.42)),
]
FPS = 20.0


class DummyAdapter(Adapter):
    def load(self) -> None:
        time.sleep(float(os.environ.get("DUMMY_LOAD_SECONDS", "2")))

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        text = str(request.inputs.get("text", "")).strip()
        if not text:
            raise AdapterError("invalid_input", "inputs.text is required")
        duration = float(request.params.get("duration", 3.0))
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        rng = np.random.default_rng(seed)
        n = int(round(duration * FPS)) + 1
        names = [j[0] for j in JOINTS]
        index = {name: i for i, name in enumerate(names)}
        parents = np.array([-1 if p is None else index[p] for _, p, _ in JOINTS], np.int32)
        offsets = np.array([o for _, _, o in JOINTS], np.float32)

        # A ポーズ: 腕を 40 度下げたレスト（左腕 +X を下げるのは Y 軸まわり +、右腕 -X は -）
        rest = np.tile(quat.IDENTITY, (len(names), 1)).astype(np.float64)
        y = np.array([0.0, 1.0, 0.0])
        rest[index["L_Arm"]] = quat.from_axis_angle(y, np.array(np.radians(40)))
        rest[index["R_Arm"]] = quat.from_axis_angle(y, np.array(np.radians(-40)))

        t = np.arange(n) / FPS
        freq = 1.0 + rng.uniform(-0.1, 0.1)
        swing = np.sin(2 * np.pi * freq * t) * np.radians(30)
        x = np.array([1.0, 0.0, 0.0])
        local = np.tile(rest, (n, 1, 1))
        # 脚は X 軸まわりに前後に振る（前 = -Y へ振るのは X 軸まわり -）。膝は後ろ（+）に曲げる
        local[:, index["L_Hip"]] = quat.from_axis_angle(x, swing)
        local[:, index["R_Hip"]] = quat.from_axis_angle(x, -swing)
        local[:, index["L_Knee"]] = quat.from_axis_angle(x, np.clip(swing, 0, None) * 1.2)
        local[:, index["R_Knee"]] = quat.from_axis_angle(x, np.clip(-swing, 0, None) * 1.2)
        # 腕は脚と逆位相で前後に振る
        local[:, index["L_Arm"]] = quat.multiply(quat.from_axis_angle(x, -swing), rest[index["L_Arm"]])
        local[:, index["R_Arm"]] = quat.multiply(quat.from_axis_angle(x, swing), rest[index["R_Arm"]])
        root = np.zeros((n, 3))
        root[:, 1] = -1.0 * t  # -Y 方向（前）へ 1 m/s
        root[:, 2] = 1.0 + 0.02 * np.cos(4 * np.pi * freq * t)

        progress(0.5)
        motion = SourceMotion(
            joint_names=names,
            parents=parents,
            offsets=offsets,
            rest_rotations=rest.astype(np.float32),
            local_rotations=quat.make_continuous(local).astype(np.float32),
            root_translation=root.astype(np.float32),
            fps=FPS,
            up_axis="+Z",
            forward_axis="-Y",
            rest_pose_kind="apose",
            meta={"adapter_id": "dummy", "prompt": text, "seed": seed},
        )
        motion.save(output_dir / "source_motion.npz")
        reference = request.inputs.get("reference")
        ref_info = f"reference={reference} size={Path(reference).stat().st_size}" if reference else "reference=none"
        (output_dir / "dummy_raw.txt").write_text(f"{text}\nseed={seed}\n{ref_info}\n")
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=["dummy_raw.txt"],
            fps=FPS,
            num_frames=n,
            duration=motion.duration,
            seed=seed,
        )


if __name__ == "__main__":
    serve(DummyAdapter(), port=int(os.environ.get("PORT", "8000")))
