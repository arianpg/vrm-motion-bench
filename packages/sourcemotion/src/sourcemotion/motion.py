"""SourceMotion NPZ の読み書きと検証（仕様: docs/spec.md「SourceMotion」）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import quat

FORMAT_VERSION = "1"
AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
REST_POSE_KINDS = ("tpose", "apose", "other")


class SourceMotionError(ValueError):
    pass


@dataclass
class SourceMotion:
    joint_names: list[str]
    parents: np.ndarray  # [J] int32
    offsets: np.ndarray  # [J, 3] float32 (m)
    rest_rotations: np.ndarray  # [J, 4] float32 xyzw
    local_rotations: np.ndarray  # [T, J, 4] float32 xyzw
    root_translation: np.ndarray  # [T, 3] float32 (m)
    fps: float
    up_axis: str
    forward_axis: str
    rest_pose_kind: str
    meta: dict[str, Any] = field(default_factory=dict)
    foot_contacts: np.ndarray | None = None  # [T, K] bool
    contact_joint_names: list[str] | None = None
    expression_names: list[str] | None = None
    expression_weights: np.ndarray | None = None  # [T, E] float32

    @property
    def num_frames(self) -> int:
        return int(self.local_rotations.shape[0])

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError:
            raise SourceMotionError(f"unknown joint: {name}") from None

    # ------------------------------------------------------------------ 検証

    def validate(self, atol: float = 1e-3) -> None:
        j = self.num_joints
        t = self.num_frames

        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise SourceMotionError(msg)

        check(j > 0, "no joints")
        check(len(set(self.joint_names)) == j, "joint_names must be unique")
        check(self.parents.shape == (j,), f"parents shape {self.parents.shape} != ({j},)")
        check(int(self.parents[0]) == -1, "root (index 0) must have parent -1")
        for i in range(1, j):
            p = int(self.parents[i])
            check(0 <= p < i, f"joint {i} ({self.joint_names[i]}): parent {p} must satisfy 0 <= parent < index")
        check(self.offsets.shape == (j, 3), f"offsets shape {self.offsets.shape} != ({j}, 3)")
        check(self.rest_rotations.shape == (j, 4), f"rest_rotations shape {self.rest_rotations.shape}")
        check(t > 0, "no frames")
        check(self.local_rotations.shape == (t, j, 4), f"local_rotations shape {self.local_rotations.shape}")
        check(self.root_translation.shape == (t, 3), f"root_translation shape {self.root_translation.shape}")
        for name, q in (("rest_rotations", self.rest_rotations), ("local_rotations", self.local_rotations)):
            check(bool(np.all(np.isfinite(q))), f"{name} contains non-finite values")
            norms = np.linalg.norm(q, axis=-1)
            check(bool(np.allclose(norms, 1.0, atol=atol)), f"{name} must be normalized (max err {np.abs(norms - 1).max():.4g})")
        check(bool(np.all(np.isfinite(self.offsets))), "offsets contains non-finite values")
        check(bool(np.all(np.isfinite(self.root_translation))), "root_translation contains non-finite values")
        check(self.fps > 0, "fps must be positive")
        check(self.up_axis in AXES and self.forward_axis in AXES, "up_axis/forward_axis must be one of " + ", ".join(AXES))
        check(self.up_axis[1] != self.forward_axis[1], "up_axis and forward_axis must differ")
        check(self.rest_pose_kind in REST_POSE_KINDS, f"rest_pose_kind must be one of {REST_POSE_KINDS}")
        if self.foot_contacts is not None:
            check(self.contact_joint_names is not None, "contact_joint_names is required with foot_contacts")
            k = len(self.contact_joint_names)
            check(self.foot_contacts.shape == (t, k), f"foot_contacts shape {self.foot_contacts.shape} != ({t}, {k})")
            for name in self.contact_joint_names:
                check(name in self.joint_names, f"contact joint not in joint_names: {name}")
        if self.expression_weights is not None:
            check(self.expression_names is not None, "expression_names is required with expression_weights")
            check(
                self.expression_weights.shape == (t, len(self.expression_names)),
                f"expression_weights shape {self.expression_weights.shape}",
            )

    # ------------------------------------------------------------------ 入出力

    def save(self, path: str | Path) -> None:
        self.validate()
        arrays: dict[str, np.ndarray] = {
            "format_version": np.array(FORMAT_VERSION),
            "joint_names": np.array(self.joint_names, dtype=str),
            "parents": self.parents.astype(np.int32),
            "offsets": self.offsets.astype(np.float32),
            "rest_rotations": self.rest_rotations.astype(np.float32),
            "local_rotations": self.local_rotations.astype(np.float32),
            "root_translation": self.root_translation.astype(np.float32),
            "fps": np.array(self.fps, dtype=np.float32),
            "up_axis": np.array(self.up_axis),
            "forward_axis": np.array(self.forward_axis),
            "rest_pose_kind": np.array(self.rest_pose_kind),
            "meta_json": np.array(json.dumps(self.meta, ensure_ascii=False)),
        }
        if self.foot_contacts is not None:
            arrays["foot_contacts"] = self.foot_contacts.astype(bool)
            arrays["contact_joint_names"] = np.array(self.contact_joint_names, dtype=str)
        if self.expression_weights is not None:
            arrays["expression_names"] = np.array(self.expression_names, dtype=str)
            arrays["expression_weights"] = self.expression_weights.astype(np.float32)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:  # savez はパスに .npz を補うため、ファイルオブジェクトで書く
            np.savez_compressed(f, **arrays)

    @classmethod
    def load(cls, path: str | Path, validate: bool = True) -> "SourceMotion":
        with np.load(path, allow_pickle=False) as z:
            version = str(z["format_version"])
            if version != FORMAT_VERSION:
                raise SourceMotionError(f"unsupported format_version: {version}")
            has_contacts = "foot_contacts" in z
            has_expr = "expression_weights" in z
            motion = cls(
                joint_names=[str(n) for n in z["joint_names"]],
                parents=z["parents"].astype(np.int32),
                offsets=z["offsets"].astype(np.float32),
                rest_rotations=z["rest_rotations"].astype(np.float32),
                local_rotations=z["local_rotations"].astype(np.float32),
                root_translation=z["root_translation"].astype(np.float32),
                fps=float(z["fps"]),
                up_axis=str(z["up_axis"]),
                forward_axis=str(z["forward_axis"]),
                rest_pose_kind=str(z["rest_pose_kind"]),
                meta=json.loads(str(z["meta_json"])),
                foot_contacts=z["foot_contacts"].astype(bool) if has_contacts else None,
                contact_joint_names=[str(n) for n in z["contact_joint_names"]] if has_contacts else None,
                expression_names=[str(n) for n in z["expression_names"]] if has_expr else None,
                expression_weights=z["expression_weights"].astype(np.float32) if has_expr else None,
            )
        if validate:
            motion.validate()
        return motion

    # ------------------------------------------------------------------ 計算

    def global_rotations(self, local: np.ndarray | None = None) -> np.ndarray:
        """各関節のグローバル回転。local を省略すると local_rotations（[T, J, 4]）を使う。"""
        local = self.local_rotations if local is None else local
        out = np.empty(local.shape, dtype=np.float64)
        for j in range(self.num_joints):
            p = int(self.parents[j])
            out[..., j, :] = local[..., j, :] if p < 0 else quat.multiply(out[..., p, :], local[..., j, :])
        return out

    def rest_global_rotations(self) -> np.ndarray:
        return self.global_rotations(self.rest_rotations)

    def global_positions(self, local: np.ndarray | None = None, root: np.ndarray | None = None) -> np.ndarray:
        """各関節のワールド位置 [T, J, 3]（引数省略時はアニメーション）。"""
        local = self.local_rotations if local is None else local
        root = self.root_translation if root is None else root
        grot = self.global_rotations(local)
        pos = np.empty(local.shape[:-1] + (3,), dtype=np.float64)
        pos[..., 0, :] = root
        for j in range(1, self.num_joints):
            p = int(self.parents[j])
            pos[..., j, :] = pos[..., p, :] + quat.rotate(grot[..., p, :], self.offsets[j].astype(np.float64))
        return pos

    def rest_positions(self) -> np.ndarray:
        """レスト姿勢での関節位置 [J, 3]。ルートは offsets[0] に置く。"""
        return self.global_positions(self.rest_rotations, self.offsets[0].astype(np.float64))
