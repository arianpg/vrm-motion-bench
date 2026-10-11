"""NVIDIA 系モーションモデル（Kimodo、ARDY）の出力を SourceMotion に変換する。

両者は同じ骨格クラスの API（bone_order_names、joint_parents、neutral_joints、root_idx、
foot_joint_names）と出力形式（local_rot_mats、root_positions、foot_contacts など）を持つ。
座標系は Y-up・+Z 前方・メートルで、回転ゼロの姿勢が標準 T ポーズ（関節位置が neutral_joints）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sourcemotion import SourceMotion, quat

from .server import AdapterError


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def to_source_motion(output: dict, skeleton, fps: float, meta: dict, contact_skeletons: tuple = ()) -> SourceMotion:
    """output: 1 サンプル分（先頭のバッチ次元を外した numpy 配列）。

    contact_skeletons: 接地フラグの関節名を探す骨格（出力の骨格の次に試す）。
    """
    names = list(skeleton.bone_order_names)
    parents = _np(skeleton.joint_parents).astype(np.int32)
    neutral = _np(skeleton.neutral_joints).astype(np.float64)  # [J, 3]
    if int(skeleton.root_idx) != 0:
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
        contacts = np.asarray(contacts) > 0.5
        for skel in (skeleton, *contact_skeletons):
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
