"""SourceMotion → VRM ヒューマノイドへのワールド空間リターゲット（仕様: docs/spec.md「共通コンバーター」）。

出力は「VRM 正規化ボーン」の形式で、glTF 座標系（+Y 上、+Z 前方、m）:
    - 各ボーンのレスト時のワールド回転を単位元とみなしたローカル回転 L_b
    - hips のワールド位置
three-vrm の正規化ボーンや、全ノードの回転が単位元の T ポーズ階層を持つ VRMA と同じ意味になる。

計算:
    D_j  = G_j(t) * G_j(rest)^-1             元骨格の関節 j のワールド回転差分（座標系は glTF に変換）
    A_b  = 対象のレスト方向 → 元のレスト方向 への最短回転（レスト姿勢の違いの補正。T ポーズ同士ならほぼ単位元）
    W_b  = D_j * A_b                           対応表に載ったボーン b のワールド回転
    W_b  = W_parent(b)                         対応表に載っていないボーン（親に追従）
    L_b  = W_parent(b)^-1 * W_b
腰の移動量は、腰の高さ（足元から）の比で縮尺する。地面は元骨格・対象とも up 軸の 0 とする。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quat
from .motion import SourceMotion
from .vrm import VrmHumanoid

TARGET_FPS = 30.0

_AXIS_VECTORS = {
    "+X": (1, 0, 0), "-X": (-1, 0, 0),
    "+Y": (0, 1, 0), "-Y": (0, -1, 0),
    "+Z": (0, 0, 1), "-Z": (0, 0, -1),
}  # fmt: skip

# レスト方向を測るときに使う子ボーンの優先順
_PRIMARY_CHILD = {
    "hips": ["spine"],
    "spine": ["chest", "upperChest", "neck"],
    "chest": ["upperChest", "neck"],
    "upperChest": ["neck"],
    "neck": ["head"],
    "leftHand": ["leftMiddleProximal", "leftIndexProximal", "leftRingProximal"],
    "rightHand": ["rightMiddleProximal", "rightIndexProximal", "rightRingProximal"],
}
# ワールド回転の補正をしないボーン（体全体の向きを表す）
_NO_ALIGN = {"hips"}


class RetargetError(ValueError):
    pass


@dataclass
class RetargetResult:
    bones: list[str]
    parents: dict[str, str | None]
    local_rotations: np.ndarray  # [T, B, 4] xyzw（VRM 正規化ボーン、glTF 座標系）
    hips_translation: np.ndarray  # [T, 3]（glTF 座標系、m）
    fps: float
    hips_scale: float
    mapped: dict[str, str]  # VRM ボーン → 元骨格の関節
    contacts: dict[str, np.ndarray] | None = None  # VRM ボーン → [T] bool（接地フラグがある場合）
    expressions: dict[str, np.ndarray] | None = None  # VRM の表情名 → [T] 0〜1（表情がある場合）

    @property
    def num_frames(self) -> int:
        return int(self.local_rotations.shape[0])

    def world_rotations(self) -> np.ndarray:
        out = np.empty_like(self.local_rotations)
        index = {b: i for i, b in enumerate(self.bones)}
        for i, b in enumerate(self.bones):
            p = self.parents[b]
            out[:, i] = self.local_rotations[:, i] if p is None else quat.multiply(out[:, index[p]], self.local_rotations[:, i])
        return out


def basis_to_gltf(up_axis: str, forward_axis: str) -> np.ndarray:
    """元骨格の座標 → glTF 座標（+Y 上、+Z 前方）の回転行列。"""
    u = np.array(_AXIS_VECTORS[up_axis], dtype=np.float64)
    f = np.array(_AXIS_VECTORS[forward_axis], dtype=np.float64)
    return np.stack([np.cross(u, f), u, f])


def shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ベクトル a を b に向ける最短回転（xyzw）。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d > 1 - 1e-9:
        return quat.IDENTITY.astype(np.float64)
    if d < -1 + 1e-9:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        return quat.from_axis_angle(axis / np.linalg.norm(axis), np.array(np.pi))
    q = np.concatenate([np.cross(a, b), [1.0 + d]])
    return quat.normalize(q)


def retarget(
    motion: SourceMotion, bone_map: dict[str, str], target: VrmHumanoid, target_fps: float = TARGET_FPS
) -> RetargetResult:
    """bone_map: 元骨格の関節名 → VRM ボーン名。"""
    # VRM ボーン → 元骨格の関節インデックス（対象に存在するボーンだけ）
    mapped: dict[str, int] = {}
    for joint, bone in bone_map.items():
        if bone in target.parents and joint in motion.joint_names:
            if bone in mapped:
                raise RetargetError(f"VRM bone {bone} is mapped twice")
            mapped[bone] = motion.joint_index(joint)
    if "hips" not in mapped:
        raise RetargetError("bone_map must map a joint to hips")

    c = basis_to_gltf(motion.up_axis, motion.forward_axis)
    cq = quat.from_matrix(c)
    cq_inv = quat.inverse(cq)

    # 元骨格: レストと各フレームのグローバル回転・位置（glTF 座標系）
    g_rest = motion.rest_global_rotations()
    g_anim = motion.global_rotations()
    delta = quat.multiply(g_anim, quat.inverse(g_rest)[None])
    delta = quat.multiply(quat.multiply(cq, delta), cq_inv)
    src_rest_pos = motion.rest_positions() @ c.T
    src_pos = motion.global_positions() @ c.T

    tgt_rest = target.rest_positions

    # レスト姿勢の補正 A_b
    align: dict[str, np.ndarray] = {}
    for b in target.bones:
        parent = target.parents[b]
        inherited = align.get(parent, quat.IDENTITY.astype(np.float64)) if parent else quat.IDENTITY.astype(np.float64)
        if b not in mapped or b in _NO_ALIGN:
            align[b] = inherited if b not in _NO_ALIGN else quat.IDENTITY.astype(np.float64)
            continue
        children = [ch for ch in target.children(b) if ch in mapped]
        preferred = [ch for ch in _PRIMARY_CHILD.get(b, []) if ch in children]
        child = (preferred or children or [None])[0]
        if child is None:
            align[b] = inherited  # 末端は親の補正を引き継ぐ
            continue
        t_dir = tgt_rest[child] - tgt_rest[b]
        s_dir = src_rest_pos[mapped[child]] - src_rest_pos[mapped[b]]
        if np.linalg.norm(t_dir) < 1e-6 or np.linalg.norm(s_dir) < 1e-6:
            align[b] = inherited
            continue
        align[b] = shortest_arc(t_dir, s_dir)

    # ワールド回転 → ローカル回転
    t = motion.num_frames
    world: dict[str, np.ndarray] = {}
    local = np.empty((t, len(target.bones), 4), dtype=np.float64)
    for i, b in enumerate(target.bones):
        parent = target.parents[b]
        if b in mapped:
            world[b] = quat.multiply(delta[:, mapped[b]], align[b][None])
        else:
            world[b] = world[parent] if parent else np.tile(quat.IDENTITY, (t, 1)).astype(np.float64)
        local[:, i] = world[b] if parent is None else quat.multiply(quat.inverse(world[parent]), world[b])
    local = quat.make_continuous(quat.normalize(local))

    # 腰の移動量
    src_min_y = src_rest_pos[:, 1].min()
    tgt_min_y = min(p[1] for p in tgt_rest.values())
    src_h = src_rest_pos[mapped["hips"], 1] - src_min_y
    tgt_h = tgt_rest["hips"][1] - tgt_min_y
    if src_h <= 1e-6 or tgt_h <= 1e-6:
        raise RetargetError("cannot determine hips height")
    scale = float(tgt_h / src_h)
    hips = src_pos[:, mapped["hips"]] * scale

    contacts = _map_contacts(motion, mapped, target_fps)
    expressions = _resample_expressions(motion, target_fps)
    local, hips = resample(local, hips, motion.fps, target_fps)
    return RetargetResult(
        contacts=contacts,
        expressions=expressions,
        bones=list(target.bones),
        parents=dict(target.parents),
        local_rotations=local.astype(np.float32),
        hips_translation=hips.astype(np.float32),
        fps=target_fps,
        hips_scale=scale,
        mapped={b: motion.joint_names[j] for b, j in mapped.items()},
    )


def _map_contacts(motion: SourceMotion, mapped: dict[str, int], target_fps: float) -> dict[str, np.ndarray] | None:
    """接地フラグを VRM ボーンに対応付けて fps を変換する（最も近いフレームを取る）。

    対応表に載っていない関節（つま先の先端など）は、対応表に載っている最も近い祖先にまとめる。
    """
    if motion.foot_contacts is None or not motion.contact_joint_names:
        return None
    joint_to_bone = {j: b for b, j in mapped.items()}
    out: dict[str, np.ndarray] = {}
    for k, name in enumerate(motion.contact_joint_names):
        j = motion.joint_index(name)
        while j >= 0 and j not in joint_to_bone:
            j = int(motion.parents[j])
        if j < 0:
            continue
        bone = joint_to_bone[j]
        flags = motion.foot_contacts[:, k].astype(bool)
        out[bone] = out[bone] | flags if bone in out else flags
    if not out:
        return None
    n = motion.num_frames
    m = int(round((n - 1) / motion.fps * target_fps)) + 1 if n > 1 else 1
    idx = np.rint(np.linspace(0, n - 1, m)).astype(int) if m > 1 else np.zeros(1, int)
    return {b: f[idx] for b, f in out.items()}


def _resample_expressions(motion: SourceMotion, target_fps: float) -> dict[str, np.ndarray] | None:
    """表情の重み（VRM の表情名ごと）を線形補間で target_fps にする。値は 0〜1 に収める。"""
    if motion.expression_weights is None or not motion.expression_names:
        return None
    n = motion.num_frames
    m = int(round((n - 1) / motion.fps * target_fps)) + 1 if n > 1 else 1
    src_t = np.arange(n) / motion.fps
    dst_t = np.arange(m) / target_fps
    w = motion.expression_weights.astype(np.float64)
    return {
        name: np.clip(np.interp(dst_t, src_t, w[:, k]), 0.0, 1.0).astype(np.float32)
        for k, name in enumerate(motion.expression_names)
    }


def resample(rot: np.ndarray, pos: np.ndarray, src_fps: float, dst_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """回転は slerp、位置は線形補間で fps を変換する。尺（最後のフレームの時刻）は保つ。"""
    n = rot.shape[0]
    if n == 1 or abs(src_fps - dst_fps) < 1e-6:
        return rot, pos
    duration = (n - 1) / src_fps
    m = int(round(duration * dst_fps)) + 1
    src_idx = np.linspace(0.0, n - 1, m) if m > 1 else np.zeros(1)
    lo = np.floor(src_idx).astype(int).clip(0, n - 1)
    hi = (lo + 1).clip(0, n - 1)
    w = src_idx - lo
    rot_out = quat.slerp(rot[lo], rot[hi], np.broadcast_to(w[:, None], (m, rot.shape[1])))
    pos_out = pos[lo] * (1 - w[:, None]) + pos[hi] * w[:, None]
    return quat.make_continuous(rot_out), pos_out
