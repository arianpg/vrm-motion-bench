"""クォータニオン演算（xyzw、スカラー最後）。すべて numpy 配列でブロードキャスト可能。"""

from __future__ import annotations

import numpy as np

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)


def inverse(q: np.ndarray) -> np.ndarray:
    """単位クォータニオンの逆。"""
    return conjugate(q)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a * b（b を先に適用し、次に a を適用する回転）。"""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ベクトル v を q で回転する。"""
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def from_matrix(m: np.ndarray) -> np.ndarray:
    """回転行列 [..., 3, 3] → クォータニオン [..., 4]（xyzw）。"""
    m = np.asarray(m, dtype=np.float64)
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # 4 通りの候補から数値的に安定なものを選ぶ
    cands = np.stack(
        [
            np.stack([1 + m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12], -1),
            np.stack([m01 + m10, 1 - m00 + m11 - m22, m12 + m21, m02 - m20], -1),
            np.stack([m02 + m20, m12 + m21, 1 - m00 - m11 + m22, m10 - m01], -1),
            np.stack([m21 - m12, m02 - m20, m10 - m01, 1 + m00 + m11 + m22], -1),
        ],
        axis=-2,
    )
    traces = np.stack([m00 - m11 - m22, -m00 + m11 - m22, -m00 - m11 + m22, m00 + m11 + m22], -1)
    best = np.argmax(traces, axis=-1)
    q = np.take_along_axis(cands, best[..., None, None], axis=-2)[..., 0, :]
    q = normalize(q)
    # w >= 0 に揃える
    return np.where(q[..., 3:4] < 0, -q, q)


def to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    half = angle[..., None] / 2.0
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def from_euler(angles_deg: np.ndarray, order: str) -> np.ndarray:
    """内因性（intrinsic）のオイラー角 → クォータニオン。

    order は BVH のチャンネル順（例 "ZXY"）。BVH では、チャンネル順に回転行列を右から
    掛けていく（R = R1 * R2 * R3）ため、内因性回転として解釈する。
    """
    angles = np.radians(np.asarray(angles_deg, dtype=np.float64))
    basis = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}
    q = np.broadcast_to(IDENTITY.astype(np.float64), angles.shape[:-1] + (4,))
    for i, axis_name in enumerate(order.upper()):
        qi = from_axis_angle(np.array(basis[axis_name]), angles[..., i])
        q = multiply(q, qi)
    return normalize(q)


def make_continuous(q: np.ndarray, axis: int = 0) -> np.ndarray:
    """時間軸方向で q と -q の反転をなくす（隣接フレームとの内積が負なら符号を反転）。"""
    q = np.array(q, copy=True)
    q = np.moveaxis(q, axis, 0)
    for t in range(1, q.shape[0]):
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0
        q[t][flip] *= -1
    return np.moveaxis(q, 0, axis)


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot)
    close = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe = np.where(close, 1.0, sin_theta)
    wa = np.where(close, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(close, t, np.sin(t * theta) / safe)
    return normalize(wa * a + wb * b)
