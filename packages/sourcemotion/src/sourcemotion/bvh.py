"""BVH → SourceMotion 変換。

BVH のレスト姿勢は「全関節の回転が単位元で、OFFSET だけで決まる姿勢」なので、
rest_rotations はすべて単位クォータニオンになる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import quat
from .motion import SourceMotion

_ROT_CHANNELS = {"Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z"}
_POS_CHANNELS = {"Xposition": 0, "Yposition": 1, "Zposition": 2}


class BvhError(ValueError):
    pass


def _tokenize(text: str) -> list[str]:
    return text.replace("{", " { ").replace("}", " } ").split()


def read_bvh(
    path: str | Path,
    *,
    unit_scale: float,
    up_axis: str,
    forward_axis: str,
    rest_pose_kind: str,
    meta: dict | None = None,
) -> SourceMotion:
    """BVH を読み込む。

    unit_scale: BVH の長さ単位をメートルに換算する係数（cm なら 0.01）。
    up_axis / forward_axis / rest_pose_kind: BVH 自体には含まれないため、呼び出し側が指定する。
    """
    tokens = _tokenize(Path(path).read_text())
    pos = 0

    def next_token() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise BvhError("unexpected end of file")
        tok = tokens[pos]
        pos += 1
        return tok

    def expect(value: str) -> None:
        tok = next_token()
        if tok != value:
            raise BvhError(f"expected {value!r}, got {tok!r}")

    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []  # 関節インデックス順

    def parse_joint(parent: int) -> None:
        name = next_token()
        index = len(names)
        names.append(name)
        parents.append(parent)
        expect("{")
        expect("OFFSET")
        offsets.append([float(next_token()) for _ in range(3)])
        channels.append([])
        joint_channels: list[str] = []
        while True:
            tok = next_token()
            if tok == "CHANNELS":
                count = int(next_token())
                joint_channels = [next_token() for _ in range(count)]
            elif tok == "JOINT":
                parse_joint(index)
            elif tok == "End":
                expect("Site")
                expect("{")
                expect("OFFSET")
                for _ in range(3):
                    next_token()
                expect("}")
            elif tok == "}":
                break
            else:
                raise BvhError(f"unexpected token {tok!r} in joint {name}")
        channels[index] = joint_channels

    expect("HIERARCHY")
    expect("ROOT")
    parse_joint(-1)

    expect("MOTION")
    expect("Frames:")
    num_frames = int(next_token())
    expect("Frame")
    expect("Time:")
    frame_time = float(next_token())
    total_channels = sum(len(c) for c in channels)
    values = np.array([float(t) for t in tokens[pos : pos + num_frames * total_channels]], dtype=np.float64)
    if values.size != num_frames * total_channels:
        raise BvhError(f"expected {num_frames * total_channels} motion values, got {values.size}")
    values = values.reshape(num_frames, total_channels)

    j = len(names)
    local = np.empty((num_frames, j, 4), dtype=np.float64)
    root = np.tile(np.array(offsets[0], dtype=np.float64), (num_frames, 1))
    col = 0
    for i, ch in enumerate(channels):
        rot_cols = [(k, _ROT_CHANNELS[c]) for k, c in enumerate(ch) if c in _ROT_CHANNELS]
        pos_cols = [(k, _POS_CHANNELS[c]) for k, c in enumerate(ch) if c in _POS_CHANNELS]
        if rot_cols:
            order = "".join(axis for _, axis in rot_cols)
            angles = values[:, [col + k for k, _ in rot_cols]]
            local[:, i] = quat.from_euler(angles, order)
        else:
            local[:, i] = quat.IDENTITY
        if i == 0:
            for k, axis in pos_cols:
                root[:, axis] = values[:, col + k]
        # 非ルート関節の位置チャンネルは無視する（オフセットは固定とみなす）
        col += len(ch)

    motion = SourceMotion(
        joint_names=names,
        parents=np.array(parents, dtype=np.int32),
        offsets=(np.array(offsets, dtype=np.float64) * unit_scale).astype(np.float32),
        rest_rotations=np.tile(quat.IDENTITY, (j, 1)),
        local_rotations=quat.make_continuous(quat.normalize(local)).astype(np.float32),
        root_translation=(root * unit_scale).astype(np.float32),
        fps=1.0 / frame_time,
        up_axis=up_axis,
        forward_axis=forward_axis,
        rest_pose_kind=rest_pose_kind,
        meta=dict(meta or {}),
    )
    motion.validate()
    return motion

