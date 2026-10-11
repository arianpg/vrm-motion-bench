"""VRM 1.0（.vrm / glb）からヒューマノイドの骨格情報を読む（numpy と標準ライブラリのみ）。"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import quat

# VRM 1.0 のヒューマノイドボーン（親が先に来る順）
VRM_BONES = [
    "hips", "spine", "chest", "upperChest", "neck", "head", "leftEye", "rightEye", "jaw",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
    "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
    "leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal",
    "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
    "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
    "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
    "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
    "rightThumbMetacarpal", "rightThumbProximal", "rightThumbDistal",
    "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
    "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
    "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
    "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal",
]  # fmt: skip


@dataclass
class VrmHumanoid:
    """ヒューマノイドボーンだけを抜き出した骨格（glTF 座標系: +Y 上、+Z 前方、m）。"""

    bones: list[str]  # VRM_BONES の順で、モデルに存在するもの
    parents: dict[str, str | None]  # 最も近いヒューマノイドの祖先
    rest_positions: dict[str, np.ndarray]  # レスト姿勢でのワールド位置
    node_names: dict[str, str]

    def children(self, bone: str) -> list[str]:
        return [b for b in self.bones if self.parents[b] == bone]


def read_glb_json(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path}: not a glb file")
    length = struct.unpack("<I", data[12:16])[0]
    if data[16:20] != b"JSON":
        raise ValueError(f"{path}: first chunk is not JSON")
    return json.loads(data[20 : 20 + length])


def load_vrm_humanoid(path: str | Path) -> VrmHumanoid:
    gltf = read_glb_json(path)
    ext = gltf.get("extensions", {}).get("VRMC_vrm")
    if ext is None:
        raise ValueError(f"{path}: VRMC_vrm extension not found (VRM 1.0 only)")
    nodes = gltf["nodes"]
    human = {name: v["node"] for name, v in ext["humanoid"]["humanBones"].items()}

    parent_of: dict[int, int] = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent_of[c] = i

    def local_matrix(n: dict) -> np.ndarray:
        if "matrix" in n:
            return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        r = np.array(n.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
        s = np.array(n.get("scale", [1, 1, 1]), dtype=np.float64)
        m[:3, :3] = quat.to_matrix(r) * s
        m[:3, 3] = n.get("translation", [0, 0, 0])
        return m

    world_cache: dict[int, np.ndarray] = {}

    def world(i: int) -> np.ndarray:
        if i not in world_cache:
            m = local_matrix(nodes[i])
            world_cache[i] = world(parent_of[i]) @ m if i in parent_of else m
        return world_cache[i]

    node_to_bone = {v: k for k, v in human.items()}
    bones = [b for b in VRM_BONES if b in human]
    parents: dict[str, str | None] = {}
    for b in bones:
        p = parent_of.get(human[b])
        while p is not None and p not in node_to_bone:
            p = parent_of.get(p)
        parents[b] = node_to_bone.get(p) if p is not None else None
    rest = {b: world(human[b])[:3, 3].copy() for b in bones}
    names = {b: nodes[human[b]].get("name", "") for b in bones}
    return VrmHumanoid(bones=bones, parents=parents, rest_positions=rest, node_names=names)
