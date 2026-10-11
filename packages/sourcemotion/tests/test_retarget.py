from pathlib import Path

import numpy as np
import pytest

from sourcemotion import SourceMotion, quat
from sourcemotion.retarget import basis_to_gltf, retarget
from sourcemotion.vrm import load_vrm_humanoid

STICKMAN = Path(__file__).resolve().parents[3] / "assets/stickman/stickman.vrm"


@pytest.fixture(scope="module")
def target():
    return load_vrm_humanoid(STICKMAN)


def motion_from_humanoid(target, frames: int, fps: float = 30.0, rng=None) -> SourceMotion:
    """スティックマン自身の骨格を、VRM ボーン名の関節を持つ SourceMotion にする。"""
    names = target.bones
    index = {b: i for i, b in enumerate(names)}
    parents = np.array([-1 if target.parents[b] is None else index[target.parents[b]] for b in names], np.int32)
    rest = np.array([target.rest_positions[b] for b in names])
    offsets = rest.copy()
    offsets[1:] = rest[1:] - rest[parents[1:]]
    local = np.tile(quat.IDENTITY, (frames, len(names), 1)).astype(np.float64)
    if rng is not None:
        axis = rng.normal(size=(frames, len(names), 3))
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        local = quat.from_axis_angle(axis, rng.uniform(-0.6, 0.6, size=(frames, len(names))))
    root = np.tile(rest[0], (frames, 1)) + np.linspace(0, 1, frames)[:, None] * [0.5, -0.1, 1.0]
    return SourceMotion(
        joint_names=list(names),
        parents=parents,
        offsets=offsets.astype(np.float32),
        rest_rotations=np.tile(quat.IDENTITY, (len(names), 1)),
        local_rotations=quat.normalize(local).astype(np.float32),
        root_translation=root.astype(np.float32),
        fps=fps,
        up_axis="+Y",
        forward_axis="+Z",
        rest_pose_kind="tpose",
    )


def test_target_hierarchy(target):
    assert target.parents["hips"] is None
    assert target.parents["leftUpperArm"] == "leftShoulder"
    assert target.parents["neck"] == "chest"  # スティックマンには upperChest がない
    assert target.rest_positions["hips"][1] == pytest.approx(0.85, abs=1e-4)
    # VRM 1.0: キャラクターの左は +X
    assert target.rest_positions["leftHand"][0] > 0.5


def test_self_retarget_is_identity(target):
    rng = np.random.default_rng(0)
    motion = motion_from_humanoid(target, frames=5, rng=rng)
    result = retarget(motion, {b: b for b in target.bones}, target)
    assert result.num_frames == 5
    assert result.hips_scale == pytest.approx(1.0, abs=1e-6)
    expected = quat.make_continuous(motion.local_rotations.astype(np.float64))
    dots = np.abs(np.sum(result.local_rotations * expected, axis=-1))
    assert np.all(dots > 1 - 1e-5)
    assert np.allclose(result.hips_translation, motion.root_translation, atol=1e-5)


def test_axis_conversion(target):
    """Z-up・-Y 前方（Blender 流）で表した同じモーションが、Y-up と同じ結果になる。"""
    rng = np.random.default_rng(1)
    motion = motion_from_humanoid(target, frames=3, rng=rng)
    ref = retarget(motion, {b: b for b in target.bones}, target)

    c = basis_to_gltf("+Z", "-Y")  # Z-up 座標 → glTF
    c_inv = c.T
    cq_inv = quat.from_matrix(c_inv)
    zup = motion_from_humanoid(target, frames=3)
    zup.offsets = (motion.offsets @ c_inv.T).astype(np.float32)
    zup.root_translation = (motion.root_translation @ c_inv.T).astype(np.float32)
    # レスト回転が単位元なので、各関節の座標系はワールドと平行。全関節のローカル回転を座標変換する
    zup.local_rotations = quat.multiply(
        quat.multiply(cq_inv, motion.local_rotations.astype(np.float64)), quat.inverse(cq_inv)
    ).astype(np.float32)
    zup.rest_rotations = motion.rest_rotations.copy()
    zup.up_axis, zup.forward_axis = "+Z", "-Y"
    out = retarget(zup, {b: b for b in target.bones}, target)
    assert np.allclose(out.hips_translation, ref.hips_translation, atol=1e-5)
    assert np.all(np.abs(np.sum(out.world_rotations() * ref.world_rotations(), axis=-1)) > 1 - 1e-5)


def test_apose_source_is_aligned(target):
    """レストが A ポーズ（腕を 45 度下げた姿勢）の元骨格: 回転ゼロのフレームで、対象の腕も 45 度下がる。"""
    motion = motion_from_humanoid(target, frames=1)
    i = motion.joint_index("leftUpperArm")
    # 腕を下げる: 左腕（+X 方向）を -Y へ → Z 軸まわりに -45 度
    down = quat.from_axis_angle(np.array([0, 0, 1.0]), np.array(-np.pi / 4))
    motion.rest_rotations = motion.rest_rotations.copy()
    motion.rest_rotations[i] = down
    motion.local_rotations = motion.local_rotations.copy()
    motion.local_rotations[0, i] = down
    motion.rest_pose_kind = "apose"
    result = retarget(motion, {b: b for b in target.bones}, target)
    world = result.world_rotations()[0, result.bones.index("leftUpperArm")]
    arm_dir = target.rest_positions["leftLowerArm"] - target.rest_positions["leftUpperArm"]
    posed = quat.rotate(world, arm_dir / np.linalg.norm(arm_dir))
    assert np.allclose(posed, [np.sqrt(0.5), -np.sqrt(0.5), 0.0], atol=1e-5)
    # 手は腕に追従する（手首から先も下を向く）
    hand_world = result.world_rotations()[0, result.bones.index("leftHand")]
    # 手の補正は手→中指の方向で測るため、腕と完全には一致しない（1 度程度）
    assert np.allclose(quat.rotate(hand_world, np.array([1.0, 0, 0])), posed, atol=2e-2)


def test_resample_to_30fps(target):
    motion = motion_from_humanoid(target, frames=21, fps=20.0, rng=np.random.default_rng(2))
    result = retarget(motion, {b: b for b in target.bones}, target)
    assert result.fps == 30.0
    assert result.num_frames == 31  # 1 秒（20 fps で 21 フレーム）→ 30 fps で 31 フレーム
    assert np.allclose(result.hips_translation[0], motion.root_translation[0], atol=1e-5)
    assert np.allclose(result.hips_translation[-1], motion.root_translation[-1], atol=1e-5)


def test_hips_scale(target):
    motion = motion_from_humanoid(target, frames=2)
    motion.offsets = (motion.offsets * 2).astype(np.float32)
    motion.root_translation = (motion.root_translation * 2).astype(np.float32)
    result = retarget(motion, {b: b for b in target.bones}, target)
    assert result.hips_scale == pytest.approx(0.5, abs=1e-6)
    assert result.hips_translation[0][1] == pytest.approx(0.85, abs=1e-4)


def test_contacts_mapped_to_nearest_mapped_ancestor(target):
    motion = motion_from_humanoid(target, frames=21, fps=20.0)
    motion.foot_contacts = np.zeros((21, 2), dtype=bool)
    motion.foot_contacts[:10, 0] = True
    motion.foot_contacts[15:, 1] = True
    motion.contact_joint_names = ["leftToes", "rightFoot"]
    # leftToes を対応表から外すと、祖先の leftFoot にまとめられる
    bone_map = {b: b for b in target.bones if b != "leftToes"}
    result = retarget(motion, bone_map, target)
    assert set(result.contacts) == {"leftFoot", "rightFoot"}
    assert len(result.contacts["leftFoot"]) == result.num_frames == 31
    assert result.contacts["leftFoot"][0] and not result.contacts["leftFoot"][-1]
    assert result.contacts["rightFoot"][-1] and not result.contacts["rightFoot"][0]
