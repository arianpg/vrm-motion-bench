from types import SimpleNamespace

import numpy as np

from sourcemotion import quat
from vmb_adapter.nv_motion import to_source_motion


def fake_skeleton():
    # Hips → LeftLeg → LeftFoot（足裏は y=-0.9）
    return SimpleNamespace(
        bone_order_names=["Hips", "LeftLeg", "LeftFoot"],
        joint_parents=np.array([-1, 0, 1]),
        neutral_joints=np.array([[0, 0, 0], [0.1, -0.05, 0], [0.1, -0.9, 0]], dtype=np.float32),
        root_idx=0,
        foot_joint_names=["LeftFoot"],
    )


def test_to_source_motion_matches_forward_kinematics():
    t = 4
    rot = quat.to_matrix(quat.from_axis_angle(np.array([1.0, 0, 0]), np.linspace(0, 1, t)))  # [T, 3, 3]
    local = np.tile(np.eye(3), (t, 3, 1, 1))
    local[:, 1] = rot
    root = np.stack([np.zeros(t), np.full(t, 0.9), np.linspace(0, 1, t)], axis=1)
    output = {"local_rot_mats": local, "root_positions": root, "foot_contacts": np.array([[1.0], [0.9], [0.1], [0.0]])}

    m = to_source_motion(output, fake_skeleton(), 20.0, {"prompt": "x"})
    assert m.offsets[0][1] == np.float32(0.9)  # 足裏が地面に接する高さ
    assert m.contact_joint_names == ["LeftFoot"]
    assert m.foot_contacts[:, 0].tolist() == [True, True, False, False]
    pos = m.global_positions()
    # 最初のフレーム（回転なし）では足は地面の上
    assert np.allclose(pos[0, 2], [0.1, 0.0, 0.0], atol=1e-5)
    # 脚の回転は足の位置に反映される
    expected = root[-1] + [0.1, -0.05, 0] + rot[-1] @ np.array([0, -0.85, 0])
    assert np.allclose(pos[-1, 2], expected, atol=1e-5)
