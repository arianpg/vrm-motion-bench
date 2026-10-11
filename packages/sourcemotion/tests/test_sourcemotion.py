import numpy as np
import pytest

from sourcemotion import SourceMotion, SourceMotionError, quat, read_bvh

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 90.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.0 10.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.0 10.0 0.0
    }
  }
  JOINT LeftUpLeg
  {
    OFFSET 10.0 0.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLeg
    {
      OFFSET 0.0 -40.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.0 -40.0 0.0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.05
0 90 0  0 0 0   0 0 0   0 0 0   0 0 0
5 80 0  90 0 0  0 0 0   0 0 0   0 0 0
"""


def make_motion(t: int = 3) -> SourceMotion:
    return SourceMotion(
        joint_names=["root", "a", "b"],
        parents=np.array([-1, 0, 1], dtype=np.int32),
        offsets=np.array([[0, 1, 0], [0, 0.5, 0], [0, 0.5, 0]], dtype=np.float32),
        rest_rotations=np.tile(quat.IDENTITY, (3, 1)),
        local_rotations=np.tile(quat.IDENTITY, (t, 3, 1)),
        root_translation=np.zeros((t, 3), dtype=np.float32),
        fps=30.0,
        up_axis="+Y",
        forward_axis="+Z",
        rest_pose_kind="tpose",
        meta={"prompt": "A person walks.", "seed": 1},
    )


def test_quat_roundtrip_matrix():
    rng = np.random.default_rng(0)
    q = quat.normalize(rng.normal(size=(100, 4)))
    q = np.where(q[:, 3:4] < 0, -q, q)
    assert np.allclose(quat.from_matrix(quat.to_matrix(q)), q, atol=1e-6)


def test_quat_rotate_matches_matrix():
    q = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.array(np.pi / 2))
    assert np.allclose(quat.rotate(q, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(quat.to_matrix(q) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_euler_intrinsic_order():
    # ZXY 内因性: R = Rz * Rx * Ry
    q = quat.from_euler(np.array([30.0, 40.0, 50.0]), "ZXY")
    rz = quat.from_axis_angle(np.array([0.0, 0, 1]), np.radians(30))
    rx = quat.from_axis_angle(np.array([1.0, 0, 0]), np.radians(40))
    ry = quat.from_axis_angle(np.array([0.0, 1, 0]), np.radians(50))
    expected = quat.multiply(quat.multiply(rz, rx), ry)
    assert np.allclose(q, expected, atol=1e-9) or np.allclose(q, -expected, atol=1e-9)


def test_make_continuous():
    q = np.array([[0, 0, 0, 1.0], [0, 0, 0, -1.0], [0, 0, 0.1, -0.995]])
    out = quat.make_continuous(q)
    assert np.all(out[:, 3] > 0)


def test_save_load_roundtrip(tmp_path):
    m = make_motion()
    m.foot_contacts = np.zeros((3, 1), dtype=bool)
    m.contact_joint_names = ["b"]
    path = tmp_path / "source_motion.npz"
    m.save(path)
    loaded = SourceMotion.load(path)
    assert loaded.joint_names == m.joint_names
    assert loaded.meta == m.meta
    assert loaded.contact_joint_names == ["b"]
    assert np.array_equal(loaded.parents, m.parents)
    assert loaded.fps == pytest.approx(30.0)
    # pickle を使わずに読めること
    with np.load(path, allow_pickle=False) as z:
        assert str(z["format_version"]) == "1"


def test_validate_rejects_bad_parent_order():
    m = make_motion()
    m.parents = np.array([-1, 2, 0], dtype=np.int32)
    with pytest.raises(SourceMotionError):
        m.validate()


def test_validate_rejects_unnormalized():
    m = make_motion()
    m.local_rotations = m.local_rotations * 2
    with pytest.raises(SourceMotionError):
        m.validate()


def test_global_positions_chain():
    m = make_motion(t=1)
    # a を Z 軸まわりに 90 度回すと、b は -X 方向へ
    m.local_rotations[0, 1] = quat.from_axis_angle(np.array([0.0, 0, 1]), np.array(np.pi / 2))
    pos = m.global_positions()
    assert np.allclose(pos[0, 2], [-0.5, 0.5, 0.0], atol=1e-6)
    assert np.allclose(m.rest_positions()[2], [0.0, 2.0, 0.0], atol=1e-6)


def test_read_bvh(tmp_path):
    path = tmp_path / "t.bvh"
    path.write_text(BVH)
    m = read_bvh(path, unit_scale=0.01, up_axis="+Y", forward_axis="+Z", rest_pose_kind="tpose")
    assert m.joint_names == ["Hips", "Spine", "LeftUpLeg", "LeftLeg"]
    assert list(m.parents) == [-1, 0, 0, 2]
    assert m.fps == pytest.approx(20.0)
    assert np.allclose(m.offsets[3], [0.0, -0.4, 0.0])
    assert np.allclose(m.root_translation[1], [0.05, 0.8, 0.0])
    # 2 フレーム目は Hips を Z 軸まわりに 90 度 → Spine の先は -X 方向へ
    pos = m.global_positions()
    assert np.allclose(pos[1, 1], [0.05 - 0.1, 0.8, 0.0], atol=1e-6)
