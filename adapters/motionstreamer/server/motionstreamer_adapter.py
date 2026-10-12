"""MotionStreamer アダプター（テキスト → SMPL-X の体 22 関節）。

公式の推論（demo_t2m.py の text to motion モデル）と同じ手順で、sentence-T5-XXL のテキスト特徴から
LLaMA 系の自己回帰モデルと拡散ヘッドで潜在ベクトルを作り、Causal TAE で 272 次元の表現（30fps）に戻す。
272 次元の表現は回転（6D）を持つので IK は要らない（公式の recover_from_local_rotation と同じ方法で戻す）。

    0:2     腰の水平速度（向きを除いた座標）
    2:8     向きの変化（Y 軸まわり、6D）
    8:74    関節位置（向きを除き、水平位置を原点に寄せた座標）
    74:140  関節速度
    140:272 関節の局所回転（6D。腰は向きを除いたもの）

- 骨格: 表現の関節位置と回転から、関節ごとの親からのオフセットを推定する（全フレームの中央値）。SMPL-X のモデルファイルは使わない
- 尺: 常にモデルが終わりを決める（終わりを表す潜在ベクトルに近づいたら止める。最大 312 フレーム＝10.4 秒）。
  このモデルは目標の長さを受け取らず先頭から順に作るため、尺で止めると動作が途中で切れる。尺の指定は受け取っても使わない
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from sourcemotion import SourceMotion, quat
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

ADAPTER_ID = "motionstreamer"
ADAPTER_VERSION = "0.1.0"
MS_DIR = Path(os.environ.get("MOTIONSTREAMER_DIR", "/opt/motionstreamer"))
CKPTS = Path(os.environ.get("MOTIONSTREAMER_CKPTS", "/models/motionstreamer/ckpts"))
T5_DIR = Path(os.environ.get("MOTIONSTREAMER_T5", "/models/motionstreamer/sentencet5-xxl"))
FPS = 30.0
MAX_FRAMES = 312
END_THRESHOLD = 0.1  # 公式のデモと同じ
NUM_JOINTS = 22

# SMPL-X の体 22 関節（EMAGE の SMPL-X と同じ名前）
JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow",
    "R_Elbow", "L_Wrist", "R_Wrist",
]  # fmt: skip
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19], dtype=np.int32)
FOOT_JOINTS = (7, 8, 10, 11)

# Causal TAE と生成モデルの設定（公式の options/option_transformer.py の既定値と demo_t2m.py）
TAE_ARGS = dict(hidden_size=1024, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, latent_dim=16)
CLIP_RANGE = [-30, 20]
NUM_DIFFUSION_HEAD_LAYERS = 9
BLOCK_SIZE = 78


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """6D 回転 [..., 6] → 回転行列 [..., 3, 3]（PyTorch3D と同じ定義。行が基底）。"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def recover(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """272 次元の表現（正規化を戻したもの）から、(局所回転 [T, 22, 3, 3], 腰の位置 [T, 3],
    向きを除いた関節位置 [T, 22, 3], 腰の向きを除いた局所回転 [T, 22, 3, 3]) を返す。公式の recover_from_local_rotation と同じ計算。"""
    t = x.shape[0]
    rot_no_heading = rot6d_to_matrix(x[:, 8 + 6 * NUM_JOINTS : 8 + 12 * NUM_JOINTS].reshape(t, NUM_JOINTS, 6))
    heading_diff = rot6d_to_matrix(x[:, 2:8])
    heading = np.empty_like(heading_diff)
    heading[0] = heading_diff[0]
    for i in range(1, t):  # 向きの変化を積み上げる
        heading[i] = heading_diff[i] @ heading[i - 1]
    inv_heading = np.transpose(heading, (0, 2, 1))
    rot = rot_no_heading.copy()
    rot[:, 0] = inv_heading @ rot_no_heading[:, 0]

    positions = x[:, 8 : 8 + 3 * NUM_JOINTS].reshape(t, NUM_JOINTS, 3)
    vel = np.zeros((t, 3))
    vel[:, 0], vel[:, 2] = x[:, 0], x[:, 1]
    vel[1:] = (inv_heading[:-1] @ vel[1:, :, None])[..., 0]
    root = np.cumsum(vel, axis=0)
    root[:, 1] = positions[:, 0, 1]
    return rot, root, positions, rot_no_heading


def estimate_offsets(positions: np.ndarray, rot_no_heading: np.ndarray) -> tuple[np.ndarray, float]:
    """向きを除いた関節位置と局所回転から、親からのオフセット [22, 3] を推定する（全フレームの中央値）。
    返り値の 2 つめは、推定した骨格で順運動学を計算したときの関節位置の最大誤差（m）。"""
    t = positions.shape[0]
    glob = np.empty_like(rot_no_heading)
    for j in range(NUM_JOINTS):
        p = PARENTS[j]
        glob[:, j] = rot_no_heading[:, j] if p < 0 else glob[:, p] @ rot_no_heading[:, j]
    offsets = np.zeros((NUM_JOINTS, 3))
    for j in range(1, NUM_JOINTS):
        p = PARENTS[j]
        local = np.einsum("tji,tj->ti", glob[:, p], positions[:, j] - positions[:, p])  # 親のレストの座標系へ
        offsets[j] = np.median(local, axis=0)
    fk = np.empty_like(positions)
    fk[:, 0] = positions[:, 0]
    for j in range(1, NUM_JOINTS):
        p = PARENTS[j]
        fk[:, j] = fk[:, p] + np.einsum("tij,j->ti", glob[:, p], offsets[j])
    error = float(np.linalg.norm(fk - positions, axis=-1).max()) if t else 0.0
    return offsets, error


class MotionStreamerAdapter(Adapter):
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        required = [
            CKPTS / "Causal_TAE" / "net_last.pth",
            CKPTS / "Experiments" / "t2m_model" / "latest.pth",
            CKPTS / "humanml3d_272" / "mean_std" / "Mean.npy",
            CKPTS / "humanml3d_272" / "mean_std" / "Std.npy",
            T5_DIR / "model.safetensors",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise AdapterError("missing_asset", "missing: " + ", ".join(missing))
        if self.device.type == "cpu":
            raise AdapterError("internal", "CUDA is not available")
        os.chdir(MS_DIR)
        sys.path.insert(0, str(MS_DIR))

        import models.tae as tae
        from models.llama_model import LLaMAHF, LLaMAHFConfig
        from sentence_transformers import SentenceTransformer

        self.t5 = SentenceTransformer(str(T5_DIR), device=str(self.device))
        self.t5.eval()
        net = tae.Causal_HumanTAE(**TAE_ARGS, activation="relu", clip_range=CLIP_RANGE)
        net.load_state_dict(torch.load(CKPTS / "Causal_TAE" / "net_last.pth", map_location="cpu")["net"], strict=True)
        self.net = net.eval().to(self.device)

        config = LLaMAHFConfig.from_name("Normal_size")
        config.block_size = BLOCK_SIZE
        trans = LLaMAHF(config, NUM_DIFFUSION_HEAD_LAYERS, TAE_ARGS["latent_dim"], self.device)
        state = torch.load(CKPTS / "Experiments" / "t2m_model" / "latest.pth", map_location="cpu")["trans"]
        trans.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        self.trans = trans.eval().to(self.device)

        self.end_latent = torch.from_numpy(np.load(MS_DIR / "reference_end_latent_t2m_272.npy")).to(self.device)
        self.mean = np.load(CKPTS / "humanml3d_272" / "mean_std" / "Mean.npy")
        self.std = np.load(CKPTS / "humanml3d_272" / "mean_std" / "Std.npy")

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        text = str(request.inputs.get("text", "")).strip()
        if not text:
            raise AdapterError("invalid_input", "inputs.text is required")
        ignored_duration = request.params.get("duration")  # 使わない（記録だけする）
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)

        progress(0.05)
        with torch.no_grad():
            latents = self.trans.sample_for_eval_CFG_inference(
                text=text, length=MAX_FRAMES, tokenizer=self.t5, device=self.device,
                reference_end_latent=self.end_latent, threshold=END_THRESHOLD,
            )  # fmt: skip
            motion = self.net.forward_decoder(latents)[0].float().cpu().numpy()
        progress(0.85)
        x = motion * self.std + self.mean  # [T, 272]

        raw_name = "motionstreamer_272.npz"
        np.savez(output_dir / raw_name, motion_272=x.astype(np.float32), latents=latents[0].float().cpu().numpy(),
                 fps=np.asarray(FPS), text=np.asarray(text))  # fmt: skip

        rot, root, positions, rot_no_heading = recover(x)
        offsets, fk_error = estimate_offsets(positions, rot_no_heading)
        rest = np.zeros_like(offsets)
        for j in range(1, NUM_JOINTS):
            rest[j] = rest[PARENTS[j]] + offsets[j]
        rest_height = rest[:, 1] - rest[:, 1].min()
        offsets[0] = [0.0, rest_height[0], 0.0]  # レスト時の腰の高さ（最も低い関節を地面とする）

        motion_out = SourceMotion(
            joint_names=list(JOINT_NAMES),
            parents=PARENTS,
            offsets=offsets.astype(np.float32),
            rest_rotations=np.tile(quat.IDENTITY, (NUM_JOINTS, 1)),
            local_rotations=quat.make_continuous(quat.normalize(quat.from_matrix(rot))).astype(np.float32),
            root_translation=root.astype(np.float32),
            fps=FPS,
            up_axis="+Y",
            forward_axis="+Z",
            rest_pose_kind="tpose",
        )
        # 正規化: 足の関節がレスト時と同じ高さ（最も低い関節が y=0）になるよう、各フレームで最も低い足の関節の
        # 中央値で上下をそろえる。最初のフレームの水平位置を原点にする。向きはモデルの出力のまま（+Z が正面）
        pos = motion_out.global_positions()
        feet = pos[:, FOOT_JOINTS, 1] - rest_height[list(FOOT_JOINTS)]
        root[:, 1] -= np.median(feet.min(axis=1))
        root[:, [0, 2]] -= root[0, [0, 2]]
        motion_out.root_translation = root.astype(np.float32)
        motion_out.meta.update(
            {
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "model": "MotionStreamer (text to motion, HumanML3D 272)",
                "prompt": text,
                "seed": seed,
                "duration_auto": True,
                "ignored_duration": ignored_duration,
                "fk_max_error_m": fk_error,  # 推定した骨格の順運動学と、モデルが出した関節位置との最大誤差
                "raw_outputs": [raw_name],
            }
        )
        motion_out.save(output_dir / "source_motion.npz")
        progress(1.0)
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=[raw_name],
            fps=FPS,
            num_frames=motion_out.num_frames,
            duration=motion_out.duration,
            seed=seed,
            extra={"fk_max_error_m": fk_error},
        )


if __name__ == "__main__":
    serve(MotionStreamerAdapter(), port=int(os.environ.get("PORT", "8000")))
