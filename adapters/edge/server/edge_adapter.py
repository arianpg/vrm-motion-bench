"""EDGE アダプター（音楽 → SMPL 24 関節のダンス）。

公式の推論（test.py と GaussianDiffusion.render_sample の long モード）と同じ手順で作る。
- 音楽を 5 秒の窓・2.5 秒ずらしで切り、窓ごとに Jukebox（5B の top prior の 66 層目）の特徴を 30fps で取る
- 窓を並べて拡散モデルでまとめて生成し（隣り合う窓の重なる半分をそろえながらサンプリングする）、
  重なりを線形補間（位置）と slerp（回転）でつなぐ
- 公式はランダムな区間（最大 30 秒）を選ぶが、ここでは音楽の先頭から全体を使う。最後の 5 秒に満たない端は使わない
- 骨格は EDGE 同梱の SMPL（中立）のオフセット。出力は Z 上向きなので、学習データ（AIST++）の Y 上向きに戻す
- 足の接地フラグ（足首・つま先の 4 関節）はモデルの出力をそのまま使う
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

from sourcemotion import SourceMotion, quat
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

ADAPTER_ID = "edge"
ADAPTER_VERSION = "0.1.0"
EDGE_DIR = Path(os.environ.get("EDGE_DIR", "/opt/edge"))
CHECKPOINT = Path(os.environ.get("EDGE_CHECKPOINT", "/models/edge/checkpoint.pt"))
JUKEBOX_CACHE = Path(os.environ.get("JUKEBOX_CACHE", "/models/edge/jukemirlib"))
SAMPLE_RATE = 44100  # Jukebox の入力
FPS = 30.0
WINDOW_S, STRIDE_S = 5.0, 2.5
JUKEBOX_LAYER = 66
MAX_SECONDS = 60.0

CONTACT_JOINTS = ("lankle", "rankle", "ltoes", "rtoes")  # 出力の接地フラグの順（関節 7, 8, 10, 11）
CONTACT_INDICES = (7, 8, 10, 11)
# 出力は「AIST++ の Y 上向きの座標を X 軸まわりに +90 度回した」Z 上向きの座標。その逆回転
Z_UP_TO_Y_UP = quat.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.array(-np.pi / 2))


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """軸角度 [..., 3] → クォータニオン [..., 4]（xyzw）。"""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, aa / np.maximum(angle, 1e-12), np.array([1.0, 0.0, 0.0]))
    return quat.from_axis_angle(axis, angle[..., 0])


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise AdapterError("internal", f"unexpected wav format: {path}")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def slice_windows(audio: np.ndarray) -> list[np.ndarray]:
    """公式の slice_audio と同じく、5 秒の窓を 2.5 秒ずつずらして切る（窓に満たない端は捨てる）。"""
    window, stride = int(WINDOW_S * SAMPLE_RATE), int(STRIDE_S * SAMPLE_RATE)
    return [audio[s : s + window] for s in range(0, len(audio) - window + 1, stride)]


def stitch(values: torch.Tensor, half: int) -> torch.Tensor:
    """重なりのある窓 [B, S, ...] を、前の窓をフェードアウト・次の窓をフェードインさせて足し合わせる（公式の位置のつなぎ方）。"""
    b, s = values.shape[:2]
    if b == 1:
        return values[0]
    shape = (1, s) + (1,) * (values.dim() - 2)
    fade_out = torch.ones(shape, device=values.device)
    fade_in = torch.ones(shape, device=values.device)
    fade_out[:, half:] = torch.linspace(1, 0, half, device=values.device).reshape((1, half) + shape[2:])
    fade_in[:, :half] = torch.linspace(0, 1, half, device=values.device).reshape((1, half) + shape[2:])
    values = values.clone()
    values[:-1] *= fade_out
    values[1:] *= fade_in
    out = torch.zeros((s + half * (b - 1),) + values.shape[2:], device=values.device)
    for k in range(b):
        out[k * half : k * half + s] += values[k]
    return out


class EdgeAdapter(Adapter):
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        required = [CHECKPOINT, JUKEBOX_CACHE / "vqvae.pth.tar", JUKEBOX_CACHE / "prior_level_2.pth.tar"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise AdapterError("missing_asset", "missing: " + ", ".join(missing))
        if self.device.type == "cpu":
            raise AdapterError("internal", "CUDA is not available")
        os.chdir(EDGE_DIR)
        sys.path.insert(0, str(EDGE_DIR))

        import jukemirlib
        from EDGE import EDGE

        # Jukebox（約 10GB）を先に GPU に載せる。jukemirlib.extract は lib モジュールの変数を見るので、そこに入れる
        jukemirlib.lib.VQVAE, jukemirlib.lib.TOP_PRIOR = jukemirlib.setup_models(
            cache_dir=str(JUKEBOX_CACHE), device="cuda", verbose=False
        )
        self.jukemirlib = jukemirlib
        model = EDGE("jukebox", str(CHECKPOINT))
        model.eval()
        self.model = model

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        from dataset.quaternion import ax_from_6v, quat_slerp
        from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle

        audio_in = request.inputs.get("audio")
        if not audio_in or not Path(audio_in).exists():
            raise AdapterError("invalid_input", f"audio not found: {audio_in}")
        audio_in = Path(audio_in)
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)

        # 元のファイルと、Jukebox に渡す 44.1kHz・モノラルの WAV を両方残す
        original = f"input_audio{audio_in.suffix.lower()}"
        shutil.copy2(audio_in, output_dir / original)
        wav_name = "input_44k.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_in), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-c:a", "pcm_s16le", str(output_dir / wav_name)]  # fmt: skip
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise AdapterError("invalid_input", f"音声を変換できない: {r.stderr.strip()[-500:]}")
        audio = read_wav(output_dir / wav_name)
        seconds = len(audio) / SAMPLE_RATE
        if not WINDOW_S <= seconds <= MAX_SECONDS + 0.5:
            raise AdapterError("invalid_input", f"音楽は {WINDOW_S:.0f}〜{MAX_SECONDS:.0f} 秒にする（{seconds:.1f} 秒）")
        windows = slice_windows(audio)

        # Jukebox の特徴（窓ごとに 1 件ずつ。公式の load_audio と同じく窓ごとに最大振幅で正規化する）
        started = time.monotonic()
        feats = []
        for k, w in enumerate(windows):
            peak = float(np.abs(w).max())
            w = w / peak if peak > 0 else w
            reps = self.jukemirlib.extract(audio=w, layers=[JUKEBOX_LAYER], downsample_target_rate=int(FPS))
            feats.append(reps[JUKEBOX_LAYER])
            progress(0.7 * (k + 1) / len(windows))
        jukebox_seconds = time.monotonic() - started
        cond = torch.from_numpy(np.stack(feats)).to(self.device)  # [B, 150, 4800]

        torch.manual_seed(seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)
        diffusion = self.model.diffusion
        shape = (len(windows), self.model.horizon, self.model.repr_dim)
        with torch.no_grad():
            samples = diffusion.long_ddim_sample(shape, cond)  # 窓が 1 つなら通常の DDIM になる
            samples = self.model.normalizer.unnormalize(samples.detach().cpu()).to(self.device)
        progress(0.9)

        contact, pos, rot6d = samples[..., :4], samples[..., 4:7], samples[..., 7:]
        b, s = samples.shape[:2]
        half = s // 2
        q = ax_from_6v(rot6d.reshape(b, s, 24, 6))  # 軸角度 [B, S, 24, 3]
        full_pos = stitch(pos, half)
        full_contact = stitch(contact, half)
        if b > 1:
            # 回転は重なりを slerp でつなぐ（公式と同じ）
            w = torch.linspace(0, 1, half, device=self.device)[None, :, None]
            merged = quaternion_to_axis_angle(
                quat_slerp(axis_angle_to_quaternion(q[:-1, half:]), axis_angle_to_quaternion(q[1:, :half]), w)
            )
            full_q = torch.cat([q[0, :half], *merged, q[-1, half:]], dim=0)
        else:
            full_q = q[0]

        poses = full_q.cpu().numpy().astype(np.float64)  # [T, 24, 3]
        trans = full_pos.cpu().numpy().astype(np.float64)  # [T, 3]
        contacts = full_contact.cpu().numpy() > 0.5  # [T, 4]
        raw_name = "edge_output.npz"
        np.savez(
            output_dir / raw_name,
            smpl_poses=poses.reshape(-1, 72).astype(np.float32),
            smpl_trans=trans.astype(np.float32),
            contacts=full_contact.cpu().numpy().astype(np.float32),
            fps=np.asarray(FPS),
        )

        motion = self.to_source_motion(poses, trans, contacts)
        raw_outputs = [raw_name, original, wav_name]
        motion.meta.update(
            {
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "model": "EDGE (jukebox)",
                "audio": audio_in.name,
                "audio_seconds": seconds,
                "windows": len(windows),
                "seed": seed,
                "jukebox_seconds": jukebox_seconds,
                "raw_outputs": raw_outputs,
            }
        )
        motion.save(output_dir / "source_motion.npz")
        progress(1.0)
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=raw_outputs,
            outputs={"audio": original},
            fps=FPS,
            num_frames=motion.num_frames,
            duration=motion.duration,
            seed=seed,
            extra={"jukebox_seconds": jukebox_seconds},
        )

    def to_source_motion(self, poses: np.ndarray, trans: np.ndarray, contacts: np.ndarray) -> SourceMotion:
        from vis import smpl_joints, smpl_offsets, smpl_parents

        local = axis_angle_to_quat(poses)
        # Z 上向き → Y 上向き（ルートの回転と位置に逆回転を掛ける）
        local[:, 0] = quat.multiply(Z_UP_TO_Y_UP[None], local[:, 0])
        root = quat.rotate(Z_UP_TO_Y_UP[None], trans)

        offsets = np.asarray(smpl_offsets, dtype=np.float64)
        parents = np.asarray(smpl_parents, dtype=np.int32)
        rest = np.zeros_like(offsets)
        for j in range(1, len(offsets)):
            rest[j] = rest[parents[j]] + offsets[j]
        rest_height = rest[:, 1] - rest[:, 1].min()  # 最も低い関節を地面としたレスト時の高さ
        offsets = offsets.copy()
        offsets[0] = [0.0, rest_height[0], 0.0]

        motion = SourceMotion(
            joint_names=list(smpl_joints),
            parents=parents,
            offsets=offsets.astype(np.float32),
            rest_rotations=np.tile(quat.IDENTITY, (len(smpl_joints), 1)),
            local_rotations=quat.make_continuous(quat.normalize(local)).astype(np.float32),
            root_translation=root.astype(np.float32),
            fps=FPS,
            up_axis="+Y",
            forward_axis="+Z",
            rest_pose_kind="tpose",
            foot_contacts=contacts.astype(bool),
            contact_joint_names=list(CONTACT_JOINTS),
        )
        # 正規化: 接地しているフレームの足の関節が、レスト時と同じ高さ（最も低い関節が y=0）になるよう上下をそろえる
        # （関節ごとのレスト時の高さとの差の中央値）。接地フレームがなければ、各フレームで最も低い足の関節の中央値を使う。
        # 最初のフレームの水平位置を原点にする
        pos = motion.global_positions()
        feet = pos[:, CONTACT_INDICES, 1] - rest_height[list(CONTACT_INDICES)]
        ground = float(np.median(feet[contacts])) if contacts.any() else float(np.median(feet.min(axis=1)))
        root[:, 1] -= ground
        root[:, [0, 2]] -= root[0, [0, 2]]
        motion.root_translation = root.astype(np.float32)
        motion.meta["ground_from_contacts"] = bool(contacts.any())
        return motion


if __name__ == "__main__":
    serve(EdgeAdapter(), port=int(os.environ.get("PORT", "8000")))
