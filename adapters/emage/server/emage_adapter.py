"""EMAGE アダプター（音声 → SMPL-X＋FLAME の全身の身振り）。

公式の推論（PantoMatrix の test_emage_audio.py）と同じ手順で、音声から体・指・顎の回転（SMPL-X の 55 関節）、
FLAME の表情係数（100 次元）、ルートの移動量を 30fps で作り、SourceMotion にする。
- 音声は ffmpeg で 16kHz・モノラルの WAV にしてから渡す（公式は librosa で同じ変換をする）
- 骨格は SMPL-X（中立、体型係数 0）のレスト時の関節位置。EMAGE も体型係数 0 で出力する
- 口の動きは、表情係数と顎の回転から唇の開き・幅を測り、VRM の表情（aa / ih / ou / ee / oh）の重みにする
- 推論は確定的（seed の影響はない）。話者は 1 人（speaker_id=0）に固定される
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sourcemotion import SourceMotion, quat
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

ADAPTER_ID = "emage"
ADAPTER_VERSION = "0.1.0"
PANTOMATRIX_DIR = Path(os.environ.get("PANTOMATRIX_DIR", "/opt/pantomatrix"))
SMPLX_PATH = Path(os.environ.get("EMAGE_SMPLX", "/models/emage/evaltools/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"))
HF_REPO = "H-Liu1997/emage_audio"
SAMPLE_RATE = 16000
FPS = 30.0
# 推論は 64 フレームの窓で進むため、それより短い音声は扱えない
MIN_SECONDS = 64 / FPS + 0.1
MAX_SECONDS = 60.0

JAW = 22
FOOT_JOINTS = (7, 8, 10, 11)  # 足首とつま先の付け根
# FLAME の 51 点の顔ランドマーク（68 点の 18 番以降）のうち、口角と唇の中央
LIP_LANDMARKS = {"corner_l": 31, "corner_r": 37, "upper_inner": 45, "lower_inner": 49}
VISEMES = ["aa", "ih", "ou", "ee", "oh"]


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """軸角度 [..., 3] → クォータニオン [..., 4]（xyzw）。"""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, aa / np.maximum(angle, 1e-12), np.array([1.0, 0.0, 0.0]))
    return quat.from_axis_angle(axis, angle[..., 0])


class SmplxModel:
    """SMPL-X の骨格（関節名・親・レスト時の関節位置）と、唇のランドマークの計算に要る部分だけを持つ。"""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=True) as z:
            joint2num = z["joint2num"].item()
            self.names = [str(n) for n, _ in sorted(joint2num.items(), key=lambda kv: kv[1])]
            parents = z["kintree_table"][0].astype(np.int64)
            parents[0] = -1
            self.parents = parents.astype(np.int32)
            v_template = z["v_template"].astype(np.float64)
            self.rest = z["J_regressor"].astype(np.float64) @ v_template  # [55, 3]
            faces = z["f"][z["lmk_faces_idx"]].astype(np.int64)  # [51, 3]
            bary = z["lmk_bary_coords"].astype(np.float64)
            # ランドマークは 3 頂点の重心座標。必要な頂点の分だけ表情のブレンドシェイプと顎のスキンウェイトを取り出す
            self.lips: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for key, i in LIP_LANDMARKS.items():
                verts = faces[i]
                self.lips[key] = (
                    v_template[verts],
                    z["shapedirs"][verts, :, 300:400].astype(np.float64),  # 体型 300 次元の後ろが表情 100 次元
                    z["weights"][verts, JAW].astype(np.float64),
                    bary[i],
                )
        if len(self.names) != 55 or any(not 0 <= p < i for i, p in enumerate(self.parents[1:], 1)):
            raise AdapterError("internal", "unexpected SMPL-X kintree")

    def offsets(self) -> np.ndarray:
        """親からのオフセット。offsets[0] はレスト時のルート位置（最も低い関節を地面とする）。"""
        offsets = np.zeros_like(self.rest)
        offsets[1:] = self.rest[1:] - self.rest[self.parents[1:]]
        offsets[0] = [0.0, self.rest[0, 1] - self.rest[:, 1].min(), 0.0]
        return offsets

    def lip_points(self, expressions: np.ndarray, jaw: np.ndarray) -> dict[str, np.ndarray]:
        """唇のランドマークの位置 [T, 3]（頭の座標系）。表情のブレンドシェイプと顎の回転（線形ブレンドスキニング）を当てる。"""
        jaw_rot = quat.to_matrix(axis_angle_to_quat(jaw))  # [T, 3, 3]
        pivot = self.rest[JAW]
        out = {}
        for key, (v, dirs, w_jaw, bary) in self.lips.items():
            shaped = v[None] + np.einsum("vcd,td->tvc", dirs, expressions)  # [T, 3, 3]
            rotated = np.einsum("tij,tvj->tvi", jaw_rot, shaped - pivot) + pivot
            posed = shaped * (1 - w_jaw)[None, :, None] + rotated * w_jaw[None, :, None]
            out[key] = (posed * bary[None, :, None]).sum(axis=1)
        return out


def mouth_features(smplx: SmplxModel, expressions: np.ndarray, jaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """唇の内側の開き（m、表情 0・顎 0 の状態との差）と、口の幅の相対変化（同じく 0 の状態との比 - 1）。"""
    pts = smplx.lip_points(expressions, jaw)
    neutral = smplx.lip_points(np.zeros((1, expressions.shape[1])), np.zeros((1, 3)))

    def measure(p):
        return p["upper_inner"][:, 1] - p["lower_inner"][:, 1], np.linalg.norm(p["corner_l"] - p["corner_r"], axis=-1)

    gap, width = measure(pts)
    gap0, width0 = measure(neutral)
    return gap - gap0, width / width0 - 1.0


def visemes_from_mouth(gap: np.ndarray, width: np.ndarray) -> np.ndarray:
    """唇の開きと幅から VRM の口の表情（VISEMES の順）の重み [T, 5] を作る。

    - open: 開きが 1.5mm を超えた分を 10mm で 1 にする（話していないときの小さな開きは閉じとみなす）
    - 幅は、この音声全体の中央値からの増減で見る（すぼめ: -5% で 1、横に引く: +6% で 1）
    - 大きく開く → aa（すぼめていれば oh、横に引いていれば ee）、少し開く → すぼめていれば ou、横に引いていれば ih
    重みの合計は 1 を超えない。
    """
    open_ = np.clip((gap - 0.0015) / 0.010, 0.0, 1.0)
    speaking = np.clip((gap - 0.001) / 0.003, 0.0, 1.0)
    dev = width - np.median(width)
    narrow = np.clip(-dev / 0.05, 0.0, 1.0)
    spread = np.clip(dev / 0.06, 0.0, 1.0)
    aa = open_ * (1 - narrow) * (1 - spread)
    oh = open_ * narrow
    ee = open_ * spread
    ih = speaking * (1 - open_) * spread
    ou = speaking * (1 - open_) * narrow
    weights = np.stack([aa, ih, ou, ee, oh], axis=1)
    return weights.astype(np.float32)


def read_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise AdapterError("internal", f"unexpected wav format: {path}")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


class EmageAdapter(Adapter):
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        if not SMPLX_PATH.exists():
            raise AdapterError("missing_asset", f"{SMPLX_PATH} がない")
        if self.device.type == "cpu" and os.environ.get("ALLOW_CPU") != "1":
            raise AdapterError("internal", "CUDA is not available")
        sys.path.insert(0, str(PANTOMATRIX_DIR))
        from huggingface_hub.errors import LocalEntryNotFoundError
        from models.emage_audio import EmageAudioModel, EmageVAEConv, EmageVQModel, EmageVQVAEConv

        try:
            vq = EmageVQModel(
                face_model=EmageVQVAEConv.from_pretrained(HF_REPO, subfolder="emage_vq/face"),
                upper_model=EmageVQVAEConv.from_pretrained(HF_REPO, subfolder="emage_vq/upper"),
                lower_model=EmageVQVAEConv.from_pretrained(HF_REPO, subfolder="emage_vq/lower"),
                hands_model=EmageVQVAEConv.from_pretrained(HF_REPO, subfolder="emage_vq/hands"),
                global_model=EmageVAEConv.from_pretrained(HF_REPO, subfolder="emage_vq/global"),
            )
            model = EmageAudioModel.from_pretrained(HF_REPO)
        except (LocalEntryNotFoundError, OSError) as e:
            raise AdapterError("missing_asset", f"{HF_REPO} の重みが Hugging Face のキャッシュにない: {e}") from e
        self.vq = vq.to(self.device).eval()
        self.model = model.to(self.device).eval()
        if int(self.model.cfg.audio_sr) != SAMPLE_RATE or float(self.model.cfg.pose_fps) != FPS:
            raise AdapterError("internal", f"unexpected config: sr={self.model.cfg.audio_sr}, fps={self.model.cfg.pose_fps}")
        self.smplx = SmplxModel(SMPLX_PATH)

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        audio_in = request.inputs.get("audio")
        if not audio_in or not Path(audio_in).exists():
            raise AdapterError("invalid_input", f"audio not found: {audio_in}")
        audio_in = Path(audio_in)

        # 元のファイルと、モデルに渡した 16kHz・モノラルの WAV を両方残す
        original = f"input_audio{audio_in.suffix.lower()}"
        shutil.copy2(audio_in, output_dir / original)
        wav_name = "input_16k.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_in), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-c:a", "pcm_s16le", str(output_dir / wav_name)]  # fmt: skip
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise AdapterError("invalid_input", f"音声を変換できない: {r.stderr.strip()[-500:]}")
        audio = read_wav_16k(output_dir / wav_name)
        seconds = len(audio) / SAMPLE_RATE
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise AdapterError("invalid_input", f"音声は {MIN_SECONDS:.1f}〜{MAX_SECONDS:.0f} 秒にする（{seconds:.1f} 秒）")

        progress(0.1)
        cfg = self.model.cfg
        with torch.no_grad():
            wav = torch.from_numpy(audio).to(self.device)[None]
            speaker = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            latent = self.model.inference(wav, speaker, self.vq, masked_motion=None, mask=None)

            def latent_or_index(part: str, layers: int, classes: int):
                rec = latent[f"rec_{part}"] if layers > 0 and classes == 0 else None
                index = torch.max(F.log_softmax(latent[f"cls_{part}"], dim=2), dim=2)[1] if classes > 0 else None
                return rec, index

            face_latent, face_index = latent_or_index("face", cfg.lf, cfg.cf)
            upper_latent, upper_index = latent_or_index("upper", cfg.lu, cfg.cu)
            hands_latent, hands_index = latent_or_index("hands", cfg.lh, cfg.ch)
            lower_latent, lower_index = latent_or_index("lower", cfg.ll, cfg.cl)
            pred = self.vq.decode(
                face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
                face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,
                get_global_motion=True, ref_trans=torch.zeros(1, 3, device=self.device),
            )  # fmt: skip
        progress(0.8)

        poses = pred["motion_axis_angle"][0].cpu().numpy().astype(np.float64)  # [T, 165]
        expressions = pred["expression"][0].cpu().numpy().astype(np.float64)  # [T, 100]
        trans = pred["trans"][0].cpu().numpy().astype(np.float64)  # [T, 3]
        t = poses.shape[0]
        raw_name = "emage_output.npz"
        # 公式と同じ BEAT2 の形式（SMPL-X 2020 の中立モデル、体型係数 0）
        np.savez(
            output_dir / raw_name,
            betas=np.zeros(300, dtype=np.float32),
            poses=poses.astype(np.float32),
            expressions=expressions.astype(np.float32),
            trans=trans.astype(np.float32),
            model="smplx2020",
            gender="neutral",
            mocap_frame_rate=30,
        )

        motion = self.to_source_motion(poses, trans)
        gap, width = mouth_features(self.smplx, expressions, poses[:, JAW * 3 : JAW * 3 + 3])
        motion.expression_names = list(VISEMES)
        motion.expression_weights = visemes_from_mouth(gap, width)
        raw_outputs = [raw_name, original, wav_name]
        motion.meta.update(
            {
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "model": "EMAGE (emage_audio)",
                "audio": audio_in.name,
                "audio_seconds": seconds,
                "speaker_id": 0,
                "mouth": {
                    "gap_mm_p50_p99": [float(np.percentile(gap, 50) * 1000), float(np.percentile(gap, 99) * 1000)],
                    "width_rel_p1_p99": [float(np.percentile(width, 1)), float(np.percentile(width, 99))],
                },
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
            num_frames=t,
            duration=motion.duration,
            seed=None,
        )

    def to_source_motion(self, poses: np.ndarray, trans: np.ndarray) -> SourceMotion:
        sm = self.smplx
        t = poses.shape[0]
        local = axis_angle_to_quat(poses.reshape(t, 55, 3))
        # SMPL-X の骨盤のワールド位置は「移動量 + レスト時の骨盤の位置」（体型係数 0）
        motion = SourceMotion(
            joint_names=sm.names,
            parents=sm.parents,
            offsets=sm.offsets().astype(np.float32),
            rest_rotations=np.tile(quat.IDENTITY, (55, 1)),
            local_rotations=quat.make_continuous(quat.normalize(local)).astype(np.float32),
            root_translation=(trans + sm.rest[0]).astype(np.float32),
            fps=FPS,
            up_axis="+Y",
            forward_axis="+Z",
            rest_pose_kind="tpose",
        )
        # 正規化: 足の関節の高さを、レスト時の最も低い関節（y=0）にそろえ、最初のフレームの水平位置を原点にする。
        # 高さの移動量は上下に数 cm 揺れるため、全体の最低点ではなく「各フレームで最も低い足の関節」の中央値で合わせる
        # （最低点で合わせると、ほとんどのフレームで 5cm ほど浮く）。
        # 向きは BEAT2 の座標のまま（+Z が正面）。水平の移動量はモデルの出力のまま使う（話しながら足を運ぶ動きも含む）
        pos = motion.global_positions()
        root = motion.root_translation.astype(np.float64)
        root[:, 1] -= np.median(pos[:, FOOT_JOINTS, 1].min(axis=1))
        root[:, [0, 2]] -= root[0, [0, 2]]
        motion.root_translation = root.astype(np.float32)
        return motion


if __name__ == "__main__":
    serve(EmageAdapter(), port=int(os.environ.get("PORT", "8000")))
