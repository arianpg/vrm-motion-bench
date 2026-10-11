"""MoMask アダプター（CPU）。

MoMask は HumanML3D の 263 次元特徴量を生成する。関節位置（22 関節）に戻したあと、
付属の Joint2BVHConvertor（IK）で BVH にし、sourcemotion.read_bvh で SourceMotion にする。
接地フラグは特徴量の末尾 4 次元（左足首・左つま先・右足首・右つま先）を使う。
"""

from __future__ import annotations

import functools
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from sourcemotion import read_bvh
from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve

ADAPTER_ID = "momask"
ADAPTER_VERSION = "0.1.0"
MOMASK_DIR = Path(os.environ.get("MOMASK_DIR", "/opt/momask"))
CHECKPOINTS = Path(os.environ.get("MOMASK_CHECKPOINTS", "/models/momask/checkpoints"))
CLIP_DIR = Path(os.environ.get("MOMASK_CLIP_DIR", "/models/momask/clip"))
FOOT_IK = os.environ.get("MOMASK_FOOT_IK", "0") == "1"

FPS = 20.0
UNIT = 4  # VQ の時間方向の縮小率（4 フレーム = 1 トークン）
MIN_FRAMES, MAX_FRAMES = 40, 196
MODEL_NAME = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
RES_NAME = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"
# 特徴量の接地フラグの順（HumanML3D: fid_l = 足首・つま先、fid_r = 同じ）→ BVH の関節名
CONTACT_JOINTS = ["LeftFoot", "LeftToe", "RightFoot", "RightToe"]


class MoMaskAdapter(Adapter):
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def load(self) -> None:
        if not (CHECKPOINTS / "t2m" / MODEL_NAME).exists():
            raise AdapterError("missing_asset", f"{CHECKPOINTS / 't2m'} にチェックポイントがない")
        # 付属コードは作業ディレクトリからの相対パス（BVH テンプレートなど）を使う
        os.chdir(MOMASK_DIR)
        sys.path.insert(0, str(MOMASK_DIR))

        import clip

        # CLIP の重みを永続化したディレクトリに置く（既定は ~/.cache/clip）
        CLIP_DIR.mkdir(parents=True, exist_ok=True)
        clip.load = functools.partial(clip.load, download_root=str(CLIP_DIR))

        from gen_t2m import load_len_estimator, load_res_model, load_trans_model, load_vq_model
        from utils.get_opt import get_opt
        from visualization.joints2bvh import Joint2BVHConvertor

        def opt_for(name: str):
            opt = get_opt(str(CHECKPOINTS / "t2m" / name / "opt.txt"), device=self.device)
            opt.checkpoints_dir = str(CHECKPOINTS)
            return opt

        main = argparse_namespace(device=self.device, name=MODEL_NAME)  # 公式の関数がログに name を使う
        model_opt = opt_for(MODEL_NAME)
        vq_opt = opt_for(model_opt.vq_name)
        vq_opt.dim_pose = 263
        self.vq, vq_opt = load_vq_model(vq_opt)
        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim
        res_opt = opt_for(RES_NAME)
        self.res = load_res_model(res_opt, vq_opt, main)
        self.t2m = load_trans_model(model_opt, main, "latest.tar")
        self.length_estimator = load_len_estimator(model_opt)
        for m in (self.vq, self.res, self.t2m, self.length_estimator):
            m.eval().to(self.device)
        meta = CHECKPOINTS / "t2m" / model_opt.vq_name / "meta"
        self.mean = np.load(meta / "mean.npy")
        self.std = np.load(meta / "std.npy")
        self.converter = Joint2BVHConvertor()

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        import torch.nn.functional as F
        from torch.distributions.categorical import Categorical
        from utils.fixseed import fixseed
        from utils.motion_process import recover_from_ric

        text = str(request.inputs.get("text", "")).strip()
        if not text:
            raise AdapterError("invalid_input", "inputs.text is required")
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        fixseed(seed)

        duration = request.params.get("duration")
        with torch.no_grad():
            if duration is None:
                # 尺の指定がなければモデルの長さ推定を使う（公式の #NA と同じ）
                probs = F.softmax(self.length_estimator(self.t2m.encode_text([text])), dim=-1)
                token_lens = Categorical(probs).sample().clamp(MIN_FRAMES // UNIT, MAX_FRAMES // UNIT)
            else:
                frames = int(round(float(duration) * FPS / UNIT)) * UNIT
                if not MIN_FRAMES <= frames <= MAX_FRAMES:
                    raise AdapterError("invalid_input", f"duration must be within {MIN_FRAMES / FPS}-{MAX_FRAMES / FPS} s")
                token_lens = torch.LongTensor([frames // UNIT])
            m_length = int(token_lens[0]) * UNIT
            progress(0.1)
            # 公式の gen_t2m.py の既定値（time_steps=18, cond_scale=4, temperature=1, topkr=0.9）
            mids = self.t2m.generate([text], token_lens, timesteps=18, cond_scale=4, temperature=1,
                                     topk_filter_thres=0.9, gsample=False)
            mids = self.res.generate(mids, [text], token_lens, temperature=1, cond_scale=5)
            features = self.vq.forward_decoder(mids).detach().cpu().numpy()[0] * self.std + self.mean
        features = features[:m_length]
        progress(0.6)

        joints = recover_from_ric(torch.from_numpy(features).float(), 22).numpy()
        bvh_path = output_dir / "momask.bvh"
        self.converter.convert(joints, filename=str(bvh_path), iterations=100, foot_ik=FOOT_IK)
        np.save(output_dir / "momask_joints.npy", joints)
        np.save(output_dir / "momask_features.npy", features)
        progress(0.9)

        raw = ["momask.bvh", "momask_joints.npy", "momask_features.npy"]
        meta = {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "model": MODEL_NAME,
            "prompt": text,
            "seed": seed,
            "duration": m_length / FPS,
            "length_estimated": duration is None,
            "foot_ik": FOOT_IK,
            "raw_outputs": raw,
        }
        motion = read_bvh(bvh_path, unit_scale=1.0, up_axis="+Y", forward_axis="+Z", rest_pose_kind="tpose", meta=meta)
        # offsets[0] はレスト時のルート位置（足裏が地面に接する高さ）にそろえる
        rest = motion.rest_positions()
        motion.offsets[0] = [0.0, motion.offsets[0][1] - rest[:, 1].min(), 0.0]
        contacts = features[:, -4:] > 0.5
        if contacts.shape[0] == motion.num_frames:
            motion.foot_contacts = contacts
            motion.contact_joint_names = CONTACT_JOINTS
        motion.save(output_dir / "source_motion.npz")
        progress(1.0)
        return GenerateResult(
            source_motion="source_motion.npz",
            raw_outputs=raw,
            fps=FPS,
            num_frames=motion.num_frames,
            duration=motion.duration,
            seed=seed,
            extra={"length_estimated": duration is None},
        )


def argparse_namespace(**kwargs):
    from argparse import Namespace

    return Namespace(**kwargs)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", str(os.cpu_count() or 4))))
    serve(MoMaskAdapter(), port=int(os.environ.get("PORT", "8000")))
