"""HunyuanVideo 1.5 アダプター（動画生成。kind: video）。

開始画像とテキストから動画を作る（I2V）。生成した動画は GEM-X などの動画入力のアダプターに渡す
（docs/research/gemx-text-to-video.md）。モデルは起動時に 1 回だけ読み込み、常駐させる。
内蔵のプロンプト書き直し（vLLM）と超解像は使わない。
"""

from __future__ import annotations

import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from vmb_adapter import Adapter, AdapterError, GenerateRequest, GenerateResult, ProgressCallback, serve  # noqa: E402

ADAPTER_ID = "hyvideo"
ADAPTER_VERSION = "0.1.0"
HYVIDEO_DIR = Path(os.environ.get("HYVIDEO_DIR", "/opt/hyvideo"))
CKPTS = Path(os.environ.get("HYVIDEO_CKPTS", "/models/hyvideo/ckpts"))
RESOLUTION = os.environ.get("HYVIDEO_RESOLUTION", "480p")
STEP_DISTILLED = os.environ.get("HYVIDEO_STEP_DISTILLED", "1") == "1"
DEFAULT_STEPS = int(os.environ.get("HYVIDEO_STEPS", "12" if STEP_DISTILLED else "50"))
FPS = 24.0
MIN_FRAMES, MAX_FRAMES = 33, 241  # 4k+1 フレーム。既定は 121（約 5 秒）


VAE_TILE_SIZE, VAE_TILE_OVERLAP = 128, 0.25


def frames_for(duration: float) -> int:
    """尺（秒）を、モデルが受け付ける 4k+1 フレームに丸める。"""
    n = int(round((duration * FPS - 1) / 4)) * 4 + 1  # 最も近い 4k+1（5 秒 → 121 フレーム）
    return max(MIN_FRAMES, min(MAX_FRAMES, n))


def infer_args() -> Namespace:
    """generate.py の既定値で InferState を作るための引数。"""
    return Namespace(
        use_sageattn=False,
        sage_blocks_range="0-53",
        enable_torch_compile=False,
        enable_cache=False,
        cache_type="deepcache",
        no_cache_block_id="53",
        cache_start_step=11,
        cache_end_step=45,
        total_steps=50,
        cache_step_interval=4,
        use_fp8_gemm=False,
        quant_type="fp8-per-token-sgl",
        include_patterns="double_blocks",
    )


class HyVideoAdapter(Adapter):
    def __init__(self) -> None:
        self.pipe = None

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise AdapterError("internal", "CUDA is not available")
        required = [
            CKPTS / "vae",
            CKPTS / "text_encoder" / "llm",
            CKPTS / "text_encoder" / "byt5-small",
            CKPTS / "text_encoder" / "Glyph-SDXL-v2" / "checkpoints" / "byt5_model.pt",
            CKPTS / "vision_encoder" / "siglip",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise AdapterError("missing_asset", "missing: " + ", ".join(missing))

        sys.path.insert(0, str(HYVIDEO_DIR))
        from hyvideo.commons.infer_state import initialize_infer_state
        from hyvideo.commons.parallel_states import initialize_parallel_state
        from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline

        initialize_parallel_state(sp=1)
        torch.cuda.set_device(0)
        infer_state = initialize_infer_state(infer_args())

        self.version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
            RESOLUTION, "i2v", cfg_distilled=False, step_distilled=STEP_DISTILLED, sparse_attn=False
        )
        if not (CKPTS / "transformer" / self.version).exists():
            # 見つからないと公式のコードは Hugging Face から取りに行くため、先に止める
            raise AdapterError("missing_asset", f"{CKPTS / 'transformer' / self.version} がない")

        # 32GB の GPU では公式の既定どおり、CPU への退避（グループ単位）を使う
        offload = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        group_offloading = offload["enable_group_offloading"]
        self.pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=str(CKPTS),
            transformer_version=self.version,
            create_sr_pipeline=False,
            transformer_dtype=torch.bfloat16,
            device=torch.device("cpu"),
            transformer_init_device=torch.device("cpu"),
        )
        self.pipe.apply_infer_optimization(
            infer_state=infer_state,
            enable_offloading=True,
            enable_group_offloading=group_offloading,
            overlap_group_offloading=True,
        )
        # VAE のデコードがメモリのピーク（121 フレームを一度にデコードし、約 26GB 使う）。
        # - 空間のタイルを公式の省メモリ設定（GPU が 21GB 以下のときに選ばれる 128px・重なり 0.25）にする。
        #   32GB の GPU では 256px が選ばれるが、デスクトップなどの使用量しだいで足りなくなるため
        # - 生成中に PyTorch が確保して使い終わったキャッシュを、デコードの直前に手放す
        self.pipe.vae.set_tile_sample_min_size(VAE_TILE_SIZE, VAE_TILE_OVERLAP)
        decode = self.pipe.vae.decode

        def decode_after_empty_cache(*args, **kwargs):
            torch.cuda.empty_cache()
            return decode(*args, **kwargs)

        self.pipe.vae.decode = decode_after_empty_cache

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, torch.cuda.OutOfMemoryError) or super().is_oom(error)

    def generate(self, request: GenerateRequest, output_dir: Path, progress: ProgressCallback) -> GenerateResult:
        import einops
        import imageio

        text = str(request.inputs.get("text", "")).strip()
        if not text:
            raise AdapterError("invalid_input", "inputs.text is required")
        image = request.inputs.get("image")
        if not image or not Path(image).exists():
            raise AdapterError("invalid_input", f"start image not found: {image}")
        duration = float(request.params.get("duration", 5.0))
        frames = frames_for(duration)
        seed = request.params.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        steps = int(request.params.get("num_inference_steps", DEFAULT_STEPS))
        negative = str(request.params.get("negative_prompt", ""))

        progress(0.05)
        started = time.monotonic()
        out = self.pipe(
            enable_sr=False,
            prompt=text,
            aspect_ratio="16:9",
            num_inference_steps=steps,
            sr_num_inference_steps=None,
            video_length=frames,
            negative_prompt=negative,
            seed=seed,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            reference_image=str(image),
            # GPU 1 枚なので、VAE のデコードを複数 GPU に分ける経路（既定で有効）を使わない。
            # その経路は GPU 1 枚でも NCCL の all_gather を通り、全タイルの受け取り用コピーも作るため、
            # メモリに余裕がないと NCCL の確保に失敗する（"Failed to CUDA calloc async"、2026-10-12 に発生）
            enable_vae_tile_parallelism=False,
        )
        progress(0.95)

        video = out.videos
        if video.ndim == 5:
            video = video[0]
        frames_u8 = einops.rearrange((video * 255).clamp(0, 255).to(torch.uint8), "c f h w -> f h w c").cpu().numpy()
        name = "video.mp4"
        imageio.mimwrite(output_dir / name, frames_u8, fps=FPS, quality=9)
        progress(1.0)
        return GenerateResult(
            source_motion=None,
            raw_outputs=[name],
            outputs={"video": name},
            fps=FPS,
            num_frames=int(frames_u8.shape[0]),
            duration=frames_u8.shape[0] / FPS,
            seed=seed,
            extra={
                "model": f"HunyuanVideo-1.5/{self.version}",
                "steps": steps,
                "resolution": f"{frames_u8.shape[2]}x{frames_u8.shape[1]}",
                "video_seconds": time.monotonic() - started,
            },
        )


if __name__ == "__main__":
    serve(HyVideoAdapter(), port=int(os.environ.get("PORT", "8000")))
