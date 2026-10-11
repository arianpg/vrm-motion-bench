"""動画生成の開始画像を縮小し、背景を延長して余白を作る。

画像生成モデルは全身写真でも人物を画面いっぱいに描きがちで、そのままだとジャンプや前進で
人物が画面から出て GEM-X の推定が崩れる。背景が無地で床が均一な画像を前提に、画像を縮小し、
外側を端の色で延長してつなぎ目をぼかす（docs/research/gemx-text-to-video.md「開始画像」）。

使い方:
    uv run --no-project --with pillow --with numpy python tools/start_images/fit_start_image.py \
        assets/start_images/z-image_man3.png
    # → assets/start_images/z-image_man3_fit.png

元画像の PNG テキスト（ComfyUI の prompt / workflow）は引き継ぎ、縮小の条件を vmb_fit に追記する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo


def fit(image: Image.Image, scale: float, bottom_margin: float, feet_offset: float, feather: int) -> Image.Image:
    """scale: 縮小率。bottom_margin: 足元を画面下端からどれだけ上に置くか（画面の高さに対する割合）。
    feet_offset: 元画像で足元が下端からどれだけ上にあるか（元画像の高さに対する割合）。"""
    image = image.convert("RGB")
    W, H = image.size
    small = image.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    w, h = small.size
    x0 = (W - w) // 2
    # 元画像の足元の位置が、出力の下端から bottom_margin の位置に来るように置く
    y0 = round(H * (1 - bottom_margin)) - round(h * (1 - feet_offset))
    y0 = max(0, min(H - h, y0))

    arr = np.asarray(small, dtype=np.float32)
    canvas = np.pad(arr, ((y0, H - h - y0), (x0, W - w - x0), (0, 0)), mode="edge")
    # 延長した部分をなめらかにし、元画像の縁とのつなぎ目をぼかす
    blurred = np.asarray(Image.fromarray(canvas.astype(np.uint8)).filter(ImageFilter.GaussianBlur(25)), dtype=np.float32)
    mask = np.zeros((H, W), np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = 255
    soft = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(feather)), dtype=np.float32) / 255
    soft = np.minimum(soft * 1.6, 1.0)  # 元画像の内側はそのまま残す
    out = canvas * soft[..., None] + blurred * (1 - soft[..., None])
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="既定は <入力>_fit.png")
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--bottom-margin", type=float, default=0.12)
    parser.add_argument("--feet-offset", type=float, default=0.04)
    parser.add_argument("--feather", type=int, default=40)
    args = parser.parse_args()

    src = Image.open(args.input)
    out = fit(src, args.scale, args.bottom_margin, args.feet_offset, args.feather)

    info = PngInfo()
    for key, value in src.info.items():
        if isinstance(value, str):
            info.add_text(key, value)
    params = {
        "source": args.input.name,
        "scale": args.scale,
        "bottom_margin": args.bottom_margin,
        "feet_offset": args.feet_offset,
        "feather": args.feather,
        "tool": "tools/start_images/fit_start_image.py",
    }
    info.add_text("vmb_fit", json.dumps(params, ensure_ascii=False))

    output = args.output or args.input.with_name(f"{args.input.stem}_fit.png")
    out.save(output, pnginfo=info)
    print(f"saved: {output} {out.size}")


if __name__ == "__main__":
    main()
