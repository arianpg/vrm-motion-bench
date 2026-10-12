# VRM Motion Bench

ローカルのAIモーション生成モデルの出力を VRM に適用して比較する検証ツール。仕様は [docs/spec.md](docs/spec.md)。

## 構成

| ディレクトリ | 内容 |
| --- | --- |
| `backend/` | オーケストレーター（FastAPI）。ジョブキュー、アダプター・LLM コンテナの起動・停止、SSE、レポート生成 |
| `frontend/` | React + three-vrm の画面と、レポート HTML のテンプレート |
| `config/llm.yaml` | プロンプト支援に使う LLM（llama-server）の設定 |
| `adapters/<id>/` | アダプター（`manifest.yaml`、`bone_map.yaml`、Docker イメージ） |
| `converter/` | 共通コンバーター（Blender 常駐。リターゲット → VRMA・BVH・プレビュー WebP） |
| `packages/sourcemotion/` | SourceMotion NPZ の読み書き、BVH 読み込み、リターゲット計算（numpy のみ） |
| `packages/adapter_kit/` | アダプター共通の HTTP サーバー（`/health`、`/generate`、`/progress`） |
| `assets/stickman/` | 基準モデルの棒人間 VRM（`tools/stickman/build_stickman.py` で生成） |
| `tests/fixtures/adapters/dummy/` | パイプライン検証用のダミーアダプター |

実行時のデータは `data/`、モデルの重みは `models/<adapter_id>/` に置く（どちらも Git 管理外）。

## 準備

```bash
# Docker イメージ（ビルドコンテキストはリポジトリのルート）
docker build -f converter/Dockerfile -t vmb-converter .
docker build -f adapters/kimodo/docker/Dockerfile -t vmb-adapter-kimodo .
docker build -f adapters/ardy/docker/Dockerfile -t vmb-adapter-ardy .
docker build -f adapters/momask/docker/Dockerfile -t vmb-adapter-momask .   # CPU で動く
docker build -f adapters/hymotion/docker/Dockerfile -t vmb-adapter-hymotion .
docker build -f adapters/hyvideo/docker/Dockerfile -t vmb-adapter-hyvideo .     # 動画生成（GEM-X の前段）
docker build -f adapters/gemx/docker/Dockerfile -t vmb-adapter-gemx .
docker build -f adapters/emage/docker/Dockerfile -t vmb-adapter-emage .

# Kimodo の重み（初回起動時にも自動で取得されるが、事前に置いておくと速い）
#   Llama-3-8B-Instruct は Hugging Face でのアクセス申請と HF_TOKEN が必要
export HF_HOME=$PWD/models/kimodo/hf
hf download nvidia/Kimodo-SOMA-RP-v1.1
hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
hf download meta-llama/Meta-Llama-3-8B-Instruct --exclude 'original/*'

# ARDY の重み。テキストエンコーダ（LLM2Vec / Llama-3-8B）は Kimodo と同じなので、
# 同じディスク上ならハードリンクで共有する（容量を使わない）
mkdir -p models/ardy/hf/hub
cp -al models/kimodo/hf/hub/blobs models/kimodo/hf/hub/models--McGill-NLP--* \
       models/kimodo/hf/hub/models--meta-llama--* models/ardy/hf/hub/
HF_HOME=$PWD/models/ardy/hf hf download nvidia/ARDY-Core-RP-20FPS-Horizon40

# MoMask の重み（Google Drive。公式の prepare/download_models.sh と同じファイル）
mkdir -p models/momask/checkpoints/t2m && cd models/momask/checkpoints/t2m
gdown 1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0 -O humanml3d_models.zip && unzip humanml3d_models.zip && rm humanml3d_models.zip
cd -   # CLIP（ViT-B/32）は初回起動時に models/momask/clip へ自動でダウンロードされる

# HY-Motion 1.0 の重みとテキストエンコーダ（CLIP-L、Qwen3-8B）。合わせて約 22GB
export HF_HOME=$PWD/models/hymotion/hf
hf download tencent/HY-Motion-1.0 --include 'HY-Motion-1.0/*' --local-dir models/hymotion/ckpts/tencent
hf download openai/clip-vit-large-patch14 --exclude '*.msgpack' --exclude '*.h5' --exclude 'pytorch_model.bin'
hf download Qwen/Qwen3-8B
unset HF_HOME

# HunyuanVideo 1.5（I2V、480p ステップ蒸留版）。テキストから GEM-X の入力動画を作るのに使う。合わせて約 55GB
#   画像エンコーダ（SigLIP）は、公式手順の FLUX.1-Redux-dev（ゲート付き）の代わりに同じ構成の公開版を使う
export HF_HOME=$PWD/models/hyvideo/hf C=models/hyvideo/ckpts
hf download tencent/HunyuanVideo-1.5 --include 'config.json' --include 'scheduler/*' --include 'vae/*' \
   --include 'transformer/480p_i2v_step_distilled/*' --local-dir $C
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir $C/text_encoder/llm
hf download google/byt5-small --exclude '*.msgpack' --exclude '*.h5' --local-dir $C/text_encoder/byt5-small
hf download nlpcvcode/Glyph-SDXL-v2 --include 'assets/*.json' --include 'checkpoints/byt5_model.pt' --local-dir $C/text_encoder/Glyph-SDXL-v2
hf download google/siglip-so400m-patch14-384 config.json model.safetensors --local-dir $C/vision_encoder/siglip/image_encoder
hf download google/siglip-so400m-patch14-384 preprocessor_config.json --local-dir $C/vision_encoder/siglip/feature_extractor
unset HF_HOME C

# GEM-X（動画 → SOMA77）。SAM 3D Body の重みを含む。合わせて約 7GB
export HF_HOME=$PWD/models/gemx/hf I=models/gemx/inputs
hf download nvidia/GEM-X gem_soma.ckpt --local-dir $I/pretrained
hf download nvidia/GEM-X vitpose.pth --local-dir $I/checkpoints/vitpose
hf download nvidia/GEM-X sam3d_body.ckpt model_config.yaml --local-dir $I/checkpoints/sam-3d-body-dinov3
hf download nvidia/GEM-X mhr_model.pt --local-dir $I/mhr_data
hf download nvidia/GEM-X scale_mean.pth scale_comps.pth --local-dir $I/soma_data
unset HF_HOME I

# EMAGE（音声 → 全身の身振り）の重みと SMPL-X のモデル（骨格と口の動きの計算に使う）。合わせて約 750MB
HF_HOME=$PWD/models/emage/hf hf download H-Liu1997/emage_audio
HF_HOME=$PWD/models/emage/hf hf download H-Liu1997/emage_evaltools smplx_models/smplx/SMPLX_NEUTRAL_2020.npz \
   --local-dir models/emage/evaltools

# プロンプト支援の LLM（llama-server の公式イメージ）
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda13
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q5_K_M.gguf --local-dir models/llm/qwen3.8-27b

# Python（uv ワークスペース）とフロントエンド（レポートのテンプレートもビルドする）
uv sync --all-packages
(cd frontend && npm install && npm run build:report)
```

## 起動

```bash
uv run --package vmb-backend vmb-backend      # http://0.0.0.0:8800（起動時にコンバーターのコンテナも立ち上がる）
(cd frontend && npx vite --host)              # http://<GPU機>:5173
```

主な環境変数: `VMB_DATA_DIR`、`VMB_MODELS_DIR`、`VMB_ADAPTERS_DIR`、`VMB_PORT`、`HF_TOKEN`。
`VMB_CONVERTER_URL` を指定すると、コンテナの代わりにホストで動かしている変換ワーカーを使う（開発用）。

ダミーアダプターも使う場合は、`adapters/` の各アダプターと `tests/fixtures/adapters/dummy` へのシンボリックリンクを置いたディレクトリを `VMB_ADAPTERS_DIR` に指定し、`docker build -f tests/fixtures/adapters/dummy/Dockerfile -t vmb-adapter-dummy .` でイメージを作る。

## テスト

```bash
uv run --package sourcemotion --with pytest pytest -q packages/sourcemotion/tests
uv run --package vmb-backend --with pytest pytest -q backend/tests
(cd frontend && npx tsc -b --noEmit)
```
