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
