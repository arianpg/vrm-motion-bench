# ライセンス一覧

VRM Motion Bench が使う・同梱するもののライセンスと、生成したモーションを公開するときの条件をまとめる（2026-10-12 時点。各リポジトリ・モデルカードの表記と、ライセンスの原文で確認した）。法的な助言ではない。ライセンスは更新されることがあるので、公開や商用の判断の前には原文を確かめる。

**前提**

- モデルの重み・学習データ・ボディモデル（SMPL など）は**同梱しない**。各自が配布元の条件に同意して取得する（README の「準備」）
- このリポジトリ自体のライセンスは未定（GitHub 公開の準備で決める）

## 出力を公開するときの条件

検証結果のモーションを動画などで公開するときに守ること。比較動画（MP4）とレポート HTML には、下の「表示する内容」が自動で入る（クレジットの定義は各アダプターの `manifest.yaml` の `credit`）。

| 条件 | 対象 | 根拠 |
| --- | --- | --- |
| AI 生成であることを、はっきり明示する | HY-Motion、GEM-X のテキスト入力（HunyuanVideo 1.5 で作った動画を経由） | Tencent の利用規定 12 項（公開の場に置くときは機械生成であることを明示的かつ目立つように示す） |
| 商用で使わない（収益化した投稿や、製品・サービスの宣伝を含む） | EDGE、EMAGE、MoMask、MotionStreamer | Jukebox（作った内容も商用不可）、SMPL・SMPL-X（非商用の研究・教育・芸術のみ）、AMASS・HumanML3D（非商用の研究用途。MotionStreamer の 272 次元版は学術用途） |
| Jukebox を使ったことを示す | EDGE | Jukebox のライセンス（OpenAI の Jukebox で作ったことを明示するよう求めている） |
| 偽の内容・誤解を招く内容・なりすましに使わない | EMAGE（SMPL-X）、HY-Motion・HunyuanVideo（Tencent） | SMPL-X Model License、Tencent の利用規定 13 項 |

**判断が分かれる点**

- Tencent の 2 つのライセンス（HY-Motion 1.0、HunyuanVideo 1.5）は EU・英国・韓国では適用されず、「地域の外で出力を表示してはならない」（5(c)）とある。日本から SNS に投稿した動画はこれらの地域でも見られるため、利用者の所在地の制限と読むか、閲覧される場所の制限と読むかで結論が変わる。避けたい場合は HY-Motion と GEM-X のテキスト入力の結果を公開対象から外す
- 学術的な引用：SMPL-X と SAM 3D Body は、出版物で使ったときに引用・謝辞を求めている。SNS の投稿が出版物に当たるかは曖昧なため、クレジットにモデル名を出している

**表示する内容**

- 比較動画（MP4）：下部に 1 行。「AI 生成のモーション」と、使ったモデル（例：`AI 生成のモーション · Kimodo (NVIDIA) · EDGE (Stanford) + OpenAI Jukebox`）
- レポート HTML：下部の「クレジット」欄に、モデルごとのクレジット・ライセンス・公開時の条件と、非商用の注意

**モデル以外で注意すること**

- 入力の音声・音楽：市販の曲や他人の音声を使った結果を、音声付きで公開すると著作権に触れる。公開用には自作の曲やフリー素材を使う。比較動画に音声を入れない選択もできる
- 実在の人物：人の声（EMAGE）や人が映った動画（GEM-X）を入力にするときは、本人の同意を得る
- ユーザーの VRM：表示に使う VRM は、そのモデルに設定された利用条件（VRM のメタ情報：アバターの利用許可・商用利用・改変の可否など）に従う

## アダプター

| アダプター | コード | 重み | テキストエンコーダなど | 学習データ・ボディモデル |
| --- | --- | --- | --- | --- |
| Kimodo | Apache 2.0 | NVIDIA Open Model License（商用可。出力の権利を主張しない） | LLM2Vec（MIT）＋ Meta-Llama-3-8B-Instruct（Llama 3 Community License。Llama を配布しない限り表示義務なし） | Bones Rigplay 1。SOMA（Apache 2.0） |
| ARDY | Apache 2.0 | NVIDIA Open Model Agreement（商用可。出力の権利を主張しない） | Kimodo と同じ | Bones Rigplay 1。SOMA（Apache 2.0） |
| MoMask | MIT | 明記なし | CLIP ViT-B/32（MIT） | HumanML3D（AMASS 由来。非商用の研究用途）。SMPL 由来の骨格 |
| HY-Motion 1.0 | Tencent HY-Motion 1.0 Community License | 同左（EU・英国・韓国では適用外。出力の権利を主張しない。出力で他の AI を改良してはならない） | CLIP ViT-L/14（MIT）、Qwen3-8B（Apache 2.0） | 同梱の木製人形モデル |
| GEM-X | Apache 2.0 | NVIDIA Open Model License | SAM 3D Body（SAM License。出版物では謝辞を求める）、ViTPose・YOLOX | SOMA・MHR（Apache 2.0） |
| HunyuanVideo 1.5（GEM-X の前段） | Tencent Hunyuan Community License | 同左（HY-Motion と同じ条件） | Qwen2.5-VL-7B（Apache 2.0）、byT5（Apache 2.0）、Glyph-SDXL-v2（明記なし）、SigLIP（Apache 2.0） | 開始画像は Z-Image Turbo（Apache 2.0）で生成 |
| EMAGE | PantoMatrix のリポジトリに明記なし | Apache 2.0（Hugging Face の表記） | — | BEAT2（Apache 2.0）。SMPL-X（SMPL-X Model License：非商用の研究・教育・芸術のみ） |
| MotionStreamer | Project Registration License v1.0（研究・教育・評価・個人の目的は登録不要。組織のプロジェクトでの利用は事前登録が必要） | 同左（Hugging Face） | sentence-T5-XXL（Apache 2.0） | HumanML3D の 272 次元版（AMASS 由来。学術用途に限る）。骨格は SMPL-X の体 22 関節（モデルファイルは使わず、出力から推定） |
| EDGE | MIT | MIT（リポジトリ） | Jukebox 5B（Noncommercial Use License：作った内容も商用不可。Jukebox の使用の明示を求める） | AIST++（注釈は CC BY 4.0、API は Apache 2.0。元の AIST Dance Video Database は独自の利用規約で、内容は未確認）。SMPL 由来の骨格（SMPL Model License：非商用） |

## そのほかのソフトウェア

| もの | ライセンス | 使い方 |
| --- | --- | --- |
| Qwen3.8-27B（プロンプト支援の LLM） | Apache 2.0 | GGUF を llama.cpp で動かす |
| llama.cpp（llama-server） | MIT | 公式の Docker イメージ |
| Blender | GPL-3.0 | コンバーターの Docker イメージに公式ビルドを入れる（同梱しない） |
| VRM Add-on for Blender | MIT | 同上 |
| ffmpeg | LGPL / GPL（ビルドによる） | プレビュー・比較動画・音声の変換 |
| three.js、three-vrm、three-vrm-animation | MIT | ビューア（レポート HTML に同梱される） |
| React | MIT | 画面（レポート HTML に同梱される） |

## このリポジトリに同梱するもの

| もの | 出どころ | 備考 |
| --- | --- | --- |
| 棒人間 VRM（`assets/stickman/stickman.vrm`） | `tools/stickman/build_stickman.py` で生成した自作 | レポート HTML にも埋め込まれる |
| 動画生成の開始画像（`assets/start_images/`） | Z-Image Turbo（Apache 2.0）で生成し、縮小・余白を加えたもの | 実在の人物ではない。公開前にメタ情報を確認する（残タスク「公開前の確認」） |
