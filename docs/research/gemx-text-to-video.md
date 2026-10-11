# GEM-X をテキストで比較するための中間動画生成 調査メモ

Oct 11, 2026 · @arianpg

GEM-X は動画を入力にするため、ほかのアダプター（テキスト入力）と同じ依頼文で比較できない。
そこで「テキスト → 動画生成 → GEM-X」と、生成した動画を GEM-X の入力にする方式を調べた。

## 決定事項

| 項目 | 決定内容 |
| --- | --- |
| 方式 | 中間に動画生成を挟んで GEM-X でモーションにする |
| 動画生成モデル | HunyuanVideo 1.5 を採用する。満足いく結果が出なければ別のモデルを検討する |
| ライセンス | HunyuanVideo 1.5 は Tencent Community License（EU・英国・韓国では適用外）。実施環境は日本なので問題なし |
| 生成方式 | 画像から動画を作る方式（I2V）。構図をそろえた開始画像を全プロンプトで共有する |
| 開始画像 | 事前に生成しておく（方法は「開始画像」の節） |

## 1. 方式の妥当性

技術的には成り立つ。ただし動画モデルの欠点がそのまま動きに出るので、その前提で比較する。

- **前例**：World2Motion（arXiv 2609.37004）は「動画を生成し、人体推定（HMR）で動きを取り出す」2 段の方式を比較の基準として扱っている。課題として、推論コストと推定した動きの時間方向の不安定さを挙げている
- **生成動画の人の動きの弱点**：HumanScore（arXiv 2604.20157）は 13 モデルを評価し、細かい揺れ、関節の可動域を超えた姿勢、手足の重複、体の位置のずれ、激しい動作での手足の伸びや足の接地の不安定さを報告している。見た目が自然でも動きは生体力学的に正しくないことがあり、動きが難しくなるほど成績が下がる
- **処理時間**：1 件あたり「動画生成（約 1〜7 分）＋ GEM-X の推論」。既存のアダプター（1〜20 秒）より桁違いに遅い
- **既存の仕組みとの相性**：動画モデルも GEM-X も GPU を使うので、GPU キューで順番に入れ替えて実行できる。アップロードと LLM のプロンプト支援もそのまま使える

**作り方の案**：動画生成を独立したアダプター（ジョブ）にし、出力した動画を GEM-X の入力ファイルとして渡す。実写の動画をアップロードした場合と、生成した動画を使った場合を同じ GEM-X で比べられる。生成した動画は結果と一緒に保存し、画面で見られるようにする。

## 2. GEM-X の入力に適した動画

GEM-X の処理は、YOLOX で人を検出 → ByteTrack で追跡 → ViTPose で 2D 関節（77 点）→ SAM3D で画像特徴 → GEM で 3D 姿勢。
カメラは既定では Visual Odometry で扱い、`-s`（`--static_cam`）で固定カメラとして扱う。
公式資料には解像度・fps・長さ・人数の条件が書かれていないため、以下は処理の流れからの推奨で、実機での検証が必要。

| 項目 | 推奨 | 理由 |
| --- | --- | --- |
| 人数 | 1 人だけ | 追跡する人の選び方が書かれていない。他の人が映ると取り違える恐れ |
| 映る範囲 | 最初から最後まで頭から足先まで全身 | 足が切れると接地と全体の位置の推定が崩れる |
| カメラ | 固定（`-s`）、ズームなし | 生成動画のカメラの動きは物理的に一貫しないことが多い |
| アングル | 斜め前（45° 程度）、目の高さ、やや引き | 真正面だと前後の動きが判別しにくい |
| 人物の大きさ | 画面の高さの 5〜8 割 | 小さすぎると関節の精度が落ち、大きすぎると歩いたときに画面から外れる |
| 背景・床 | 無地の背景、床が見える | 誤検出を減らし、地面との関係を分かりやすくする |
| 服装 | 体の線が分かる服、背景とのコントラスト | 関節の位置が分かりやすい |
| 映像の作り | カット割りなし、スローモーションなし、実写調 | 動画モデルは演出を勝手に入れることがある |
| 解像度・長さ | 720p 前後、24〜30fps、10 秒以内 | 動画モデルの得意な範囲と、ほかのアダプターとの比較条件に合わせる |

## 3. 動画生成モデルの候補（RTX 5090 32GB）

| モデル | HumanScore（実写は 94.3） | RTX 5090 での実行 | ライセンス | 評価 |
| --- | --- | --- | --- | --- |
| **HunyuanVideo 1.5**（8.3B） | **91.1（オープンモデルで 1 位、全体でも同点 1 位）** | 公式の最小は 14GB（退避あり）。720p・5 秒で約 22GB・約 2.5 分（ブログの値、未検証）。480p の I2V 高速版（8〜12 ステップ）あり | Tencent Community（EU・英国・韓国では適用外） | **採用** |
| Wan 2.2（14B MoE / 5B） | 86.1 | 5B は軽量、14B は 720p で 24GB 以上 | Apache 2.0 | 第二候補。ライセンスが最も扱いやすい |
| Kandinsky 5.0 pro | 86.5 | 未確認 | 未確認 | 評価は良いが情報が少ない |
| LTX-2.5（22B ＋ Gemma 4 12B） | 評価対象外 | int8 なら 32GB で動く。Blackwell 向け NVFP4 版あり。8 ステップで高速 | LTX-2.x Community（年間売上 1000 万ドル未満は無料） | 速度は魅力だが人の動きの質は要検証 |
| MiniMax-H3（33B、推論時は約 20B） | 評価対象外（前世代の Hailuo 02 は 85.8） | int8 と CPU 退避で動く。ピーク VRAM 約 31.8GB、メインメモリ 70〜93GB。8 秒・0.7MP で約 7 分 | MiniMax H3 Community（米国・EU・英国・韓国は申請が必要。生成物の商用利用は別ライセンス） | 重く VRAM がほぼ上限 |

- VRAM と速度の数値の多くは第三者のブログで、ばらつきがある
- Wan 2.5 以降は API のみで、重みは公開されていない

### HunyuanVideo 1.5 の要点

- T2V と I2V に対応。480p / 720p（超解像で 1080p）。既定は 121 フレーム（24fps で約 5 秒）
- I2V は `--image_path` で開始画像を渡す
- 内蔵のプロンプト書き直し（`--rewrite`）は既定で有効で、vLLM の外部エンドポイント（Qwen3-235B 系）が必要。本ツールでは使わず、LLM プロンプト支援で代替する（HY-Motion と同じ方針）
- 注意の実装は Flash Attention / flex-block-attn / SageAttention から選ぶ

## 4. 開始画像

構図・体型・服装をそろえた開始画像を事前に用意し、全プロンプトで共有する。開始姿勢は「斜め前を向いて自然に立つ」とする
（どのモーションも立った状態から始まる。ほかのモーション生成モデルも立った状態から始まることが多いので、比較の条件はそろう）。

作り方の候補:

| 方法 | 内容 | 長所 | 短所 |
| --- | --- | --- | --- |
| A. 画像生成モデルで事前に作る | Z-Image Turbo（6B、Apache 2.0、16GB で動く）や Qwen-Image 2512（Apache 2.0、24GB 以上）で、構図ルールを満たす画像を数枚作り、プリセットとして保存する | 画質と構図を細かく選べる。一度作れば使い回せる | 画像生成モデルの環境が別に必要（ただし一度きり） |
| B. HunyuanVideo 1.5 の T2V で作る | 「立って静止している人」の動画を T2V で作り、1 フレームを切り出す | 追加のモデルが不要 | 静止画としての画質は画像モデルに劣る可能性 |
| C. 実写の写真を使う | 撮影した写真や、権利的に問題のない写真を使う | 最も自然 | 用意の手間と、人物の権利の確認が必要 |

## 出典

- [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) / [GEM-X DEMO.md](https://github.com/NVlabs/GEM-X/blob/main/docs/DEMO.md)
- [World2Motion (arXiv 2609.37004)](https://arxiv.org/html/2609.37004)
- [HumanScore (arXiv 2604.20157)](https://arxiv.org/html/2604.20157)
- [Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) / [Runpod: LTX-2.5](https://www.runpod.io/blog/ltx-2-5-the-open-weights-world-model-built-for-speed-and-how-to-run-it-on-runpod)
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) / [MiniMax H3 local requirements (36 runs on 5090)](https://wan2-7.io/blog/minimax-h3-local-requirements/) / [Atlas Cloud: H3 weights & excluded countries](https://www.atlascloud.ai/blog/tips/minimax-h3-open-source-weights)
- [Thunder Compute: open-source video models 2026](https://www.thundercompute.com/blog/best-open-source-ai-video-generation-models) / [CrePal: LTX 2.3 vs Wan 2.2](https://crepal.ai/blog/aivideo/ltx-2-3-vs-wan-2-2/)
- [dev.to: RTX 5090 vs 4090 for video gen](https://dev.to/thurmon_demich/rtx-5090-vs-4090-for-video-gen-in-2026-32gb-matters-k3i)
- [InVideo: open-source image model licenses (Aug 2026)](https://invideo.io/blog/open-source-image-models-licenses/) / [Thunder Compute: open-source image models](https://thundercompute.com/blog/best-open-source-image-generation-models)
