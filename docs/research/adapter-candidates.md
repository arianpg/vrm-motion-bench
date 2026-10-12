# 他のアダプター候補の調査（2026-10-12）

仕様書「残タスク」#5 の調査結果。導入済みの 7 モデル（Kimodo・ARDY・MoMask・HY-Motion・GEM-X・EMAGE・EDGE）以外で、ローカルで動くモーション生成モデルを調べた。

**調査の範囲**

- 入力の種類は問わない（テキスト・動画・音声・音楽）
- RTX 5090（32GB）で動くこと
- 重みが公開されていて、個人の検証用途で使えること
- 調べたのは各リポジトリの README・モデルカード・GitHub の情報まで。動作確認はしていない（「要確認」とした点は導入前に確かめる）

## 結論

いまの比較の空白は「音声入力」と「音楽入力」が 1 モデルずつしかなく、モデル間の比較ができないこと。そこで、次の順で導入を勧める。

| 優先 | 候補 | 入力 | 理由 |
| --- | --- | --- | --- |
| 1 | **SemTalk** | 音声 | EMAGE と同じ BEAT2・SMPL-X（＋顔）の系統で、音声どうしの比較ができるようになる。EMAGE の骨格の対応表・口の表情の処理を使い回せる見込み |
| 2 | **Lodge** | 音楽 | EDGE とダンスどうしの比較ができるようになる。音楽の特徴が librosa の 35 次元で、Jukebox（約 10GB）が要らない |
| 3 | **GEM-SMPL** | テキスト・動画（音声・音楽は要確認） | 1 つのモデルで複数の入力を扱える。重みは GEM-X と同じ Hugging Face のリポジトリにあり、環境も GEM-X に近い |
| 4 | **MotionStreamer**（2026-10-12 導入） | テキスト | 回転を直接出力する（MoMask のような IK が要らない）。複数のテキストを続けてつなぐストリーミング生成ができる |

テキスト入力はすでに 4 モデル（＋動画経由の GEM-X）あるため、追加の優先度は下げた。動画入力は GEM-X があるので、比較対象として GVHMR を候補に残す。

## 候補の一覧

### 音声 → 身振り

| モデル | 公開 | 出力 | 重み | ライセンス | メモ |
| --- | --- | --- | --- | --- | --- |
| [SemTalk](https://github.com/Xiangyue-Zhang/SemTalk)（ICCV 2025） | コード・推論・重み（[HF](https://huggingface.co/X-Zhang/SemTalk)） | SMPL-X（BEAT2。顔の有無は要確認） | Base・Sparse の 2 種。単一話者版と 25 話者版 | リポジトリに明記なし（要確認） | 手持ちの wav から推論できる（`--audio_infer_path`）。音声特徴に hubert-large、文字起こしに faster-whisper-large-v3 を使う。SMPL-X のモデルは同梱しない。公式の環境は torch 2.1 |
| [GestureLSM](https://github.com/andypinxinliu/GestureLSM)（ICCV 2025） | コード・推論・Web デモ・重み（[HF](https://huggingface.co/pliu23/GestureLSM)） | SMPL-X（体・指。顔は要確認） | Diffusion・Shortcut・MeanFlow（1 ステップで最速） | MIT | 手持ちの音声とテキストから推論できる。単語の時刻合わせに Montreal Forced Aligner が必要で、環境づくりが SemTalk より重い。公式の環境は torch 2.1.2 |

### 音楽 → ダンス

| モデル | 公開 | 出力 | 重み | ライセンス | メモ |
| --- | --- | --- | --- | --- | --- |
| [Lodge](https://github.com/li-ronghui/LODGE)（CVPR 2024） | コード・重み（Google Drive） | SMPL-X（FineDance。指あり） | あり | リポジトリに明記なし。学習データの FineDance は非商用 | 長いダンスを 2 段階の拡散で作る。音楽の特徴は librosa の 35 次元（`extract_musicfea35.py`）。公式の環境は CUDA 11・pytorch-lightning 1.9.5 で、移植が必要 |
| [AtomicDance](https://github.com/oceanflowlab/AtomicDance)（ECCV 2026） | コード | SMPL | **未公開**（「公開予定」） | MIT | 重みが出たら再検討 |
| Lodge++ | 論文のみ | — | — | — | コード未公開 |
| [MACE-Dance](https://github.com/AMAP-ML/MACE-Dance)（SIGGRAPH 2026） | コード | SMPL のダンス → 動画 | 一部は申請制 | 明記なし | ダンス動画の生成が主目的のため対象外 |

### 複数の入力（テキスト・動画・音声・音楽）

| モデル | 公開 | 出力 | 重み | ライセンス | メモ |
| --- | --- | --- | --- | --- | --- |
| [GEM-SMPL](https://github.com/NVlabs/GENMO)（旧 GENMO、ICCV 2025） | コード・重み（2026-03） | SMPL | `gem_smpl.ckpt`（約 5.5GB）が [nvidia/GEM-X](https://huggingface.co/nvidia/GEM-X) にある | コードは NVIDIA OneWay Noncommercial、重みは NVIDIA Open Model License | 公開されているデモは「動画＋テキスト」の組み合わせ（`--input_list video.mp4 "text:..."`、テキスト 1 区間は既定で 10 秒）。**テキストだけ・音声・音楽での推論は README に例がなく要確認**。前処理（YOLOX・ViTPose・HMR2）は GEM-X と似ている |

### テキスト → モーション

| モデル | 公開 | 出力 | 重み | ライセンス | メモ |
| --- | --- | --- | --- | --- | --- |
| [MotionStreamer](https://github.com/zju3dv/MotionStreamer)（ICCV 2025） | コード・推論デモ・重み | 272 次元（関節位置と 6D 回転）。`--mode rot` で回転 | あり（[HF](https://huggingface.co/lxxiao/MotionStreamer)） | 明記なし（データは AMASS の条件で学術用途のみ） | テキストエンコーダに sentence-T5-XXL を使う。HumanML3D・BABEL で学習 |
| [Kimodo-SMPLX-RP-v1](https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1) | 重み（2026-03） | SMPL-X（AMASS 形式の NPZ） | あり | NVIDIA R&D Model License | 導入済みの Kimodo の別版。同じアダプターで骨格だけ変えて比べられる（リターゲットの影響を見るのに使える）。手間は小さい |
| [MotionGPT3](https://github.com/OpenMotionLab/MotionGPT3) | コード・重み（2025-10） | HumanML3D（関節位置） | あり | 明記なし | MoMask と同じく IK が必要。同じ学習データのため優先度は低い |
| [MARDM](https://github.com/neu-vi/MARDM)（CVPR 2025） | コード・重み | HumanML3D 系 | あり（HF に移行） | MIT | 同上 |
| [MotionLCM-V2](https://github.com/Dai-Wenxun/MotionLCM) | コード・重み | HumanML3D 系 | あり | 明記なし | 高速（リアルタイム向け）。同上 |
| [SALAD](https://github.com/seokhyeonhong/salad)（CVPR 2025） | コード・重み | HumanML3D 系 | あり | 明記なし | torch 1.13 固定。同上 |
| OpenT2M / MonoFrill（2026-03） | 論文 | — | 未確認 | — | 大規模な公開データセットの論文。重みは確認できず |

### 動画 → モーション

| モデル | 公開 | 出力 | 重み | ライセンス | メモ |
| --- | --- | --- | --- | --- | --- |
| [GVHMR](https://github.com/zju3dv/GVHMR)（SIGGRAPH Asia 2024） | コード・重み | SMPL（世界座標） | あり | 非商用の研究用途 | 動画からのモーション推定の定番。GEM-X との比較対象になる |
| [FlowHMR](https://github.com/flowhmr/flowhmr)（2026-10） | コード・重み（[HF](https://huggingface.co/fafsaf1/flowhmr-0.46B)） | SMPL 系 | 0.46B | 明記なし | 物理的にもっともらしい動きを重視する最新の手法。VGGT-Omega と SAM 3D Body はゲート付き。公開から日が浅いので様子を見る |
| [TRAM](https://github.com/yufu-wang/tram)・[WHAM](https://github.com/yohanshin/WHAM) | コード・重み | SMPL | あり | MIT | 世代が古く、GEM-X・GVHMR より優先度は低い |

### 対象外としたもの

- **動画生成モデル**（Wan-Dancer-14B など）：出力が動画でモーションではない。ただし、音楽からダンス動画を作って GEM-X に通す使い方（GEM-X のテキスト → 動画と同じ形）はできる
- **コード・重みが確認できなかったもの**：GestureFAR、LiveGesture、MIBURI、StreamTalk など（2026 年のストリーミング系の身振り生成）、Lodge++、OpenT2M

## 導入時の共通の注意

- SMPL・SMPL-X のモデルファイルは各リポジトリが同梱しないことが多い。EMAGE で使っている `SMPLX_NEUTRAL_2020.npz` を共有できるかを確かめる
- 公式の環境が古い（torch 1.x・2.1、CUDA 11）ものは、EDGE と同じく PyTorch 2.8（CUDA 12.8）への移植が要る
- ライセンスが明記されていないもの（SemTalk・Lodge・MotionStreamer など）は、個人の検証用途に限って使い、GitHub 公開時の一覧（残タスク）に「明記なし」として載せる
