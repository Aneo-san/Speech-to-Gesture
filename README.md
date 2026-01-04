# Speech_to_Gesture

## 概要
Real-time avatar gesture generation using HuBERT audio features.

音声入力からHuBERTの特徴量を抽出し、LSTMPモデルによってリアルタイムで生成されるアバタージェスチャー（2D/3D）を生成して可視化します。
すぐに簡単に試せるようにジェスチャーボーンはmatplotlibで映し出していますがUnity等で動かす方がはるかに軽量です。
デフォルトで内蔵カメラを使用するようにしています。

## ディレクトリ構成
- `src/` — 実行用スクリプト: `s2g_2d.py`, `s2g_3d.py`, `download_model.py`
- `models/` — 独自で学習させた学習済みLSTMPモデル
- `data/` — BVHなどのデータファイル（`wayne_skeleton.bvh`）

## 特徴
- リアルタイム音声からジェスチャーを生成するパイプライン
- 2D/3D の可視化オプションどちらでもお選びください
- 実行中に処理遅延を計測するベンチ表示（BENCH）機能があります

## 必要条件
- Python 3.9+ 推奨
- 主な依存パッケージ: `torch`, `transformers`, `sounddevice`, `matplotlib`, `numpy`
- 詳細はプロジェクトルートの `requirements.txt` を参照してください。

## 使い方
1. 仮想環境を作成して有効化:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. 学習済みモデルを `models/` に配置（例: ダウンロードスクリプトを使用）:

```bash
python src/download_model.py --url <MODEL_URL> --out models/lstmp.pt
```

3. 2D 実行（リアルタイム）:

```bash
python src/s2g_2d.py
```

4. 3D 実行:

```bash
python src/s2g_3d.py
```

## 各パラメータの説明
以下の変数は `src/s2g_2d.py` と `src/s2g_3d.py` の250行付近にあり、値を変更して即座に挙動を試せます。ご自由に触ってください。

- `chunk_sec` — 1チャンクの長さ（秒）: 小さくすると遅延が下がるが入力情報が減少。例: 0.05〜0.3。
- `hubert_win_sec` — HuBERT に渡すウィンドウ長（秒）: 長くすると文脈が増え滑らかだが遅延が増加。例: 0.5〜2.0。
- `fps_motion` — モーションの内部フレームレート: 高いと詳細だが計算量が増加。例: 15〜60（通常30）。
- `take_frames` / `T_win` — モデルに渡すフレーム数: 多いと一度に多く生成するが遅延・負荷増。
- `ema_alpha` — EMA の係数（0.0〜1.0）: 値が小さいほど最新の反応が優先（応答速いがジッタあり）、大きいほど過去を重視して滑らかに。ただしかなり重くなります。もっさり感をなくしたいならば0推奨です。
	- 例: クイックに反応したい場合 `ema_alpha = 0.0`、滑らかさ優先なら `0.7`〜`0.95`。
- `mic_sr` — マイクのサンプリングレート（入力 SR）: スクリプトは内部で16kにリサンプリングするため高SRはリサンプリングコストが増える。
- `block_sec` / `chunk_len` / `win_len` — コールバックやバッファ長: 小さくするとI/O回数が増えるが遅延は下がる。
- `audio_q` の `maxsize` — 入力キューのサイズ: 小さいと古いチャンクが捨てられやすく最新優先に。大きいとバックログが生じやすい。
- `enable_bench` — ベンチ計測を有効にするか: `False` にすると計測ログを止める（わずかなオーバーヘッド削減）。
- `device` 判定（`cuda` の有無） — GPU が使えると `hubert` / `model` の処理が大幅に高速化されます。

### チューニング例
- 低遅延: `chunk_sec=0.05`, `hubert_win_sec=0.5`, `ema_alpha=0.0`, `fps_motion=30`

## BENCH（ベンチマーク）について
スクリプト実行中に各処理段階の経過時間が表示されます。出力は次の順序です:

- `get`: キュー/入力からチャンク取得にかかる時間
- `resample`: 16kHz へのリサンプリング時間
- `hubert`: HuBERT 特徴抽出の時間
- `model`: LSTMP 推論時間
- `post`: 正規化・角度変換・キュー格納などの後処理時間
- `total`: エンドツーエンドの合計遅延

終了時（`q`キー）に平均値が表示されます。計測を無効化するにはソース内の `enable_bench = False` に変更してください。ctrl^cでも停止できます。

## 注意事項
- `models/` と `data/` は大きなバイナリを含むため通常はリポジトリに含めません（`.gitignore` に登録済み）。
- GPUが利用可能ならPyTorchが自動的にGPUを使いますが、環境とドライバの整備が必要です。
- 本リポジトリは研究紹介用のプロトタイプ版です。現在更なる軽量化と自然さの両立を進めています。研究資産故、今後も完全版ではなくプロトタイプ版を公開していく予定です。ただし研究発表後に完全版を公開するかもしれません。
- 最初に記述していますがmatplotlibではなくUnity等で動かす方が動作はより滑らかになります。

## ライセンス
MIT License — 詳細はリポジトリルートの `LICENSE` を参照してください。

## その他
Issue と Pull Request を歓迎しています。問題や改善提案は GitHub 上で提出してください。
