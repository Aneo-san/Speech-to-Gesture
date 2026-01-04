# Speech-to-Gesture (Realtime / Light)

日本語の簡潔かつ実用的なREADMEです。ここではローカル環境でのセットアップ、実行方法、含まれるファイルの説明、トラブルシューティングをまとめています。

**プロジェクト概要**
- 音声入力（マイクまたは音声ファイル）からリアルタイムでジェスチャ（BVH/FKベース）を推論可視化するための軽量推論コードと学習済みモデルを含みます。
- 学習用の大規模データやトレーニングパイプラインは含まれていません。推論用途に特化しています。

**特徴**
- マイク入力からのリアルタイム推論と可視化（2D/3D）
- 軽量な LSTMP 推論モデルを同梱（推論向けに最適化）
- 最小構成での実行を想定（依存は requirements.txt に記載）

**同梱ファイル（主なもの）**
- [Speech_to_Gesture/s2g_2d.py](Speech_to_Gesture/s2g_2d.py)  2D 表示用のリアルタイム推論スクリプト（マイク入力音声特徴モデル2D可視化）
- [Speech_to_Gesture/s2g_3d.py](Speech_to_Gesture/s2g_3d.py)  3D 表示用（簡易的な FK 表示）
- [Speech_to_Gesture/lstmp.pt](Speech_to_Gesture/lstmp.pt)  推論に使う学習済みモデル（バイナリ）
- [Speech_to_Gesture/wayne_skeleton.bvh](Speech_to_Gesture/wayne_skeleton.bvh)  軽量スケルトン（HIERARCHY のみ）
- [Speech_to_Gesture/download_model.py](Speech_to_Gesture/download_model.py)  モデルをダウンロードする簡易スクリプト
- [Speech_to_Gesture/README.md](Speech_to_Gesture/README.md)  このファイル

（ワークスペース上位にも類似のスクリプトやモデルが存在する場合があります。ルート直下のファイルと混同しないよう注意してください。）

**前提 / 必要条件**
- OS: Windows / macOS / Linux（GUI 描画のため matplotlib のバックエンドに依存）
- Python 3.8+
- 必要パッケージは [requirements.txt](../requirements.txt) を参照してインストールしてください。

推奨セットアップ例（仮想環境）:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1   # PowerShell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

**使い方（基本）**
1. モデルファイルを配置する
- [Speech_to_Gesture/lstmp.pt](Speech_to_Gesture/lstmp.pt) をこのディレクトリに置くか、適切なパスに配置します。ファイルがリポジトリに無い場合は配布元からダウンロードしてください。

2. モデルをダウンロードする（簡易スクリプト）

```powershell
python Speech_to_Gesture/download_model.py --url "https://example.com/path/to/lstmp.pt" --out Speech_to_Gesture/lstmp.pt
```

注意: private/認証が必要なホスティング（Hugging Face private, Google Drive 共有など）はこのスクリプトではそのまま動かない可能性があります。その場合は `huggingface_hub` などの専用ツールを使ってください。

3. 2D 表示で動かす（マイク入力）:

```powershell
python Speech_to_Gesture/s2g_2d.py
```

音声ファイルを入力して推論したい場合（スクリプトがファイル入力に対応している前提）:

```powershell
python Speech_to_Gesture/s2g_2d.py --audio path\to\example.wav
```

（スクリプトがコマンドライン引数を受け取らない場合は、`s2g_2d.py` 内の入力設定を編集してファイル入力に切り替えてください。）

4. 3D 表示で動かす（マイク入力）:

```powershell
python Speech_to_Gesture/s2g_3d.py
```

**スケルトンと出力**
- 出力は BVH 互換のフォーマット（ジョイント角度等）を想定しており、簡易 FK 表示で可視化されます。
- スケルトン定義は [Speech_to_Gesture/wayne_skeleton.bvh](Speech_to_Gesture/wayne_skeleton.bvh) を参照してください。

**Windows 固有の注意点**
- PowerShell で仮想環境を有効化する際は実行ポリシーのために次を実行する必要がある場合があります（管理者として実行しないでください。必要ならポリシーを一時的に変更してください）:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
```

- matplotlib の GUI（TkAgg 等）を使う場合、Windows に Tk がインストールされていないとエラーになります。`python -m pip install tk` で試すか、Python の配布に同梱されている Tk を有効にしてください。
- マイクアクセスの許可: Windows 設定でアプリのマイクアクセスが無効だと入力が取れません。設定 > プライバシー > マイク でターミナル/コンソールアプリのアクセスを許可してください。
- オーディオデバイスが複数ある場合、スクリプト内で入力デバイスを明示的に指定してください（`sounddevice` 等を使っている場合、`device` 引数を設定する必要があります）。

**注意トラブルシューティング（共通）**
- GUI が表示されない／matplotlib のバックエンドエラーが出る場合は、環境変数や matplotlib のバックエンドを調整してください（例: `MPLBACKEND=TkAgg`）。
- モデルロード時にメモリ不足が起きる場合は、CPU 実行に切り替えるか、軽量なモデルを使ってください。

**開発メモ / カスタマイズ**
- 音声特徴抽出（例: HuBERT 等） モデル  ポストプロセッシング の流れです。各段階を差し替えて実験できます。
- 推論のレイテンシ計測やベンチマークはスクリプト内の BENCH 表示（あれば）を利用してください。

**貢献ライセンス**
- 本リポジトリは研究/実験用途向けです。商用利用や再配布は同梱のライセンス表記に従ってください。必要であればライセンスファイルを追加してください。
- バグ報告や機能追加は Issues を立ててください。プルリク歓迎です。

---

その他、README に追記してほしい実行例（特定のコマンドライン引数、Hugging Face 連携、音声ファイルでのバッチ推論手順など）があれば教えてください。
