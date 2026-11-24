# GPTビルド実行計画

参考: Andrej Karpathy "Let's build GPT: from scratch, in code, spelled out"
(https://www.youtube.com/watch?v=kCc8FmEb1nY)

## 0. GPTとは

**GPT = Generative Pre-trained Transformer**

- Generative: 文章を「生成」する(次のトークンを予測し続ける)
- Pre-trained: 大量テキストで事前学習済み
- Transformer: 2017年「Attention Is All You Need」の注意機構ベースのニューラルネット構造(GPTはそのデコーダ部のみ)

本質は **「ここまでの文字列から、次の1トークンの確率分布を出す関数」**。
これを1トークンずつ繰り返してサンプリングすると文章になる。
ChatGPTは、この基盤モデルにさらに指示追従・人間のフィードバックによる調整(SFT / RLHF)を加えたもの。

本計画で作るのは「事前学習」部分のみの小型GPT(Shakespeareの文字レベル言語モデル)。

## 1. ゴールと完了条件

| 項目 | 内容 |
|---|---|
| ゴール | 小型GPTをPyTorchでゼロから実装し、学習・テキスト生成まで動かす |
| 完了条件 | ①学習ロスが下がる ②生成文がシェイクスピア風の体裁になる ③各部品を自分の言葉で説明できる |
| スコープ外 | RLHF、大規模データ、分散学習、ChatGPT級の対話能力 |

## 2. 前提・環境

- Python 3.10+、PyTorch、(任意)GPU。CPUでも極小設定なら可。GPUが必要ならGoogle Colab利用
- データ: tiny shakespeare (約1MB、`input.txt`)
- 事前知識: Python基礎、行列積、勾配降下の概念

## 3. フェーズ別工程

### Phase 1: 準備(目安 0.5h)
- [ ] 動画を一度通して視聴し、全体像を把握
- [ ] 環境構築(venv, torch)、tiny shakespeareを取得
- 成果物: `data/input.txt`、動作確認スクリプト

### Phase 2: データ処理(目安 1h)
- [ ] 文字レベルのトークナイザ(encode/decode)
- [ ] train/val 分割(90/10)
- [ ] `block_size`(文脈長)とバッチ生成 `get_batch`
- 確認: encode→decodeで元文字列に戻る

### Phase 3: ベースライン — Bigramモデル(目安 1.5h)
- [ ] `nn.Embedding` だけの最小言語モデル
- [ ] クロスエントロピー損失、`generate()`
- [ ] AdamWで学習ループ
- 確認: 損失が約4.2→約2.5に低下(文は無意味でOK)

### Phase 4: Self-Attentionの理解(目安 2h) ★最重要
- [ ] 平均による過去トークンの集約(for文→行列積→softmax+下三角マスク)
- [ ] Query / Key / Value による重み付き集約
- [ ] スケーリング(1/√head_size)の理由
- [ ] 単一ヘッド Attention を `Head` クラス化
- 確認: 未来のトークンを見ていない(マスクが効いている)

### Phase 5: Transformerブロック構築(目安 2h)
- [ ] 位置埋め込み(position embedding)
- [ ] Multi-Head Attention(複数ヘッドを並列→連結)
- [ ] FeedForward(Linear→ReLU→Linear)
- [ ] 残差接続(residual)、LayerNorm、Dropout
- [ ] `Block` を `n_layer` 個積む
- 確認: Bigramよりval lossが明確に下がる(約1.5前後)

### Phase 6: 学習・評価(目安 1.5h)
- [ ] ハイパーパラメータ設定(n_embd, n_head, n_layer, block_size, lr)
- [ ] train/val loss の定期評価
- [ ] 生成テキストの確認
- 確認: シェイクスピア風の台詞形式の出力

### Phase 7: 発展・整理(目安 2h)
- [ ] GPT-2 / ChatGPTとの差分整理(BPE、規模、SFT、RLHF)
- [ ] (任意)BPEトークナイザ(tiktoken)に差し替え
- [ ] (任意)自分のデータ(日本語テキスト等)で学習
- [ ] README・学習メモ整備

## 4. 想定ディレクトリ構成

```
ai_build_gpt/
├── PLAN.md
├── README.md
├── data/input.txt
├── bigram.py      # Phase 3
├── gpt.py         # Phase 5-6 の最終版
└── notes/         # 各Phaseの学習メモ
```

## 5. リスクと対策

| リスク | 対策 |
|---|---|
| CPUで学習が遅い | モデルを縮小(n_layer=2, n_embd=64, block_size=64)、またはColab GPU |
| Attentionの理解が詰まる | Phase 4を小さなテンソルで手計算・print確認 |
| ロスが下がらない | 学習率、マスク漏れ、バッチ形状(B,T,C)を確認 |
| 動画の内容を丸写しして理解が浅くなる | 各Phase後に何も見ずに書き直す |

## 6. 総工数目安

約10〜11時間(動画視聴+実装+復習)。2〜3日に分割して進行。
