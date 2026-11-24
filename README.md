# ai_build_gpt

Andrej Karpathy の動画 [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
を参考に、小型のGPTをPyTorchでゼロから実装するプロジェクト。

## GPTとは

**GPT = Generative Pre-trained Transformer**

- **Generative(生成的)**: 次のトークンを予測し続けて文章を生成する
- **Pre-trained(事前学習済み)**: 大量のテキストで事前に学習する
- **Transformer**: Attention(注意機構)ベースのニューラルネット構造(GPTはデコーダ部のみ)

中身は「ここまでの文字列から、次の1トークンの確率分布を出す関数」。
これを1トークンずつ繰り返しサンプリングすると文章になる。
ChatGPTは、この基盤モデルに指示追従・人間のフィードバック学習(SFT / RLHF)を加えたもの。
このリポジトリで作るのは事前学習部分のみ(Shakespeareの文字レベル言語モデル)。

## 使い方

```bash
make setup       # venv作成 + PyTorchインストール
make test        # 単体テスト
make bigram      # Phase 3: Bigramベースライン (val loss ≒ 2.5)
make train-tiny  # GPTの動作確認 (数十秒)
make train       # GPT学習 + 生成 (CPUで約2分, val loss ≒ 1.8)
make train-full  # 動画相当の設定 (GPU推奨, val loss ≒ 1.5)
make generate    # 保存済み out/gpt.pt で生成のみ
```

`make train PRESET=tiny|small|full` でプリセットを切替。データ(tiny shakespeare)は `make data` で自動取得される。

## 構成

| ファイル | 内容 |
|---|---|
| `gpt/data.py` | 文字トークナイザ、train/val分割、バッチ生成 |
| `gpt/bigram.py` | Bigramモデル(ベースライン) |
| `gpt/model.py` | `Head` → `MultiHeadAttention` → `FeedForward` → `Block` → `GPT` |
| `gpt/config.py` | ハイパーパラメータのプリセット |
| `gpt/train.py` | 学習・評価・生成のエントリポイント |
| `tests/` | トークナイザ、バッチ、Attentionの因果性、GPT forward/generate |

工程の詳細は [PLAN.md](PLAN.md) を参照(今回は Phase 1〜6 まで実装済み)。
