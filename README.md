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

要[uv](https://docs.astral.sh/uv/)。依存関係は `pyproject.toml` で管理し、`uv.lock` で固定している。

```bash
make                 # 全自動: セットアップ → データ取得 → 学習 → 対話
make PRESET=tiny     # 数十秒で試す (tiny / small / full)
make train           # 学習のみ (学習済みならスキップ)
make retrain         # 強制再学習
make generate        # 学習済みモデルで生成のみ
make bigram          # Phase 3: Bigramベースライン (val loss ≒ 2.5)
make test            # 単体テスト
```

Makefile は依存関係(DAG)になっており、足りない工程だけ実行される。
2回目以降は学習済みモデル `out/gpt-<preset>.pt` を再利用して、すぐ対話が始まる。

```
.venv/.synced ─┐
data/input.txt ┼─> out/gpt-<preset>.pt ─> chat / generate
gpt/*.py ──────┘
```

`small` はCPUで約2分(val loss ≒ 1.8)、`full` は動画相当の設定でGPU推奨(val loss ≒ 1.5)。

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
