"""ハイパーパラメータ設定。

プリセットを `PRESETS` に集約し、`--preset` で切り替える。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    batch_size: int      # 1回の学習で並列に処理する系列数 (B)
    block_size: int      # 文脈長。1系列あたりのトークン数 (T)
    n_embd: int          # 埋め込み次元 (C)
    n_head: int          # Attentionのヘッド数
    n_layer: int         # Transformerブロックの段数
    dropout: float       # ドロップアウト率
    learning_rate: float
    max_iters: int       # 学習ステップ数
    eval_interval: int   # 何ステップごとにlossを評価するか
    eval_iters: int      # lossを平均する際のバッチ数
    seed: int = 1337


PRESETS = {
    # Bigram用。単純なモデルなので学習率を高めにする。
    "bigram": Config(batch_size=32, block_size=8, n_embd=0, n_head=0, n_layer=0,
                     dropout=0.0, learning_rate=1e-2, max_iters=3000,
                     eval_interval=500, eval_iters=100),
    # 動作確認用。数十秒で終わる。
    "tiny": Config(batch_size=32, block_size=32, n_embd=32, n_head=4, n_layer=2,
                   dropout=0.0, learning_rate=1e-3, max_iters=300,
                   eval_interval=100, eval_iters=20),
    # CPUで数分〜十数分。それなりに「それっぽい」文が出る。
    "small": Config(batch_size=32, block_size=64, n_embd=96, n_head=4, n_layer=3,
                    dropout=0.1, learning_rate=1e-3, max_iters=2000,
                    eval_interval=250, eval_iters=50),
    # GPU向け(動画と同等の設定)。
    "full": Config(batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6,
                   dropout=0.2, learning_rate=3e-4, max_iters=5000,
                   eval_interval=500, eval_iters=200),
}
