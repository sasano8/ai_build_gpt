"""Phase 2: データ処理(文字トークナイザ・train/val分割・バッチ生成)。"""
from pathlib import Path

import torch


class CharTokenizer:
    """1文字 = 1トークンの最小トークナイザ。"""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}
        self._itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self._stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_train_val(data: torch.Tensor, train_ratio: float = 0.9):
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu"):
    """ランダムな位置から (x, y) を切り出す。y は x を1文字左にずらしたもの。

    x: (B, T)  入力トークン列
    y: (B, T)  各位置の「次のトークン」(正解)
    """
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
