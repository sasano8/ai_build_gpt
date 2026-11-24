"""最小限の動作確認テスト。 `make test` で実行。"""
import torch

from gpt.data import CharTokenizer, get_batch
from gpt.model import GPT, Head


def test_tokenizer_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("low")) == "low"


def test_get_batch_shift():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(x + 1, y)  # yはxを1つ先にずらしたもの


def test_attention_is_causal():
    """未来のトークンを変えても、過去位置の出力は変わらない。"""
    torch.manual_seed(0)
    head = Head(n_embd=16, head_size=8, block_size=10, dropout=0.0)
    x = torch.randn(1, 10, 16)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 5, 16)  # 位置5以降を差し替え
    assert torch.allclose(head(x)[:, :5], head(x2)[:, :5], atol=1e-6)


def test_gpt_forward_and_generate():
    model = GPT(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=2, dropout=0.0)
    idx = torch.randint(20, (3, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 8, 20) and loss.item() > 0
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)  # block_size超過でも動く
