"""Phase 3/6: 学習・評価・生成のエントリポイント。

使い方:
    python -m gpt.train --model gpt --preset small
    python -m gpt.train --model bigram
    python -m gpt.train --generate-only --checkpoint out/gpt-small.pt
"""
import argparse
from pathlib import Path

import torch

from gpt.bigram import BigramLanguageModel
from gpt.config import PRESETS, Config
from gpt.data import CharTokenizer, get_batch, load_text, split_train_val
from gpt.model import GPT

DATA_PATH = "data/input.txt"
OUT_DIR = Path("out")


def checkpoint_path(model: str, preset: str) -> Path:
    """モデルとプリセットごとに別ファイルへ保存する(Makefileの依存関係と対応)。"""
    return OUT_DIR / f"{model}-{preset}.pt"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(kind: str, vocab_size: int, cfg: Config) -> torch.nn.Module:
    if kind == "bigram":
        return BigramLanguageModel(vocab_size)
    return GPT(vocab_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.dropout)


@torch.no_grad()
def estimate_loss(model, splits: dict, cfg: Config, device: str) -> dict:
    """train/val それぞれ複数バッチの平均lossを返す。"""
    model.eval()
    result = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, cfg.block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        result[name] = losses.mean().item()
    model.train()
    return result


def train(model, splits: dict, cfg: Config, device: str) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, cfg, device)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        x, y = get_batch(splits["train"], cfg.batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def sample_text(model, tokenizer: CharTokenizer, n_tokens: int, device: str) -> str:
    start = torch.zeros((1, 1), dtype=torch.long, device=device)  # 先頭は改行(id 0)
    model.eval()
    ids = model.generate(start, max_new_tokens=n_tokens)[0].tolist()
    return tokenizer.decode(ids)


def parse_args():
    p = argparse.ArgumentParser(description="Tiny GPT trainer")
    p.add_argument("--model", choices=["bigram", "gpt"], default="gpt")
    p.add_argument("--preset", choices=list(PRESETS), default="small")
    p.add_argument("--max-iters", type=int, help="presetのmax_itersを上書き")
    p.add_argument("--gen-tokens", type=int, default=500, help="生成するトークン数")
    p.add_argument("--generate-only", action="store_true", help="学習せず保存済みモデルで生成")
    p.add_argument("--checkpoint", help="読み込み/保存先 (既定: out/<model>-<preset>.pt)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PRESETS[args.preset]
    if args.max_iters is not None:
        cfg = Config(**{**cfg.__dict__, "max_iters": args.max_iters})
    torch.manual_seed(cfg.seed)
    device = pick_device()
    ckpt_path = Path(args.checkpoint or checkpoint_path(args.model, args.preset))

    text = load_text(DATA_PATH)
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_train_val(data)
    print(f"device={device} | model={args.model} | preset={args.preset} | vocab={tokenizer.vocab_size}")

    model = build_model(args.model, tokenizer.vocab_size, cfg).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.generate_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        train(model, {"train": train_data, "val": val_data}, cfg, device)
        ckpt_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved: {ckpt_path}")

    print("\n===== generated text =====")
    print(sample_text(model, tokenizer, args.gen_tokens, device))


if __name__ == "__main__":
    main()
