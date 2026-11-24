"""学習済みGPTと対話するCLI。

このモデルは「続きを書く」だけの事前学習モデル(ChatGPTのような指示追従はしない)。
入力した文章を書き出しとして、その続きを生成する。

使い方:
    python -m gpt.chat --preset small
    (入力例)  ROMEO:
"""
import argparse

import torch

from gpt.config import PRESETS
from gpt.data import CharTokenizer, load_text
from gpt.train import DATA_PATH, OUT_DIR, build_model, pick_device

HELP = "コマンド: /temp <数値> 温度を変更 | /len <数値> 生成文字数を変更 | /quit 終了"


def load_model(preset: str, checkpoint: str, tokenizer: CharTokenizer, device: str):
    cfg = PRESETS[preset]
    model = build_model("gpt", tokenizer.vocab_size, cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


def encode_prompt(tokenizer: CharTokenizer, text: str, device: str) -> torch.Tensor:
    """学習データに無い文字は取り除いてトークン化する。空なら改行(id 0)から始める。"""
    known = [c for c in text if c in tokenizer.chars]
    if len(known) < len(text):
        print("(学習データに無い文字は無視しました)")
    ids = tokenizer.encode("".join(known)) or [0]
    return torch.tensor([ids], dtype=torch.long, device=device)


def complete(model, tokenizer, prompt: str, length: int, temperature: float, device: str) -> str:
    idx = encode_prompt(tokenizer, prompt, device)
    out = model.generate(idx, max_new_tokens=length, temperature=temperature)[0].tolist()
    return tokenizer.decode(out[idx.shape[1]:])  # 入力部分を除いた続きだけ返す


def handle_command(line: str, settings: dict) -> bool:
    """`/` で始まる入力を処理する。終了なら False。"""
    cmd, *rest = line.split()
    if cmd == "/quit":
        return False
    try:
        if cmd == "/temp":
            settings["temperature"] = float(rest[0])
        elif cmd == "/len":
            settings["length"] = int(rest[0])
        else:
            print(HELP)
    except (IndexError, ValueError):
        print(HELP)
    return True


def repl(model, tokenizer, device: str, settings: dict) -> None:
    print("GPT CLI -- 書き出しを入力すると続きを生成します。")
    print(HELP)
    while True:
        try:
            line = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.startswith("/"):
            if not handle_command(line, settings):
                break
            continue
        text = complete(model, tokenizer, line.replace("\\n", "\n"),
                        settings["length"], settings["temperature"], device)
        print(line + text)


def main():
    p = argparse.ArgumentParser(description="学習済みGPTと対話する")
    p.add_argument("--preset", choices=["tiny", "small", "full"], default="small",
                   help="学習時と同じプリセットを指定する")
    p.add_argument("--checkpoint", default=str(OUT_DIR / "gpt.pt"))
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    device = pick_device()
    tokenizer = CharTokenizer(load_text(DATA_PATH))
    model = load_model(args.preset, args.checkpoint, tokenizer, device)
    repl(model, tokenizer, device, {"length": args.length, "temperature": args.temperature})


if __name__ == "__main__":
    main()
