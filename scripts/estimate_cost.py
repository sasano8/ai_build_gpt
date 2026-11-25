"""LLM学習のリソース・費用の概算(標準ライブラリのみ。torch不要)。

根拠となる式:
    学習計算量   C ≈ 6 · N · D            [FLOPs]  (N=パラメータ数, D=学習トークン数)
    Chinchilla   D ≈ 20 · N               (compute-optimal の目安)
    GPU時間      = C / (GPUの公称FLOPS · MFU) / 3600
    学習メモリ   ≈ 16 byte × N            (bf16重み2 + fp32マスタ4 + Adam 8 + 勾配2。活性化は別)

使い方:
    python scripts/estimate_cost.py --params 7e9                 # D は Chinchilla(20N)
    python scripts/estimate_cost.py --params 7e9 --tokens 1e12   # 過学習気味(over-training)
    python scripts/estimate_cost.py --params 70e9 --gpu h100 --n-gpus 2048 --price 2.5
"""
import argparse

# GPU公称ピーク性能(bf16 Tensor Core, dense) [FLOPS]。実効はMFUで割り引く。
GPU_PEAK_FLOPS = {"h100": 989e12, "a100": 312e12, "rtx4090": 165e12}
# 仮のクラウド単価 [USD/GPU時間]。変動が大きいので --price で必ず上書きして使うこと。
GPU_DEFAULT_PRICE = {"h100": 2.5, "a100": 1.5, "rtx4090": 0.4}

BYTES_PER_PARAM_TRAIN = 16   # AdamW + 混合精度(上記)
BYTES_PER_TOKEN_RAW = 4.5    # 英語テキスト1トークンあたりの生テキスト量の目安
BYTES_PER_TOKEN_STORED = 2   # uint16(語彙6.5万未満)で保存した場合


def human(x: float) -> str:
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:.3g}{unit}"
    return f"{x:.3g}"


def estimate(params: float, tokens: float, gpu: str, mfu: float, price: float, n_gpus: int) -> dict:
    flops = 6 * params * tokens
    gpu_hours = flops / (GPU_PEAK_FLOPS[gpu] * mfu) / 3600
    return {
        "flops": flops,
        "tokens_per_param": tokens / params,
        "gpu_hours": gpu_hours,
        "cost_usd": gpu_hours * price,
        "days": gpu_hours / n_gpus / 24,
        "train_mem_gb": params * BYTES_PER_PARAM_TRAIN / 1e9,
        "raw_text_tb": tokens * BYTES_PER_TOKEN_RAW / 1e12,
        "stored_tb": tokens * BYTES_PER_TOKEN_STORED / 1e12,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", type=float, required=True, help="パラメータ数 N (例: 7e9)")
    p.add_argument("--tokens", type=float, help="学習トークン数 D (省略時は Chinchilla の 20N)")
    p.add_argument("--gpu", choices=list(GPU_PEAK_FLOPS), default="h100")
    p.add_argument("--mfu", type=float, default=0.40, help="Model FLOPs Utilization (大規模学習で0.3〜0.5程度)")
    p.add_argument("--price", type=float, help="USD/GPU時間 (省略時は仮の既定値)")
    p.add_argument("--n-gpus", type=int, default=1, help="使うGPU枚数(所要日数の計算用)")
    a = p.parse_args()

    tokens = a.tokens or 20 * a.params
    price = a.price if a.price is not None else GPU_DEFAULT_PRICE[a.gpu]
    r = estimate(a.params, tokens, a.gpu, a.mfu, price, a.n_gpus)

    print(f"N={human(a.params)} params | D={human(tokens)} tokens ({r['tokens_per_param']:.3g} tokens/param)")
    print(f"  計算量          : {r['flops']:.2e} FLOPs")
    print(f"  GPU時間         : {human(r['gpu_hours'])} h  ({a.gpu}, MFU={a.mfu})")
    print(f"  所要日数        : {r['days']:.3g} 日  ({a.n_gpus} GPU)")
    print(f"  GPU費用(概算)   : ${human(r['cost_usd'])}  (単価 ${price}/h)")
    print(f"  学習メモリ(状態): {r['train_mem_gb']:.3g} GB  (活性化・通信バッファは別)")
    print(f"  データ量        : 生テキスト約{r['raw_text_tb']:.3g} TB / トークン化後約{r['stored_tb']:.3g} TB")


if __name__ == "__main__":
    main()
