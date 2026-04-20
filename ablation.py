"""
Ablation study for HybridKV.

1. SnapKV+Sink: all layers use SnapKV voting + sink preservation (shallow_layers=[])
2. Layer partitioning: vary which layers use recency vs voting

Usage:
    python ablation.py --device cpu
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from baseline import load_data
from evaluate_decode import evaluate_decode_ppl


def run_ablation(model, tokenizer, device, context_len=1024, continuation_len=512,
                 max_capacity=128, n_sink=4, window_size=32, kernel_size=5):
    """Run ablation experiments on PG-19."""
    dataset = "pg19"
    input_ids = load_data(dataset, tokenizer, max_tokens=context_len + continuation_len)

    common = dict(
        model=model, input_ids=input_ids, context_len=context_len, device=device,
        n_sink=n_sink, window_size=window_size, max_capacity=max_capacity,
        kernel_size=kernel_size,
    )

    results = {}

    # --- Ablation 1: SnapKV+Sink (all layers use voting + sink) ---
    print("\n" + "=" * 60)
    print("Ablation 1: SnapKV+Sink (shallow_layers=[], all layers vote)")
    print("=" * 60)
    r = evaluate_decode_ppl(method="hybridkv", shallow_layers=[], **common)
    print(f"  PPL={r['ppl']:.2f}, KV={r['context_kv_len']}")
    results["snapkv_sink"] = r

    # --- Ablation 2: Layer partitioning ---
    print("\n" + "=" * 60)
    print("Ablation 2: Layer partitioning (vary shallow set)")
    print("=" * 60)

    partitions = {
        "S={}": [],
        "S={0}": [0],
        "S={0,1}": [0, 1],
        "S={0,1,2}": [0, 1, 2],
        "S={0,1,2,3}": [0, 1, 2, 3],
        "S={3,4,5}": [3, 4, 5],
        "S={0,...,5}": [0, 1, 2, 3, 4, 5],
    }

    partition_results = {}
    for name, shallow in partitions.items():
        print(f"\n  [{name}] ", end="")
        r = evaluate_decode_ppl(method="hybridkv", shallow_layers=shallow, **common)
        print(f"PPL={r['ppl']:.2f}, KV={r['context_kv_len']}")
        partition_results[name] = r

    results["partitions"] = partition_results

    # --- Reference baselines ---
    print("\n" + "=" * 60)
    print("Reference baselines")
    print("=" * 60)

    for method in ["dense", "snapkv"]:
        print(f"\n  [{method}] ", end="")
        r = evaluate_decode_ppl(method=method, **common)
        print(f"PPL={r['ppl']:.2f}, KV={r['context_kv_len']}")
        results[method] = r

    return results


def print_ablation_summary(results):
    """Print summary table."""
    print("\n" + "=" * 60)
    print("Ablation Summary (PG-19, ctx=1024, C=128)")
    print("=" * 60)

    print(f"\n{'Configuration':<25} {'PPL':>8} {'KV Size':>8}")
    print("-" * 43)

    # Reference
    print(f"{'Dense (no compression)':<25} {results['dense']['ppl']:>8.2f} {results['dense']['context_kv_len']:>8}")
    print(f"{'SnapKV (original)':<25} {results['snapkv']['ppl']:>8.2f} {results['snapkv']['context_kv_len']:>8}")
    print(f"{'SnapKV+Sink':<25} {results['snapkv_sink']['ppl']:>8.2f} {results['snapkv_sink']['context_kv_len']:>8}")

    print("-" * 43)
    for name, r in results["partitions"].items():
        label = f"HybridKV {name}"
        print(f"{label:<25} {r['ppl']:>8.2f} {r['context_kv_len']:>8}")


def main():
    parser = argparse.ArgumentParser(description="HybridKV ablation study")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--max-capacity", type=int, default=128)
    args = parser.parse_args()

    print("Loading Pythia-70M...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", attn_implementation="eager"
    )

    results = run_ablation(
        model, tokenizer, args.device,
        context_len=args.context_len,
        max_capacity=args.max_capacity,
    )
    print_ablation_summary(results)

    os.makedirs("results", exist_ok=True)
    with open("results/ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/ablation.json")


if __name__ == "__main__":
    main()
