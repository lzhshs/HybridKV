"""
Unified evaluation: compare Dense, StreamingLLM, SnapKV, and HybridKV on PPL.

Usage:
    python evaluate.py --dataset wikitext --max-tokens 8192
    python evaluate.py --dataset pg19 --max-tokens 8192
    python evaluate.py --run-all
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from baseline import load_data, evaluate_ppl
from snapkv import evaluate_ppl_snapkv
from streaming_llm import evaluate_ppl_streaming
from hybridkv import evaluate_ppl_hybridkv


def run_experiment(
    model, tokenizer, dataset_name, max_tokens, stride, device,
    max_capacity, window_size, kernel_size, n_sink, shallow_layers,
):
    """Run all 4 methods on the same data."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Max tokens: {max_tokens}")
    print(f"{'='*60}")

    input_ids = load_data(dataset_name, tokenizer, max_tokens=max_tokens)

    # Dense baseline
    print("\n--- Dense Baseline ---")
    dense = evaluate_ppl(model, input_ids, stride=stride, device=device)
    print(f"  PPL: {dense['ppl']:.2f} | Time: {dense['time_seconds']}s")

    # StreamingLLM
    print(f"\n--- StreamingLLM (sink={n_sink}, window={window_size}) ---")
    streaming = evaluate_ppl_streaming(
        model, input_ids, n_sink=n_sink, window_size=window_size,
        stride=stride, device=device,
    )
    print(f"  PPL: {streaming['ppl']:.2f} | Time: {streaming['time_seconds']}s")

    # SnapKV
    print(f"\n--- SnapKV (capacity={max_capacity}, window={window_size}) ---")
    snapkv = evaluate_ppl_snapkv(
        model, input_ids, window_size=window_size, max_capacity=max_capacity,
        kernel_size=kernel_size, stride=stride, device=device,
    )
    print(f"  PPL: {snapkv['ppl']:.2f} | Time: {snapkv['time_seconds']}s")

    # HybridKV
    print(f"\n--- HybridKV (sink={n_sink}, capacity={max_capacity}, window={window_size}) ---")
    hybrid = evaluate_ppl_hybridkv(
        model, input_ids, n_sink=n_sink, window_size=window_size,
        max_capacity=max_capacity, kernel_size=kernel_size,
        shallow_layers=shallow_layers, stride=stride, device=device,
    )
    print(f"  PPL: {hybrid['ppl']:.2f} | Time: {hybrid['time_seconds']}s")

    return {
        "dataset": dataset_name,
        "max_tokens": max_tokens,
        "dense": dense,
        "streaming_llm": streaming,
        "snapkv": snapkv,
        "hybridkv": hybrid,
        "config": {
            "n_sink": n_sink,
            "max_capacity": max_capacity,
            "window_size": window_size,
            "kernel_size": kernel_size,
            "shallow_layers": shallow_layers,
        },
    }


def print_comparison(result):
    """Print a comparison table."""
    d = result["dense"]
    s = result["streaming_llm"]
    k = result["snapkv"]
    h = result["hybridkv"]

    print(f"\n{'='*70}")
    print(f"Summary: {result['dataset']} ({result['max_tokens']} tokens)")
    print(f"{'='*70}")

    print(f"\n{'Method':<25} {'PPL':>8} {'Time(s)':>10} {'Throughput':>12} {'KV Budget':>10}")
    print(f"{'-'*65}")
    print(f"{'Dense':<25} {d['ppl']:>8.2f} {d['time_seconds']:>10.2f} "
          f"{d['tokens_per_second']:>10.1f}/s {'full':>10}")
    print(f"{'StreamingLLM':<25} {s['ppl']:>8.2f} {s['time_seconds']:>10.2f} "
          f"{s['tokens_per_second']:>10.1f}/s {s['kv_budget']:>10}")
    print(f"{'SnapKV':<25} {k['ppl']:>8.2f} {k['time_seconds']:>10.2f} "
          f"{k['tokens_per_second']:>10.1f}/s {k['kv_budget']:>10}")
    print(f"{'HybridKV':<25} {h['ppl']:>8.2f} {h['time_seconds']:>10.2f} "
          f"{h['tokens_per_second']:>10.1f}/s {h['kv_budget']:>10}")


def main():
    parser = argparse.ArgumentParser(description="Unified 4-method PPL evaluation")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max-capacity", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--run-all", action="store_true",
                        help="Run on both datasets with multiple capacities")
    args = parser.parse_args()

    shallow_layers = [0, 1, 2]

    print("Loading Pythia-70M...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", attn_implementation="eager"
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    os.makedirs("results", exist_ok=True)

    if args.run_all:
        all_results = []
        configs = [("wikitext", 16384), ("pg19", 16384)]
        capacities = [64, 128, 256]

        for dataset, max_tokens in configs:
            for cap in capacities:
                result = run_experiment(
                    model, tokenizer, dataset, max_tokens, args.stride, args.device,
                    max_capacity=cap, window_size=args.window_size,
                    kernel_size=args.kernel_size, n_sink=args.n_sink,
                    shallow_layers=shallow_layers,
                )
                print_comparison(result)
                all_results.append(result)

        with open("results/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to results/all_results.json")
    else:
        result = run_experiment(
            model, tokenizer, args.dataset, args.max_tokens, args.stride, args.device,
            max_capacity=args.max_capacity, window_size=args.window_size,
            kernel_size=args.kernel_size, n_sink=args.n_sink,
            shallow_layers=shallow_layers,
        )
        print_comparison(result)

        with open(f"results/{args.dataset}_result.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
