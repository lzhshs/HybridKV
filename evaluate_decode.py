"""
Decode PPL evaluation: measures how KV compression affects continuation quality.

Unlike sliding-window PPL (which uses prefill logits), this evaluates the ACTUAL
impact of compression by:
  1. Prefilling context and compressing KV cache
  2. Running continuation with the compressed cache
  3. Computing PPL on the continuation tokens

This is where compression methods show real differences.

Usage:
    python evaluate_decode.py --dataset wikitext --context-len 1024
    python evaluate_decode.py --dataset pg19 --context-len 1024
    python evaluate_decode.py --run-all
"""

import argparse
import json
import os
import time
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, DynamicCache

from baseline import load_data
from snapkv import SnapKVWrapper
from streaming_llm import StreamingLLMWrapper, streaming_llm_compress
from hybridkv import HybridKVWrapper


def evaluate_decode_ppl(
    model, input_ids, context_len, device,
    method="dense",
    n_sink=4, window_size=32, max_capacity=128, kernel_size=5,
    shallow_layers=None,
):
    """Evaluate continuation PPL with compressed KV cache.

    Splits input into [context | continuation]. Prefills context, compresses,
    then runs continuation in one forward pass to measure PPL.
    """
    if shallow_layers is None:
        shallow_layers = [0, 1, 2]

    model.eval()
    model.to(device)

    seq_len = input_ids.shape[1]
    if seq_len <= context_len + 10:
        raise ValueError(f"Input too short ({seq_len}), need > {context_len + 10}")

    context_ids = input_ids[:, :context_len].to(device)
    continuation_ids = input_ids[:, context_len:].to(device)
    num_continuation = continuation_ids.shape[1]

    start_time = time.time()

    with torch.no_grad():
        if method == "dense":
            ctx_out = model(input_ids=context_ids, use_cache=True)
            past = ctx_out.past_key_values
        elif method == "streaming_llm":
            wrapper = StreamingLLMWrapper(model, n_sink, window_size)
            ctx_out, past = wrapper.prefill_and_compress(context_ids)
        elif method == "snapkv":
            wrapper = SnapKVWrapper(model, window_size, max_capacity, kernel_size)
            ctx_out, past = wrapper.prefill_and_compress(context_ids)
        elif method == "hybridkv":
            wrapper = HybridKVWrapper(
                model, n_sink, window_size, max_capacity, kernel_size, shallow_layers
            )
            ctx_out, past = wrapper.prefill_and_compress(context_ids)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Explicit position_ids for compressed cache
    position_ids = torch.arange(
        context_len, context_len + num_continuation, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        cont_out = model(
            input_ids=continuation_ids,
            past_key_values=past,
            position_ids=position_ids,
            use_cache=False,
        )

    shift_logits = cont_out.logits[:, :-1, :].contiguous()
    shift_labels = continuation_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )

    elapsed = time.time() - start_time
    ppl = torch.exp(loss).item()

    if isinstance(past, DynamicCache):
        kv_len = past.layers[0].keys.shape[2]
    else:
        kv_len = past[0][0].shape[2]

    return {
        "method": method,
        "ppl": ppl,
        "context_kv_len": kv_len,
        "context_len": context_len,
        "num_continuation_tokens": num_continuation,
        "time_seconds": round(elapsed, 2),
    }


def run_decode_experiment(
    model, tokenizer, dataset_name, context_len, continuation_len, device,
    n_sink, window_size, max_capacity, kernel_size, shallow_layers,
):
    """Run decode PPL for all 4 methods."""
    print(f"\n{'='*70}")
    print(f"Decode PPL: {dataset_name} | context={context_len}, continuation={continuation_len}")
    print(f"{'='*70}")

    input_ids = load_data(dataset_name, tokenizer, max_tokens=context_len + continuation_len)
    actual_cont = input_ids.shape[1] - context_len
    print(f"  Actual continuation tokens: {actual_cont}")

    common = dict(
        model=model, input_ids=input_ids, context_len=context_len, device=device,
        n_sink=n_sink, window_size=window_size, max_capacity=max_capacity,
        kernel_size=kernel_size, shallow_layers=shallow_layers,
    )

    results = {}
    methods = ["dense", "streaming_llm", "snapkv", "hybridkv"]

    for method in methods:
        print(f"\n  [{method}] ", end="")
        r = evaluate_decode_ppl(method=method, **common)
        print(f"PPL={r['ppl']:.2f}, KV={r['context_kv_len']}, Time={r['time_seconds']}s")
        results[method] = r

    return results


def print_decode_comparison(results, dataset, context_len):
    """Print decode PPL comparison table."""
    print(f"\n{'='*70}")
    print(f"Decode PPL Summary: {dataset} (context={context_len})")
    print(f"{'='*70}")
    print(f"\n{'Method':<25} {'Decode PPL':>12} {'KV Size':>10} {'Time(s)':>10}")
    print(f"{'-'*57}")
    for method in ["dense", "streaming_llm", "snapkv", "hybridkv"]:
        r = results[method]
        print(f"{method:<25} {r['ppl']:>12.2f} {r['context_kv_len']:>10} {r['time_seconds']:>10.2f}")

    # Show PPL delta vs dense
    dense_ppl = results["dense"]["ppl"]
    print(f"\n{'Method':<25} {'PPL Delta':>12} {'% Increase':>12}")
    print(f"{'-'*49}")
    for method in ["streaming_llm", "snapkv", "hybridkv"]:
        r = results[method]
        delta = r["ppl"] - dense_ppl
        pct = (delta / dense_ppl) * 100
        print(f"{method:<25} {delta:>+12.2f} {pct:>+11.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Decode PPL evaluation (4 methods)")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--continuation-len", type=int, default=512)
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
        configs = [
            ("wikitext", 512, 512),
            ("wikitext", 1024, 512),
            ("pg19", 512, 512),
            ("pg19", 1024, 512),
        ]
        capacities = [64, 128, 256]

        for dataset, ctx_len, cont_len in configs:
            for cap in capacities:
                results = run_decode_experiment(
                    model, tokenizer, dataset, ctx_len, cont_len, args.device,
                    n_sink=args.n_sink, window_size=args.window_size,
                    max_capacity=cap, kernel_size=args.kernel_size,
                    shallow_layers=shallow_layers,
                )
                print_decode_comparison(results, dataset, ctx_len)
                all_results.append({
                    "dataset": dataset,
                    "context_len": ctx_len,
                    "max_capacity": cap,
                    "results": results,
                })

        with open("results/decode_ppl_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to results/decode_ppl_results.json")
    else:
        results = run_decode_experiment(
            model, tokenizer, args.dataset, args.context_len, args.continuation_len,
            args.device,
            n_sink=args.n_sink, window_size=args.window_size,
            max_capacity=args.max_capacity, kernel_size=args.kernel_size,
            shallow_layers=shallow_layers,
        )
        print_decode_comparison(results, args.dataset, args.context_len)

        with open(f"results/decode_ppl_{args.dataset}.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
