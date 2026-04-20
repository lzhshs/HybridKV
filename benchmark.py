"""
Speed benchmark: TTFT, TPOT, and throughput for Dense, StreamingLLM, SnapKV, HybridKV.
"""

import argparse
import json
import os
import time
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, DynamicCache

from baseline import load_data
from snapkv import SnapKVWrapper
from streaming_llm import StreamingLLMWrapper
from hybridkv import HybridKVWrapper


def benchmark_generation(
    model, input_ids, num_generate: int = 128, device: str = "cpu",
    method: str = "dense",
    window_size: int = 32, max_capacity: int = 128, kernel_size: int = 5,
    n_sink: int = 4, shallow_layers: list[int] = None,
    warmup: int = 1,
) -> dict:
    """Measure TTFT and TPOT for a prefill + decode run."""
    model.eval()
    model.to(device)
    context = input_ids.to(device)

    if shallow_layers is None:
        shallow_layers = [0, 1, 2]

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(context[:, :64], use_cache=False)

    # Prefill (TTFT)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        if method == "snapkv":
            wrapper = SnapKVWrapper(model, window_size, max_capacity, kernel_size)
            outputs, past = wrapper.prefill_and_compress(context)
        elif method == "streaming_llm":
            wrapper = StreamingLLMWrapper(model, n_sink, window_size)
            outputs, past = wrapper.prefill_and_compress(context)
        elif method == "hybridkv":
            wrapper = HybridKVWrapper(
                model, n_sink, window_size, max_capacity, kernel_size, shallow_layers
            )
            outputs, past = wrapper.prefill_and_compress(context)
        else:
            outputs = model(input_ids=context, use_cache=True)
            past = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    if device == "mps":
        torch.mps.synchronize()
    ttft = time.perf_counter() - t0

    # Decode (TPOT)
    generated_tokens = [next_token]
    needs_position_ids = method in ("snapkv", "hybridkv", "streaming_llm")
    pos_offset = context.shape[1] if needs_position_ids else None

    if device == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()

    for i in range(num_generate - 1):
        with torch.no_grad():
            if pos_offset is not None:
                position_ids = torch.tensor([[pos_offset + i]], device=device)
                out = model(
                    input_ids=next_token, past_key_values=past,
                    position_ids=position_ids, use_cache=True,
                )
            else:
                out = model(
                    input_ids=next_token, past_key_values=past, use_cache=True,
                )
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token)

    if device == "mps":
        torch.mps.synchronize()
    decode_time = time.perf_counter() - t1

    num_decoded = num_generate - 1
    tpot = (decode_time / num_decoded * 1000) if num_decoded > 0 else 0
    throughput = num_decoded / decode_time if decode_time > 0 else 0

    if isinstance(past, DynamicCache):
        kv_len = past.layers[0].keys.shape[2]
    else:
        kv_len = past[0][0].shape[2]

    return {
        "method": method,
        "context_len": context.shape[1],
        "num_generated": num_generate,
        "ttft_ms": round(ttft * 1000, 2),
        "tpot_ms": round(tpot, 2),
        "throughput_tok_s": round(throughput, 1),
        "decode_time_s": round(decode_time, 3),
        "kv_cache_len": kv_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Speed benchmark: 4 methods")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--num-generate", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max-capacity", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-sink", type=int, default=4)
    args = parser.parse_args()

    shallow_layers = [0, 1, 2]

    print("Loading Pythia-70M...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", attn_implementation="eager"
    )

    input_ids = load_data(args.dataset, tokenizer, max_tokens=args.context_len + 256)
    context = input_ids[:, :args.context_len]
    print(f"Context: {context.shape[1]} tokens, Generate: {args.num_generate} tokens\n")

    common = dict(
        window_size=args.window_size, max_capacity=args.max_capacity,
        kernel_size=args.kernel_size, n_sink=args.n_sink,
        shallow_layers=shallow_layers,
    )

    # Run all 4 methods
    print("Benchmarking Dense...")
    dense = benchmark_generation(model, context, args.num_generate, args.device, "dense", **common)

    print("Benchmarking StreamingLLM...")
    streaming = benchmark_generation(model, context, args.num_generate, args.device, "streaming_llm", **common)

    print("Benchmarking SnapKV...")
    snapkv = benchmark_generation(model, context, args.num_generate, args.device, "snapkv", **common)

    print("Benchmarking HybridKV...")
    hybrid = benchmark_generation(model, context, args.num_generate, args.device, "hybridkv", **common)

    # Print results
    results = [dense, streaming, snapkv, hybrid]
    print(f"\n{'='*75}")
    print(f"Speed Benchmark (context={args.context_len}, generate={args.num_generate})")
    print(f"{'='*75}")
    print(f"{'Method':<25} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'Throughput':>12} {'KV Size':>10}")
    print(f"{'-'*67}")
    for r in results:
        print(f"{r['method']:<25} {r['ttft_ms']:>10.1f} {r['tpot_ms']:>10.2f} "
              f"{r['throughput_tok_s']:>10.1f}/s {r['kv_cache_len']:>10}")

    os.makedirs("results", exist_ok=True)
    out = {"results": results, "args": vars(args)}
    with open("results/benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to results/benchmark.json")


if __name__ == "__main__":
    main()
