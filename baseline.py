"""
Dense (full KV cache) perplexity evaluation on Pythia-70M.
Serves as the baseline for comparing against SnapKV compression.
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from tqdm import tqdm


def load_data(dataset_name: str, tokenizer, max_tokens: int = 0) -> torch.Tensor:
    """Load dataset and concatenate into a single token sequence."""
    if dataset_name == "wikitext":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    elif dataset_name == "pg19":
        dataset = load_dataset("emozilla/pg19", split="test", streaming=True)
        text = next(iter(dataset))["text"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    if max_tokens > 0:
        input_ids = input_ids[:, :max_tokens]

    print(f"Dataset '{dataset_name}' loaded: {input_ids.shape[1]} tokens")
    return input_ids


def evaluate_ppl(model, input_ids, stride: int = 512, device: str = "cpu") -> dict:
    """Sliding-window perplexity evaluation.

    Uses overlapping windows of size max_position_embeddings, sliding by `stride`.
    Only the newly exposed tokens in each window contribute to the loss.
    """
    model.eval()
    model.to(device)
    max_length = model.config.max_position_embeddings
    seq_len = input_ids.shape[1]

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    start_time = time.time()

    for begin in tqdm(range(0, seq_len, stride), desc="Evaluating PPL"):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        ids = input_ids[:, begin:end].to(device)

        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid = (target_ids != -100).sum().item()
        total_nll += neg_log_likelihood.float().item() * num_valid
        total_tokens += num_valid

        prev_end = end
        if end == seq_len:
            break

    elapsed = time.time() - start_time
    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()

    return {
        "ppl": ppl,
        "total_tokens": total_tokens,
        "time_seconds": round(elapsed, 2),
        "tokens_per_second": round(total_tokens / elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline dense PPL evaluation")
    parser.add_argument("--dataset", choices=["wikitext", "pg19"], default="wikitext")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Max tokens to evaluate (0 = all)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    print("=" * 60)
    print("Baseline Dense Evaluation")
    print("=" * 60)

    print("\nLoading Pythia-70M...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    print(f"\nLoading dataset: {args.dataset}")
    input_ids = load_data(args.dataset, tokenizer, max_tokens=args.max_tokens)

    print(f"\nEvaluating (stride={args.stride}, device={args.device})...")
    results = evaluate_ppl(model, input_ids, stride=args.stride, device=args.device)

    print("\n" + "=" * 60)
    print("Results (Dense Baseline)")
    print("=" * 60)
    print(f"  Dataset:          {args.dataset}")
    print(f"  PPL:              {results['ppl']:.2f}")
    print(f"  Total tokens:     {results['total_tokens']}")
    print(f"  Time:             {results['time_seconds']}s")
    print(f"  Throughput:       {results['tokens_per_second']} tokens/s")


if __name__ == "__main__":
    main()
