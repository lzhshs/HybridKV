"""
StreamingLLM: KV Cache compression via attention sink + sliding window.

Reference: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)

Core idea:
  1. The first few tokens act as "attention sinks" that stabilize attention
  2. Keep [sink tokens] + [recent window] and discard everything in between
  3. No attention scores needed — purely positional strategy
"""

import time
import torch
from transformers import GPTNeoXForCausalLM, DynamicCache
from tqdm import tqdm


def streaming_llm_compress(
    past_key_values: DynamicCache,
    n_sink: int = 4,
    window_size: int = 32,
) -> DynamicCache:
    """Compress KV cache to [sink] + [recent window]."""
    max_cache_length = n_sink + window_size
    num_layers = len(past_key_values.layers)

    if num_layers == 0:
        return past_key_values

    seq_len = past_key_values.layers[0].keys.shape[2]
    if seq_len <= max_cache_length:
        return past_key_values

    new_cache = DynamicCache()
    for layer_idx in range(num_layers):
        layer = past_key_values.layers[layer_idx]
        k, v = layer.keys, layer.values

        sink_k = k[:, :, :n_sink, :]
        sink_v = v[:, :, :n_sink, :]
        recent_k = k[:, :, -window_size:, :]
        recent_v = v[:, :, -window_size:, :]

        new_k = torch.cat([sink_k, recent_k], dim=2)
        new_v = torch.cat([sink_v, recent_v], dim=2)
        new_cache.update(new_k, new_v, layer_idx)

    return new_cache


class StreamingLLMWrapper:
    """Wraps a GPTNeoX model to apply StreamingLLM compression after prefill."""

    def __init__(self, model: GPTNeoXForCausalLM,
                 n_sink: int = 4, window_size: int = 32):
        self.model = model
        self.n_sink = n_sink
        self.window_size = window_size

    @torch.no_grad()
    def prefill_and_compress(self, input_ids: torch.Tensor):
        """Run prefill, then compress KV cache to [sink] + [recent window]."""
        outputs = self.model(input_ids=input_ids, use_cache=True)
        past = outputs.past_key_values

        compressed = streaming_llm_compress(past, self.n_sink, self.window_size)

        original_len = past.layers[0].keys.shape[2]
        compressed_len = compressed.layers[0].keys.shape[2]
        print(f"  KV Cache: {original_len} -> {compressed_len} tokens "
              f"({compressed_len/original_len*100:.1f}% retained)")

        return outputs, compressed


def evaluate_ppl_streaming(
    model: GPTNeoXForCausalLM, input_ids: torch.Tensor,
    n_sink: int = 4, window_size: int = 32,
    stride: int = 512, device: str = "cpu",
) -> dict:
    """Sliding-window PPL evaluation with StreamingLLM compression.

    Same evaluation protocol as snapkv: uses prefill logits for loss computation,
    then compresses KV cache. This ensures fair comparison.
    """
    model.eval()
    model.to(device)
    max_length = model.config.max_position_embeddings
    seq_len = input_ids.shape[1]
    kv_budget = n_sink + window_size

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0
    start_time = time.time()
    first_window = True

    for begin in tqdm(range(0, seq_len, stride), desc="Evaluating PPL (StreamingLLM)"):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        window_len = ids.shape[1]

        if window_len <= kv_budget:
            target_ids = ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids=ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        else:
            with torch.no_grad():
                outputs = model(input_ids=ids, use_cache=True)

            past = outputs.past_key_values
            compressed = streaming_llm_compress(past, n_sink, window_size)

            if first_window:
                orig = past.layers[0].keys.shape[2]
                comp = compressed.layers[0].keys.shape[2]
                print(f"\n  KV Cache: {orig} -> {comp} tokens "
                      f"({comp/orig*100:.1f}% retained)")
                first_window = False

            logits = outputs.logits
            target_ids = ids.clone()
            target_ids[:, :-trg_len] = -100

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            mask = shift_labels != -100
            if mask.sum() > 0:
                neg_log_likelihood = loss_fct(shift_logits[mask], shift_labels[mask])
            else:
                prev_end = end
                continue

        num_valid = (target_ids[:, 1:] != -100).sum().item() if window_len > kv_budget else (target_ids != -100).sum().item()
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
        "kv_budget": kv_budget,
    }
