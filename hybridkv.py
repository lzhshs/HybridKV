"""
HybridKV: Layer-adaptive KV Cache compression.

Combines StreamingLLM's attention sink insight with SnapKV's content-based selection,
applying different strategies based on layer depth:
  - Shallow layers (0-2): [Sink] + [Recent] + [Window]  (local/positional bias)
  - Deep layers (3-5):    [Sink] + [SnapKV] + [Window]  (semantic/content bias)

Every layer preserves:
  - Sink tokens (first n_sink positions) for attention stability
  - Window tokens (last window_size positions) as observation context
"""

import time
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, DynamicCache
from typing import Optional
from tqdm import tqdm


def hybrid_kv_compress(
    layer_idx: int,
    attention_scores: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    n_sink: int = 4,
    window_size: int = 32,
    max_capacity: int = 128,
    kernel_size: int = 5,
    shallow_layers: list[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress KV cache using layer-adaptive strategy.

    Output structure: [Sink (n_sink)] + [Selected (budget)] + [Window (window_size)]
    Total output length: max_capacity + window_size
    """
    if shallow_layers is None:
        shallow_layers = [0, 1, 2]

    batch_size, num_heads, seq_len, head_dim = key_states.shape

    if seq_len <= window_size + max_capacity:
        return key_states, value_states

    prefix_len = seq_len - window_size
    budget = max_capacity - n_sink

    # Sink: first n_sink tokens
    sink_keys = key_states[:, :, :n_sink, :]
    sink_values = value_states[:, :, :n_sink, :]

    # Window: last window_size tokens
    window_keys = key_states[:, :, -window_size:, :]
    window_values = value_states[:, :, -window_size:, :]

    # Middle: between sink and window
    middle_keys = key_states[:, :, n_sink:prefix_len, :]
    middle_values = value_states[:, :, n_sink:prefix_len, :]
    middle_len = middle_keys.shape[2]

    if middle_len <= budget:
        selected_keys = middle_keys
        selected_values = middle_values
    elif layer_idx in shallow_layers:
        # Shallow layer: Recent strategy
        selected_keys = middle_keys[:, :, -budget:, :]
        selected_values = middle_values[:, :, -budget:, :]
    else:
        # Deep layer: SnapKV strategy — attention voting on middle region
        obs_attn = attention_scores[:, :, -window_size:, n_sink:prefix_len]

        obs_flat = obs_attn.reshape(-1, window_size, middle_len)
        padding = kernel_size // 2
        pooled = F.max_pool1d(obs_flat, kernel_size=kernel_size, stride=1, padding=padding)
        pooled = pooled.reshape(batch_size, num_heads, window_size, middle_len)

        vote_scores = pooled.sum(dim=2)

        _, top_indices = vote_scores.topk(budget, dim=-1)
        top_indices = top_indices.sort(dim=-1).values

        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        selected_keys = torch.gather(middle_keys, dim=2, index=idx_expanded)
        selected_values = torch.gather(middle_values, dim=2, index=idx_expanded)

    # Concatenate: [Sink] + [Selected] + [Window]
    compressed_keys = torch.cat([sink_keys, selected_keys, window_keys], dim=2)
    compressed_values = torch.cat([sink_values, selected_values, window_values], dim=2)

    return compressed_keys, compressed_values


class HybridKVWrapper:
    """Wraps a GPTNeoX model to apply HybridKV compression after prefill."""

    def __init__(self, model: GPTNeoXForCausalLM,
                 n_sink: int = 4, window_size: int = 32,
                 max_capacity: int = 128, kernel_size: int = 5,
                 shallow_layers: Optional[list[int]] = None):
        self.model = model
        self.n_sink = n_sink
        self.window_size = window_size
        self.max_capacity = max_capacity
        self.kernel_size = kernel_size
        self.shallow_layers = shallow_layers if shallow_layers is not None else [0, 1, 2]
        self._attn_scores: dict[int, torch.Tensor] = {}
        self._hooks = []

    def _register_hooks(self):
        self._remove_hooks()
        self._attn_scores.clear()

        for layer_idx, layer in enumerate(self.model.gpt_neox.layers):
            attn_module = layer.attention

            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            self._attn_scores[idx] = attn_weights.detach()
                return hook_fn

            handle = attn_module.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compress_kv_cache(self, past_key_values):
        new_cache = DynamicCache()

        for layer_idx in range(len(past_key_values.layers)):
            layer = past_key_values.layers[layer_idx]
            key, value = layer.keys, layer.values

            attn = self._attn_scores.get(layer_idx)
            if attn is None:
                attn = torch.zeros(
                    key.shape[0], key.shape[1], key.shape[2], key.shape[2],
                    device=key.device,
                )

            new_key, new_value = hybrid_kv_compress(
                layer_idx=layer_idx,
                attention_scores=attn,
                key_states=key,
                value_states=value,
                n_sink=self.n_sink,
                window_size=self.window_size,
                max_capacity=self.max_capacity,
                kernel_size=self.kernel_size,
                shallow_layers=self.shallow_layers,
            )
            new_cache.update(new_key, new_value, layer_idx)

        return new_cache

    @torch.no_grad()
    def prefill_and_compress(self, input_ids: torch.Tensor):
        self._register_hooks()

        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=True,
        )
        self._remove_hooks()

        if not self._attn_scores and outputs.attentions is not None:
            for idx, attn in enumerate(outputs.attentions):
                if attn is not None:
                    self._attn_scores[idx] = attn.detach()

        compressed = self.compress_kv_cache(outputs.past_key_values)

        original_len = outputs.past_key_values.layers[0].keys.shape[2]
        compressed_len = compressed.layers[0].keys.shape[2]
        print(f"  KV Cache: {original_len} -> {compressed_len} tokens "
              f"({compressed_len/original_len*100:.1f}% retained)")

        return outputs, compressed


def evaluate_ppl_hybridkv(
    model: GPTNeoXForCausalLM, input_ids: torch.Tensor,
    n_sink: int = 4, window_size: int = 32,
    max_capacity: int = 128, kernel_size: int = 5,
    shallow_layers: Optional[list[int]] = None,
    stride: int = 512, device: str = "cpu",
) -> dict:
    """Sliding-window PPL evaluation with HybridKV compression."""
    model.eval()
    model.to(device)
    max_length = model.config.max_position_embeddings
    seq_len = input_ids.shape[1]
    kv_budget = max_capacity + window_size

    if shallow_layers is None:
        shallow_layers = [0, 1, 2]

    wrapper = HybridKVWrapper(
        model, n_sink=n_sink, window_size=window_size,
        max_capacity=max_capacity, kernel_size=kernel_size,
        shallow_layers=shallow_layers,
    )

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0
    start_time = time.time()
    first_window = True

    for begin in tqdm(range(0, seq_len, stride), desc="Evaluating PPL (HybridKV)"):
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
            wrapper._register_hooks()
            with torch.no_grad():
                outputs = model(input_ids=ids, use_cache=True, output_attentions=True)
            wrapper._remove_hooks()

            if not wrapper._attn_scores and outputs.attentions is not None:
                for idx, attn in enumerate(outputs.attentions):
                    if attn is not None:
                        wrapper._attn_scores[idx] = attn.detach()

            compressed = wrapper.compress_kv_cache(outputs.past_key_values)
            wrapper._attn_scores.clear()

            if first_window:
                orig = outputs.past_key_values.layers[0].keys.shape[2]
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
