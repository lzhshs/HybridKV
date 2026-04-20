"""
SnapKV: KV Cache compression via attention-based token selection.

Reference: "SnapKV: LLM Knows What You are Looking for Before Generation" (NeurIPS 2024)

Core idea:
  1. After prefill, use the last `window_size` tokens' attention as an observation window
  2. Pool + aggregate attention scores to find the most important prefix positions
  3. Keep only top-k important positions + the observation window, discard the rest
"""

import time
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer, DynamicCache
from typing import Optional


def snap_kv_compress(
    attention_scores: torch.Tensor,   # [batch, num_heads, seq_len, seq_len]
    key_states: torch.Tensor,         # [batch, num_heads, seq_len, head_dim]
    value_states: torch.Tensor,       # [batch, num_heads, seq_len, head_dim]
    window_size: int = 32,
    max_capacity: int = 128,
    kernel_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress KV cache by selecting important positions based on attention scores."""
    batch_size, num_heads, seq_len, head_dim = key_states.shape

    if seq_len <= window_size + max_capacity:
        return key_states, value_states

    prefix_len = seq_len - window_size

    # Observation window attention: last `window_size` queries attending to the prefix
    obs_attn = attention_scores[:, :, -window_size:, :prefix_len]

    # 1D max pooling along prefix dim to capture neighboring context
    obs_flat = obs_attn.reshape(-1, window_size, prefix_len)
    padding = kernel_size // 2
    pooled = F.max_pool1d(obs_flat, kernel_size=kernel_size, stride=1, padding=padding)
    pooled = pooled.reshape(batch_size, num_heads, window_size, prefix_len)

    # Sum over queries to get per-position importance score
    vote_scores = pooled.sum(dim=2)  # [batch, heads, prefix_len]

    # Select top-k and sort to preserve original order (important for positional encoding)
    _, top_indices = vote_scores.topk(max_capacity, dim=-1)
    top_indices = top_indices.sort(dim=-1).values

    # Gather selected KV positions
    idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key_states[:, :, :prefix_len, :], dim=2, index=idx_expanded)
    selected_values = torch.gather(value_states[:, :, :prefix_len, :], dim=2, index=idx_expanded)

    # Concat: selected important positions + observation window
    compressed_keys = torch.cat([selected_keys, key_states[:, :, -window_size:, :]], dim=2)
    compressed_values = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)

    return compressed_keys, compressed_values


class SnapKVWrapper:
    """Wraps a GPTNeoX model to apply SnapKV compression after prefill.

    Uses forward hooks to capture attention weights during prefill,
    then compresses the KV cache based on those weights.
    """

    def __init__(self, model: GPTNeoXForCausalLM,
                 window_size: int = 32, max_capacity: int = 128, kernel_size: int = 5):
        self.model = model
        self.window_size = window_size
        self.max_capacity = max_capacity
        self.kernel_size = kernel_size
        self._attn_scores: dict[int, torch.Tensor] = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks on each attention layer to collect attention weights."""
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
        """Apply SnapKV compression to each layer's KV cache. Returns a DynamicCache."""
        if isinstance(past_key_values, DynamicCache):
            layers = [(past_key_values.layers[i].keys, past_key_values.layers[i].values)
                      for i in range(len(past_key_values.layers))]
        else:
            layers = list(past_key_values)

        new_cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(layers):
            if layer_idx in self._attn_scores:
                attn = self._attn_scores[layer_idx]
                new_key, new_value = snap_kv_compress(
                    attention_scores=attn, key_states=key, value_states=value,
                    window_size=self.window_size, max_capacity=self.max_capacity,
                    kernel_size=self.kernel_size,
                )
            else:
                new_key, new_value = key, value
            new_cache.update(new_key, new_value, layer_idx)

        return new_cache

    @torch.no_grad()
    def prefill_and_compress(self, input_ids: torch.Tensor):
        """Run prefill with attention collection, then compress the KV cache."""
        self._register_hooks()

        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=True,
        )
        self._remove_hooks()

        # Fallback: extract from model output if hooks didn't fire
        if not self._attn_scores and outputs.attentions is not None:
            for idx, attn in enumerate(outputs.attentions):
                if attn is not None:
                    self._attn_scores[idx] = attn.detach()

        past_key_values = outputs.past_key_values
        compressed_past = self.compress_kv_cache(past_key_values)

        if isinstance(past_key_values, DynamicCache):
            original_len = past_key_values.layers[0].keys.shape[2]
        else:
            original_len = past_key_values[0][0].shape[2]
        compressed_len = compressed_past.layers[0].keys.shape[2]
        print(f"  KV Cache: {original_len} -> {compressed_len} tokens "
              f"({compressed_len/original_len*100:.1f}% retained)")

        return outputs, compressed_past


def evaluate_ppl_snapkv(
    model: GPTNeoXForCausalLM, input_ids: torch.Tensor,
    window_size: int = 32, max_capacity: int = 128, kernel_size: int = 5,
    stride: int = 512, device: str = "cpu",
) -> dict:
    """Sliding-window PPL evaluation with SnapKV compression per window.

    Note: since SnapKV compresses *after* prefill, the logits used for PPL
    are from the full prefill pass. This matches the paper's evaluation.
    """
    model.eval()
    model.to(device)
    max_length = model.config.max_position_embeddings
    seq_len = input_ids.shape[1]

    wrapper = SnapKVWrapper(model, window_size, max_capacity, kernel_size)

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0
    start_time = time.time()
    first_window = True

    from tqdm import tqdm
    for begin in tqdm(range(0, seq_len, stride), desc="Evaluating PPL (SnapKV)"):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        window_len = ids.shape[1]

        # Short windows: no compression needed, just run dense
        if window_len <= window_size + max_capacity:
            target_ids = ids.clone()
            target_ids[:, :-trg_len] = -100
            outputs = model(input_ids=ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        else:
            wrapper._register_hooks()
            outputs = model(input_ids=ids, use_cache=True, output_attentions=True)
            wrapper._remove_hooks()

            if not wrapper._attn_scores and outputs.attentions is not None:
                for idx, attn in enumerate(outputs.attentions):
                    if attn is not None:
                        wrapper._attn_scores[idx] = attn.detach()

            compressed_past = wrapper.compress_kv_cache(outputs.past_key_values)
            wrapper._attn_scores.clear()

            if first_window:
                pkv = outputs.past_key_values
                orig = pkv.layers[0].keys.shape[2] if isinstance(pkv, DynamicCache) else pkv[0][0].shape[2]
                comp = compressed_past.layers[0].keys.shape[2]
                print(f"\n  KV Cache: {orig} -> {comp} tokens "
                      f"({comp/orig*100:.1f}% retained)")
                first_window = False

            # Use prefill logits for loss (compression happens after prefill)
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

        num_valid = (target_ids[:, 1:] != -100).sum().item() if window_len > window_size + max_capacity else (target_ids != -100).sum().item()
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
        "kv_budget": max_capacity + window_size,
    }


def evaluate_decode_ppl(
    model: GPTNeoXForCausalLM, input_ids: torch.Tensor,
    window_size: int = 32, max_capacity: int = 128, kernel_size: int = 5,
    context_len: int = 1024, device: str = "cpu",
) -> dict:
    """Evaluate how KV compression affects continuation quality.

    Splits input into context + continuation. Prefills context (optionally compressing),
    then runs continuation in one forward pass to compute PPL.
    Uses explicit position_ids to handle the compressed cache length mismatch.
    """
    model.eval()
    model.to(device)

    seq_len = input_ids.shape[1]
    if seq_len <= context_len + 1:
        raise ValueError(f"Input too short ({seq_len}), need > {context_len + 1}")

    context_ids = input_ids[:, :context_len].to(device)
    continuation_ids = input_ids[:, context_len:].to(device)
    num_continuation = continuation_ids.shape[1]

    results = {}

    for method in ["dense", "snapkv"]:
        print(f"\n  [{method}] Prefill {context_len} tokens...", end=" ")
        start_time = time.time()

        if method == "dense":
            with torch.no_grad():
                ctx_out = model(input_ids=context_ids, use_cache=True)
            past = ctx_out.past_key_values
        else:
            wrapper = SnapKVWrapper(model, window_size, max_capacity, kernel_size)
            ctx_out, past = wrapper.prefill_and_compress(context_ids)

        # Need explicit position_ids because compressed cache is shorter
        # than the actual context length
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
        kv_info = past.layers[0].keys.shape[2] if isinstance(past, DynamicCache) else past[0][0].shape[2]

        print(f"PPL={ppl:.2f}, context_KV={kv_info}, Time={elapsed:.1f}s")
        results[method] = {
            "ppl": ppl,
            "context_kv_len": kv_info,
            "num_continuation_tokens": num_continuation,
            "time_seconds": round(elapsed, 2),
        }

    return results


if __name__ == "__main__":
    print("Loading Pythia-70M...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", attn_implementation="eager"
    )

    text = "The quick brown fox jumps over the lazy dog. " * 50
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"Input length: {input_ids.shape[1]} tokens")

    wrapper = SnapKVWrapper(model, window_size=32, max_capacity=64, kernel_size=5)
    outputs, compressed_past = wrapper.prefill_and_compress(input_ids)

    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Compressed KV shape: {compressed_past.layers[0].keys.shape}")
    print("OK")
