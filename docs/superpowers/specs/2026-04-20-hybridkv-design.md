# HybridKV: Layer-Adaptive KV Cache Compression

## Goal

Build a unified experiment platform that:
1. Integrates Dense baseline, StreamingLLM, SnapKV, and HybridKV (novel)
2. Runs PPL + speed benchmarks (TTFT, TPOT, throughput) across all 4 methods
3. Supports wikitext and pg19 datasets on Pythia-70M

HybridKV is the novel contribution: fuses StreamingLLM's sink tokens with SnapKV's attention-based selection, applying different strategies per layer depth.

## Architecture

### KV Cache Structure (per layer)

```
[Sink] + [Selected] + [Window]
  ^         ^            ^
 n_sink   budget      window_size
```

### Layer-Adaptive Strategy (Pythia-70M = 6 layers)

| Layers | Selected Strategy | Rationale |
|--------|------------------|-----------|
| 0-2 (shallow) | **Recent**: take last `budget` tokens from middle | Shallow attention is local/positional |
| 3-5 (deep) | **SnapKV**: attention voting + pooling top-k | Deep attention is semantic/content-based |

All layers share the same `n_sink`, `window_size`, and total `max_capacity`.

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `n_sink` | 4 | Sink tokens (StreamingLLM default) |
| `window_size` | 32 | Observation/recent window at sequence end |
| `max_capacity` | 128 | Total selected budget (includes sink) |
| `kernel_size` | 5 | SnapKV pooling kernel |
| `shallow_layers` | [0,1,2] | Layers using Recent strategy |

Effective budget for Selected segment: `max_capacity - n_sink` = 124 tokens.

### Compression Flow

```
For each layer:
  prefix = kv_cache[:seq_len - window_size]
  window = kv_cache[-window_size:]
  sink = prefix[:n_sink]
  middle = prefix[n_sink:]
  budget = max_capacity - n_sink

  if layer in shallow_layers:
      selected = middle[-budget:]           # Recent
  else:
      selected = snapkv_vote(attn, middle, budget)  # SnapKV voting

  compressed = cat(sink, selected, window)
```

## File Structure

```
~/HybridKV/
├── baseline.py          # Dense PPL evaluation (from snapkv-reproduce)
├── streaming_llm.py     # StreamingLLM: [Sink]+[Window] compression + PPL eval
├── snapkv.py            # SnapKV: attention voting compression + PPL eval
├── hybridkv.py          # HybridKV: layer-adaptive [Sink]+[Selected]+[Window]
├── evaluate.py          # Unified 4-method comparison (PPL)
├── benchmark.py         # Unified 4-method speed test (TTFT/TPOT/throughput)
├── requirements.txt
├── results/
└── README.md
```

## Unified Interface

Each method provides:

```python
class XxxWrapper:
    def __init__(self, model, **config)
    def prefill_and_compress(self, input_ids) -> (outputs, compressed_cache)

def evaluate_ppl_xxx(model, input_ids, ...) -> dict
    # Returns: {ppl, total_tokens, time_seconds, tokens_per_second, kv_budget}
```

## Key Adaptations

### streaming_llm.py (rewrite from teammate's code)
- Keep core logic: `compress_dynamic_cache` doing `[sink] + [recent]`
- Wrap in `StreamingLLMWrapper` with `prefill_and_compress` interface
- Add sliding-window `evaluate_ppl_streaming` matching snapkv.py's evaluation style
- No `output_attentions` needed (StreamingLLM doesn't use attention weights)

### evaluate.py
- Runs all 4 methods on same data with same sliding-window PPL evaluation
- Also runs continuation/decode PPL for each method
- Outputs comparison table + JSON results

### benchmark.py
- Measures TTFT, TPOT, throughput for all 4 methods
- Same warmup/measurement protocol for fairness

## What's NOT in Scope
- Training or fine-tuning
- Head-level adaptive budgets (Ada-KV style)
- Cross-layer KV sharing
- The NeurIPS paper itself (separate task)
