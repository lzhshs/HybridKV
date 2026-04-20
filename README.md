# HybridKV: Layer-Adaptive KV Cache Compression

A unified platform comparing KV cache compression methods for LLM inference on Pythia-70M.

## Methods

| Method | Strategy | KV Structure |
|--------|----------|-------------|
| Dense | No compression (baseline) | Full KV cache |
| StreamingLLM | Attention sink + sliding window | [Sink] + [Window] |
| SnapKV | Attention voting + observation window | [Selected] + [Window] |
| **HybridKV** | Layer-adaptive sink + selection | [Sink] + [Selected] + [Window] |

**HybridKV** combines StreamingLLM's sink token insight with SnapKV's content-based selection:
- **Shallow layers (0-2):** [Sink] + [Recent] + [Window] — shallow attention is local
- **Deep layers (3-5):** [Sink] + [SnapKV voting] + [Window] — deep attention is semantic

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### PPL Evaluation

```bash
# Single run
python evaluate.py --dataset wikitext --max-tokens 8192

# Full comparison (both datasets, multiple capacities)
python evaluate.py --run-all
```

### Speed Benchmark

```bash
python benchmark.py --context-len 1024 --num-generate 128
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## References

- StreamingLLM: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
- SnapKV: "LLM Knows What You are Looking for Before Generation" (NeurIPS 2024)
- PyramidKV: "Dynamic KV Cache Compression based on Pyramidal Information Funneling" (TMLR 2025)
