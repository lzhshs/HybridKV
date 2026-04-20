# HybridKV: Layer-Adaptive KV Cache Compression

Unified experiment platform for comparing KV cache compression methods on Pythia-70M. Implements and evaluates Dense (baseline), StreamingLLM, SnapKV, and HybridKV.

## Methods

| Method | Strategy | KV Structure |
|--------|----------|-------------|
| Dense | No compression (baseline) | Full KV cache |
| StreamingLLM | Attention sink + sliding window | [Sink] + [Window] |
| SnapKV | Attention voting + observation window | [Selected] + [Window] |
| **HybridKV** | Layer-adaptive sink + selection | [Sink] + [Selected] + [Window] |

**HybridKV** combines StreamingLLM's sink token insight with SnapKV's content-based selection, applying different strategies per layer depth:
- **Shallow layers (0-2):** [Sink] + [Recent] + [Window] — shallow attention is local/positional
- **Deep layers (3-5):** [Sink] + [SnapKV voting] + [Window] — deep attention is semantic

## Setup

```bash
pip install -r requirements.txt
```

Model and datasets are downloaded automatically from HuggingFace on first run.

## Usage

### 1. Decode PPL Evaluation (Recommended)

Measures the **actual impact of compression** on continuation quality. Prefills context, compresses KV cache, then evaluates PPL on continuation tokens:

```bash
# Single config
python evaluate_decode.py --dataset pg19 --context-len 1024

# Full comparison (both datasets x multiple context lengths x multiple capacities)
python evaluate_decode.py --run-all

# Custom parameters
python evaluate_decode.py --dataset pg19 --context-len 1024 --continuation-len 512 --max-capacity 128
```

### 2. Ablation Study

Runs SnapKV+Sink baseline and layer partitioning ablations:

```bash
python ablation.py --device cpu
```

### 3. Sliding-Window PPL Evaluation

Compares prefill-stage perplexity across all 4 methods:

```bash
python evaluate.py --dataset wikitext --max-tokens 8192
python evaluate.py --run-all
```

### 4. Speed Benchmark (TTFT / TPOT / Throughput)

```bash
python benchmark.py --context-len 1024 --num-generate 128
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-sink` | 4 | Number of sink tokens (StreamingLLM / HybridKV) |
| `--window-size` | 32 | Observation window size |
| `--max-capacity` | 128 | Selected token budget (including sink) |
| `--kernel-size` | 5 | SnapKV pooling kernel size |

## Experiment Results

### Decode PPL (PG-19, context=1024, continuation=512)

| Method | cap=64 | cap=128 | cap=256 |
|--------|--------|---------|---------|
| Dense | 26.57 | 26.57 | 26.57 |
| StreamingLLM | 28.78 | 28.78 | 28.78 |
| SnapKV | **27.75** | 27.34 | 27.10 |
| **HybridKV** | 27.87 | **27.12** | **26.87** |

At cap=128 and cap=256, HybridKV achieves lower PPL than both StreamingLLM and SnapKV, approaching the Dense baseline.

### Ablation Study (PG-19, context=1024, cap=128)

| Configuration | Decode PPL | vs SnapKV |
|---------------|-----------|-----------|
| SnapKV (original) | 27.34 | --- |
| SnapKV+Sink (all voting + sink) | 27.35 | +0.01 |
| HybridKV S={0,1,2} (default) | 27.12 | -0.22 |
| HybridKV S={0,...,5} (all recency) | 26.96 | -0.38 |

Key findings: (1) Sink preservation alone does not improve SnapKV. (2) Replacing voting with recency in any layer subset improves over pure voting. (3) The optimal layer partitioning is model-scale dependent.

### Speed Benchmark (context=1024, generate=128)

| Method | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | KV Size |
|--------|-----------|-----------|--------------------|---------|
| Dense | 217.5 | 6.98 | 143.3 | 1151 |
| StreamingLLM | 239.5 | 4.67 | 214.1 | 163 |
| SnapKV | 243.4 | 5.10 | 196.0 | 287 |
| HybridKV | 215.7 | 5.14 | 194.7 | 287 |

## File Structure

```
├── baseline.py          # Dense baseline: load_data() + evaluate_ppl()
├── streaming_llm.py     # StreamingLLM compression + evaluation
├── snapkv.py            # SnapKV compression + evaluation
├── hybridkv.py          # HybridKV layer-adaptive compression + evaluation
├── evaluate.py          # Unified sliding-window PPL comparison (4 methods)
├── evaluate_decode.py   # Unified decode PPL comparison (4 methods)
├── ablation.py          # Ablation study: SnapKV+Sink, layer partitioning
├── benchmark.py         # Speed benchmark: TTFT, TPOT, throughput
├── main.tex             # NeurIPS-style paper (compile with xelatex)
├── neurips_2025.sty     # NeurIPS 2025 style file
├── tests/
│   └── test_compress.py # Unit tests for all compression methods
├── results/             # Experiment output (JSON)
└── requirements.txt
```

## References

- StreamingLLM: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
- SnapKV: "LLM Knows What You are Looking for Before Generation" (NeurIPS 2024)
- PyramidKV: "Dynamic KV Cache Compression based on Pyramidal Information Funneling" (TMLR 2025)
