"""Unit tests for KV cache compression methods."""

import pytest
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, DynamicCache
from snapkv import snap_kv_compress, SnapKVWrapper
from streaming_llm import streaming_llm_compress, StreamingLLMWrapper


@pytest.fixture(scope="module")
def model_and_input():
    """Load Pythia-70M and create a test input (loaded once for all tests)."""
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", attn_implementation="eager"
    )
    model.eval()
    text = "The quick brown fox jumps over the lazy dog. " * 30
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    return model, input_ids


def test_baseline_forward(model_and_input):
    """Dense baseline produces logits without error."""
    model, input_ids = model_and_input
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    assert out.logits.shape[0] == 1
    assert out.logits.shape[1] == input_ids.shape[1]


def test_snap_kv_compress_reduces_length(model_and_input):
    """SnapKV compress returns KV with correct reduced length."""
    model, input_ids = model_and_input
    window_size, max_capacity = 32, 64

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, output_attentions=True)

    key = out.past_key_values.layers[0].keys
    value = out.past_key_values.layers[0].values
    attn = out.attentions[0]

    new_key, new_value = snap_kv_compress(
        attn, key, value, window_size=window_size, max_capacity=max_capacity
    )
    expected_len = max_capacity + window_size
    assert new_key.shape[2] == expected_len
    assert new_value.shape[2] == expected_len


def test_snap_kv_compress_noop_short_seq():
    """SnapKV returns input unchanged if seq_len <= window + capacity."""
    batch, heads, seq_len, dim = 1, 8, 50, 64
    attn = torch.randn(batch, heads, seq_len, seq_len)
    key = torch.randn(batch, heads, seq_len, dim)
    value = torch.randn(batch, heads, seq_len, dim)
    new_key, new_value = snap_kv_compress(attn, key, value, window_size=32, max_capacity=64)
    assert new_key.shape == key.shape


def test_snapkv_wrapper_prefill(model_and_input):
    """SnapKVWrapper.prefill_and_compress returns compressed cache."""
    model, input_ids = model_and_input
    wrapper = SnapKVWrapper(model, window_size=32, max_capacity=64)
    outputs, compressed = wrapper.prefill_and_compress(input_ids)
    assert compressed.layers[0].keys.shape[2] == 64 + 32


def test_streaming_llm_compress(model_and_input):
    """StreamingLLM keeps exactly sink + window tokens."""
    model, input_ids = model_and_input
    n_sink, window_size = 4, 32

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)

    cache = out.past_key_values
    compressed = streaming_llm_compress(cache, n_sink=n_sink, window_size=window_size)
    expected_len = n_sink + window_size
    assert compressed.layers[0].keys.shape[2] == expected_len
    assert compressed.layers[0].values.shape[2] == expected_len


def test_streaming_llm_compress_noop_short():
    """StreamingLLM returns cache unchanged if seq <= sink + window."""
    cache = DynamicCache()
    key = torch.randn(1, 8, 30, 64)
    value = torch.randn(1, 8, 30, 64)
    cache.update(key, value, 0)
    result = streaming_llm_compress(cache, n_sink=4, window_size=32)
    assert result.layers[0].keys.shape[2] == 30


def test_streaming_llm_wrapper_prefill(model_and_input):
    """StreamingLLMWrapper.prefill_and_compress returns compressed cache."""
    model, input_ids = model_and_input
    wrapper = StreamingLLMWrapper(model, n_sink=4, window_size=32)
    outputs, compressed = wrapper.prefill_and_compress(input_ids)
    assert compressed.layers[0].keys.shape[2] == 4 + 32
    assert outputs.logits.shape[1] == input_ids.shape[1]
