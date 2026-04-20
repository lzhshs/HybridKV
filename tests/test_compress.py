"""Unit tests for KV cache compression methods."""

import pytest
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer, DynamicCache


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
