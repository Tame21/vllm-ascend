# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.patch.worker import patch_qwen3_5_dense_lora as patch


def test_shrink_padding_preserves_logical_result():
    inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights = torch.tensor([[[[2.0, 3.0, 4.0]]]])
    output = torch.zeros((2, 1), dtype=torch.float32)

    def shrink(padded_inputs, padded_weights, padded_output):
        assert padded_weights.shape == (1, 1, 8, 3)
        assert padded_output.shape == (2, 8)
        assert torch.count_nonzero(padded_weights[..., 1:, :]) == 0
        padded_output.copy_(padded_inputs @ padded_weights[0, 0].T)

    patch._wrap_shrink_with_padding(shrink)(inputs, weights, output)

    expected = inputs @ weights[0, 0].T
    torch.testing.assert_close(output, expected)


def test_expand_padding_preserves_logical_result():
    inputs = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
    weights = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1)
    output = torch.ones((2, 16), dtype=torch.float32)

    def expand(padded_inputs, padded_weights, padded_output):
        assert padded_inputs.shape == (2, 8)
        assert padded_weights.shape == (1, 1, 16, 8)
        assert torch.count_nonzero(padded_inputs[:, 1:]) == 0
        assert torch.count_nonzero(padded_weights[..., 1:]) == 0
        padded_output.add_(padded_inputs @ padded_weights[0, 0].T)

    patch._wrap_expand_with_padding(expand)(inputs, weights, output)

    expected = torch.ones_like(output) + inputs @ weights[0, 0].T
    torch.testing.assert_close(output, expected)


def test_aligned_expand_calls_native_op_without_padding():
    inputs = torch.randn(2, 8)
    weights = torch.randn(1, 1, 16, 8)
    output = torch.zeros(2, 16)
    received = []

    def expand(native_inputs, native_weights, native_output):
        received.append((native_inputs, native_weights, native_output))

    patch._wrap_expand_with_padding(expand)(inputs, weights, output)

    assert len(received) == 1
    assert received[0][0] is inputs
    assert received[0][1] is weights
    assert received[0][2] is output


@pytest.mark.parametrize(
    ("input_rank", "weight_rank"),
    ((2, 1), (1, 2)),
)
def test_expand_rejects_mismatched_ranks(input_rank, weight_rank):
    inputs = torch.zeros(2, input_rank)
    weights = torch.zeros(1, 1, 16, weight_rank)
    output = torch.zeros(2, 16)

    with pytest.raises(ValueError, match="input rank must match"):
        patch._wrap_expand_with_padding(lambda *_: None)(inputs, weights, output)


def test_expand_rejects_unsupported_native_rank():
    inputs = torch.zeros(2, 24)
    weights = torch.zeros(1, 1, 32, 24)
    output = torch.zeros(2, 32)

    with pytest.raises(ValueError, match="Unsupported AscendC LoRA expand rank: 24"):
        patch._wrap_expand_with_padding(lambda *_: None)(inputs, weights, output)
