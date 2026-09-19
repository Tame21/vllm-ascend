# SPDX-License-Identifier: Apache-2.0

from math import prod
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.patch.worker import patch_qwen3_5_dense_lora as dense_lora


@pytest.mark.parametrize(
    ("rank", "element_size", "expected"),
    [
        (4, 4, 8),
        (8, 4, 8),
        (9, 4, 16),
        (8, 2, 16),
    ],
)
def test_aligned_copy_elements(rank, element_size, expected):
    assert dense_lora._aligned_copy_elements(rank, element_size) == expected


def _config(max_lora_rank):
    return SimpleNamespace(
        lora_config=SimpleNamespace(max_lora_rank=max_lora_rank),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
            enable_sleep_mode=False,
        ),
        use_v2_model_runner=False,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
    )


@pytest.mark.parametrize("rank", [8, 16])
def test_validate_config_accepts_deployed_ranks(rank):
    dense_lora.validate_config(_config(rank))


def test_validate_config_rejects_other_ranks():
    with pytest.raises(ValueError, match="max_lora_rank 8 or 16"):
        dense_lora.validate_config(_config(32))


@pytest.mark.parametrize("rank", [8, 16])
def test_expand_slice_keeps_aligned_native_path(rank):
    original = MagicMock(return_value="kernel-result")
    expand = dense_lora._wrap_expand_slice(original)
    wrapper = SimpleNamespace(
        _ascend_qwen3_5_lora=True,
        _ascend_specialize_lora=False,
        no_lora=False,
    )
    y = torch.zeros(2, 32, dtype=torch.bfloat16)
    x = torch.zeros(2, rank, dtype=torch.float32)
    weights = torch.zeros(3, 1, 16, rank, dtype=torch.bfloat16)

    result = expand(wrapper, y, x, weights, 0, 16, True)

    assert result == "kernel-result"
    original.assert_called_once()
    args = original.call_args.args
    assert args[0] is wrapper
    assert args[1] is y
    assert args[2] is x
    assert args[3] is weights
    assert args[4:] == (0, 16, True)


@pytest.mark.parametrize(
    ("rank", "width", "offset", "full_width"),
    [
        (8, 8, 0, 16),
        (16, 8, 8, 24),
    ],
)
def test_expand_slice_pads_lora_b_rows_and_output(rank, width, offset, full_width):
    y = (
        torch.arange(2 * full_width, dtype=torch.float32)
        .to(torch.bfloat16)
        .view(2, full_width)
    )
    original_y = y.clone()
    x = torch.arange(2 * rank, dtype=torch.float32).view(2, rank)
    weights = (
        torch.arange(3 * width * rank, dtype=torch.float32)
        .to(torch.bfloat16)
        .view(3, 1, width, rank)
    )
    expected_delta = torch.full((2, width), 17.0, dtype=y.dtype)
    wrapper = SimpleNamespace(
        _ascend_qwen3_5_lora=True,
        _ascend_specialize_lora=False,
        no_lora=False,
    )

    def original(
        self_arg,
        padded_output,
        x_arg,
        padded_weights,
        kernel_offset,
        kernel_width,
        add_inputs,
    ):
        assert self_arg is wrapper
        assert x_arg is x
        assert kernel_offset == 0
        assert kernel_width == 16
        assert add_inputs is True
        assert padded_output.shape == (2, 16)
        assert padded_weights.shape == (3, 1, 16, rank)
        assert torch.equal(padded_output[..., :width], original_y[..., offset : offset + width])
        assert torch.count_nonzero(padded_output[..., width:]) == 0
        assert torch.equal(padded_weights[..., :width, :], weights)
        assert torch.count_nonzero(padded_weights[..., width:, :]) == 0
        padded_output[..., :width].copy_(expected_delta)
        return "kernel-result"

    expand = dense_lora._wrap_expand_slice(original)
    result = expand(wrapper, y, x, weights, offset, width, True)

    expected_y = original_y.clone()
    expected_y[..., offset : offset + width].copy_(expected_delta)
    assert result == "kernel-result"
    assert torch.equal(y, expected_y)


def test_expand_slice_uses_zeroed_temporary_for_overwrite():
    wrapper = SimpleNamespace(
        _ascend_qwen3_5_lora=True,
        _ascend_specialize_lora=False,
        no_lora=False,
    )
    y = torch.full((2, 16), 5.0, dtype=torch.float16)
    x = torch.zeros(2, 8, dtype=torch.float32)
    weights = torch.zeros(3, 1, 16, 8, dtype=torch.float16)

    def original(
        self_arg,
        padded_output,
        x_arg,
        weights_arg,
        kernel_offset,
        kernel_width,
        add_inputs,
    ):
        assert self_arg is wrapper
        assert x_arg is x
        assert weights_arg is weights
        assert kernel_offset == 0
        assert kernel_width == 16
        assert add_inputs is True
        assert torch.count_nonzero(padded_output) == 0
        padded_output.fill_(3.0)
        return None

    expand = dense_lora._wrap_expand_slice(original)
    expand(wrapper, y, x, weights, 0, 16, False)

    assert torch.equal(y, torch.full_like(y, 3.0))


def test_expand_slice_rejects_unsupported_rank():
    original = MagicMock()
    expand = dense_lora._wrap_expand_slice(original)
    wrapper = SimpleNamespace(
        _ascend_qwen3_5_lora=True,
        _ascend_specialize_lora=False,
        no_lora=False,
    )
    y = torch.zeros(2, 32, dtype=torch.float16)
    x = torch.zeros(2, 32, dtype=torch.float32)
    weights = torch.zeros(3, 1, 16, 32, dtype=torch.float16)

    with pytest.raises(ValueError, match="LoRA-B rank"):
        expand(wrapper, y, x, weights, 0, 16, True)

    original.assert_not_called()


def test_install_shrink_padding_wraps_each_kernel_once():
    wrapper = SimpleNamespace(bgmv_shrink=MagicMock(), sgmv_shrink=MagicMock())

    dense_lora._install_shrink_padding(wrapper)
    wrapped_bgmv = wrapper.bgmv_shrink
    wrapped_sgmv = wrapper.sgmv_shrink
    dense_lora._install_shrink_padding(wrapper)

    assert wrapper.bgmv_shrink is wrapped_bgmv
    assert wrapper.sgmv_shrink is wrapped_sgmv
    assert wrapper._ascend_qwen3_5_shrink_padding_installed is True


def test_shrink_kernel_keeps_original_path_for_aligned_rank():
    original = MagicMock(return_value="kernel-result")
    shrink = dense_lora._wrap_shrink_kernel(original)
    y = torch.zeros(2, 8, dtype=torch.float32)
    x = torch.zeros(2, 16, dtype=torch.float16)
    weights = torch.zeros(3, 1, 8, 16, dtype=torch.float16)
    indices = torch.tensor([0, 1])

    result = shrink(x, weights, y, indices, scaling=0.25)

    assert result == "kernel-result"
    original.assert_called_once()
    args = original.call_args.args
    assert args[0] is x
    assert args[1] is weights
    assert args[2] is y
    assert args[3] is indices
    assert original.call_args.kwargs == {"scaling": 0.25}


@pytest.mark.parametrize(
    ("weight_shape", "padded_weight_shape"),
    [
        ((3, 4, 16), (3, 8, 16)),
        ((3, 1, 4, 16), (3, 1, 8, 16)),
    ],
)
def test_shrink_kernel_pads_rank_weights_and_output(weight_shape, padded_weight_shape):
    rank = 4
    y = torch.arange(2 * rank, dtype=torch.float32).view(2, rank)
    original_y = y.clone()
    x = torch.zeros(2, 16, dtype=torch.bfloat16)
    weights = (
        torch.arange(prod(weight_shape), dtype=torch.float32)
        .to(torch.bfloat16)
        .view(weight_shape)
    )
    indices = torch.tensor([0, 1])
    expected_output = torch.full_like(y, 17.0)

    def original(x_arg, padded_weights, padded_output, indices_arg, scale_arg):
        assert x_arg is x
        assert indices_arg is indices
        assert scale_arg == 0.5
        assert padded_output.shape == (2, 8)
        assert padded_weights.shape == padded_weight_shape
        assert torch.equal(padded_output[..., :rank], original_y)
        assert torch.count_nonzero(padded_output[..., rank:]) == 0
        assert torch.equal(padded_weights[..., :rank, :], weights)
        assert torch.count_nonzero(padded_weights[..., rank:, :]) == 0
        padded_output.fill_(-1.0)
        padded_output[..., :rank].copy_(expected_output)
        return "kernel-result"

    shrink = dense_lora._wrap_shrink_kernel(original)

    result = shrink(x, weights, y, indices, 0.5)

    assert result == "kernel-result"
    assert torch.equal(y, expected_output)


def test_shrink_kernel_rejects_mismatched_weight_rank():
    original = MagicMock()
    shrink = dense_lora._wrap_shrink_kernel(original)
    y = torch.zeros(2, 4, dtype=torch.float32)
    x = torch.zeros(2, 16, dtype=torch.float16)
    weights = torch.zeros(3, 5, 16, dtype=torch.float16)
    indices = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="LoRA-A shrink shape"):
        shrink(x, weights, y, indices, 1.0)

    original.assert_not_called()
