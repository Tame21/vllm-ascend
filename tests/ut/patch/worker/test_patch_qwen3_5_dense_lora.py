# SPDX-License-Identifier: Apache-2.0

from math import prod
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


@pytest.mark.parametrize("missing_group", [0, 1])
def test_qwen_qkvz_packed_lora_allows_missing_group(missing_group):
    module = SimpleNamespace(
        _ascend_qwen3_5_qkvz_lora=True,
        n_slices=4,
        output_sizes=(4, 4, 8, 8),
    )
    lora_a = [torch.ones(8, 16), torch.full((8, 16), 2.0)]
    lora_b = [torch.arange(16 * 8).view(16, 8), torch.arange(8 * 8).view(8, 8)]
    lora_a[missing_group] = None
    lora_b[missing_group] = None

    expanded_a, expanded_b = dense_lora._expand_packed_lora(module, lora_a, lora_b)

    assert len(expanded_a) == len(expanded_b) == 4
    start, end = (0, 3) if missing_group == 0 else (3, 4)
    assert expanded_a[start:end] == [None] * (end - start)
    assert expanded_b[start:end] == [None] * (end - start)
    for index in range(4):
        if start <= index < end:
            continue
        group_index = 0 if index < 3 else 1
        assert expanded_a[index] is lora_a[group_index]
        assert expanded_b[index].shape == (module.output_sizes[index], 8)
    if missing_group == 1:
        assert torch.equal(torch.cat(expanded_b[:3], dim=0), lora_b[0])
    else:
        assert torch.equal(expanded_b[3], lora_b[1])


def test_qwen_qkvz_packed_lora_rejects_inconsistent_group_width():
    module = SimpleNamespace(
        _ascend_qwen3_5_qkvz_lora=True,
        n_slices=4,
        output_sizes=(4, 4, 8, 8),
    )
    lora_a = [torch.ones(8, 16), None]
    lora_b = [torch.ones(15, 8), None]

    with pytest.raises(ValueError, match="LoRA-B group width"):
        dense_lora._expand_packed_lora(module, lora_a, lora_b)


def test_qwen_qkvz_packed_lora_preserves_unrelated_modules():
    module = SimpleNamespace(_ascend_qwen3_5_qkvz_lora=False)
    lora_a, lora_b = [None], [None]
    with patch.object(dense_lora, "_ORIGINAL_EXPAND_PACKED_LORA", return_value=(lora_a, lora_b)) as original:
        result = dense_lora._expand_packed_lora(module, lora_a, lora_b)

    assert result == (lora_a, lora_b)
    original.assert_called_once_with(module, lora_a, lora_b)


def test_manager_marks_only_qwen_qkvz_packed_layers():
    class FakeMergedLayer:
        pass

    manager = SimpleNamespace()
    qkvz_layer = FakeMergedLayer()
    other_layer = FakeMergedLayer()
    wrapper = SimpleNamespace()

    def initialize_manager(*args):
        manager.supports_mm = False
        manager.punica_wrapper_mapping = {"language_model": wrapper}
        manager.modules = {
            "model.layers.0.linear_attn.in_proj_qkvz": qkvz_layer,
            "model.layers.0.linear_attn.in_proj_ba": other_layer,
        }

    with (
        patch.object(dense_lora, "MergedColumnParallelLinearWithLoRA", FakeMergedLayer),
        patch.object(dense_lora, "patch_applies", return_value=True),
        patch.object(dense_lora, "validate_config"),
        patch.object(dense_lora, "specialize_lora", return_value=False),
        patch.object(dense_lora, "_ORIGINAL_MANAGER_INIT", side_effect=initialize_manager),
        patch.object(dense_lora, "_install_shrink_padding"),
    ):
        dense_lora._manager_init(manager, "model", 8, 64, 32000, "lora_config", "npu", "config")

    assert qkvz_layer._ascend_qwen3_5_qkvz_lora is True
    assert not hasattr(other_layer, "_ascend_qwen3_5_qkvz_lora")


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
