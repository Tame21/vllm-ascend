# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.ops.triton.mamba.copy import copy_overlapping_conv_state


@triton.jit
def _copy_overlapping_conv_test_kernel(
    state_ptr,
    token_bias,
    conv_width,
    state_inner_size,
    state_elem_size,
    dim_rows,
    row_stride,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    copy_overlapping_conv_state(
        state_ptr,
        state_ptr,
        token_bias,
        conv_width,
        state_inner_size,
        state_elem_size,
        dim_rows,
        row_stride,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
    )


@pytest.mark.parametrize("token_bias", [1, 2, 3])
def test_sd_conv_same_block_left_shift_has_memmove_semantics(token_bias):
    device = torch.device("npu:0")
    conv_width = 5
    state_inner_size = 257
    state = torch.arange(
        conv_width * state_inner_size,
        dtype=torch.float32,
        device=device,
    ).reshape(conv_width, state_inner_size)
    snapshot = state.clone()
    expected = snapshot.clone()
    expected[: conv_width - token_bias].copy_(snapshot[token_bias:])

    _copy_overlapping_conv_test_kernel[(1,)](
        state,
        token_bias,
        conv_width,
        state_inner_size,
        state.element_size(),
        0,
        0,
        COPY_BLOCK_SIZE=256,
        CONV_STATE_DIM_FIRST=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(state, expected, rtol=0, atol=0)


@pytest.mark.parametrize("token_bias", [1, 2, 3])
def test_ds_conv_same_block_left_shift_has_memmove_semantics(token_bias):
    device = torch.device("npu:0")
    dim_rows = 513
    conv_width = 5
    state = torch.arange(
        dim_rows * conv_width,
        dtype=torch.float32,
        device=device,
    ).reshape(dim_rows, conv_width)
    snapshot = state.clone()
    expected = snapshot.clone()
    expected[:, : conv_width - token_bias].copy_(snapshot[:, token_bias:])

    _copy_overlapping_conv_test_kernel[(1,)](
        state,
        token_bias,
        conv_width,
        1,
        state.element_size(),
        dim_rows,
        state.stride(0) * state.element_size(),
        COPY_BLOCK_SIZE=256,
        CONV_STATE_DIM_FIRST=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(state, expected, rtol=0, atol=0)


def test_batch_memcpy_left_overlap_has_memmove_semantics():
    device = torch.device("npu:0")
    batch = 32
    row_bytes = 32 * 1024
    shift = 16
    copy_size = row_bytes - shift
    pattern = (torch.arange(row_bytes, dtype=torch.int32, device=device) % 251).to(torch.uint8)
    state = pattern.expand(batch, -1).clone()
    snapshot = state.clone()
    expected = snapshot.clone()
    expected[:, :copy_size].copy_(snapshot[:, shift:])

    row_stride_bytes = state.stride(0) * state.element_size()
    dst_ptrs = torch.tensor(
        [state.data_ptr() + row * row_stride_bytes for row in range(batch)],
        dtype=torch.uint64,
        device=device,
    )
    src_ptrs = torch.tensor(
        [state.data_ptr() + row * row_stride_bytes + shift for row in range(batch)],
        dtype=torch.uint64,
        device=device,
    )
    sizes = torch.full((batch,), copy_size, dtype=torch.int32, device=device)

    for _ in range(10):
        state.copy_(snapshot)
        batch_memcpy_kernel[(batch,)](
            src_ptrs,
            dst_ptrs,
            sizes,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
