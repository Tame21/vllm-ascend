# Adapted from vllm/v1/worker/mamba_utils.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def copy_overlapping_conv_state(
    src_block_addr,
    dst_block_addr,
    token_bias,
    conv_width,
    state_inner_size,
    state_elem_size,
    dim_rows,
    row_stride,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    """Copy a same-block convolution-state left shift safely.

    A speculative accept can make the source and destination ranges overlap:
    ``state[token_bias:] -> state[:conv_width-token_bias]``. A bulk parallel
    memcpy can overwrite a source lane before it is loaded. Copy complete token
    slices from low to high instead. The source and destination of each token
    slice are disjoint, so no program-wide barrier is required.

    Pointer casts stay outside dynamic loops for Triton-Ascend AxisInfo.
    """
    src_block_ptr = src_block_addr.to(tl.pointer_type(tl.uint8))
    dst_block_ptr = dst_block_addr.to(tl.pointer_type(tl.uint8))
    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    num_dst_tokens = conv_width - token_bias

    if CONV_STATE_DIM_FIRST:
        # DS layout: [block, dim, state_len]. Keep each row on a stable lane
        # while advancing the state_len/token axis in memmove order.
        for token_idx in range(0, num_dst_tokens):
            for row_base in range(0, dim_rows, COPY_BLOCK_SIZE):
                rows = row_base + offsets
                row_mask = rows < dim_rows
                src_token_offset = rows * row_stride + (token_idx + token_bias) * state_elem_size
                dst_token_offset = rows * row_stride + token_idx * state_elem_size
                # Element sizes are runtime metadata. Byte-wise copies keep all
                # pointer casts outside the loop and cover every supported dtype.
                for byte_idx in range(0, state_elem_size):
                    data = tl.load(
                        src_block_ptr + src_token_offset + byte_idx,
                        mask=row_mask,
                    )
                    tl.store(
                        dst_block_ptr + dst_token_offset + byte_idx,
                        data,
                        mask=row_mask,
                    )
        return

    # SD layout: [block, state_len, ...]. A complete token slice is contiguous.
    token_bytes = state_inner_size * state_elem_size
    for token_idx in range(0, num_dst_tokens):
        src_token_offset = (token_idx + token_bias) * token_bytes
        dst_token_offset = token_idx * token_bytes
        for offset in range(0, token_bytes, COPY_BLOCK_SIZE):
            mask = offset + offsets < token_bytes
            data = tl.load(
                src_block_ptr + src_token_offset + offset + offsets,
                mask=mask,
            )
            tl.store(
                dst_block_ptr + dst_token_offset + offset + offsets,
                data,
                mask=mask,
            )
