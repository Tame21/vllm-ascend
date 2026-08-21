# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)
        tl.store(num_rejected_tokens_gpu_ptr + offsets, num_rejected, mask=mask)


@triton.jit
def copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    context_slot_mapping_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    num_kv_cache_blocks,  # maximum logical block id (exclusive)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    seq_lens_ptr,  # [num_reqs]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
    SAMPLE_FROM_ANCHOR: tl.constexpr = False,
):
    req_idx = tl.program_id(axis=0)
    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    for j in range(0, num_ctx):
        ctx_pos_idx = ctx_start + j
        pos = tl.load(target_positions_ptr + ctx_pos_idx)
        tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

        slot = tl.load(context_slot_mapping_ptr + ctx_pos_idx)
        tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

    if HAS_NUM_REJECTED:
        num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
    else:
        num_rejected = 0
        valid_ctx_end = ctx_end

    seq_len = tl.load(seq_lens_ptr + req_idx)
    effective_seq_len = seq_len - num_rejected
    last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)

    for q_idx in range(0, num_query_per_req):
        query_pos = last_pos + 1 + q_idx
        query_out_idx = req_idx * num_query_per_req + q_idx

        tl.store(out_query_positions_ptr + query_out_idx, query_pos)

        query_cache_pos = effective_seq_len + q_idx
        block_num_q = query_cache_pos // block_size
        valid_block_num = (block_num_q >= 0) & (block_num_q < block_table_stride)
        block_id_q = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_num_q,
            mask=valid_block_num,
            other=-1,
        ).to(tl.int64)
        valid_block_id = valid_block_num & (block_id_q >= 0) & (block_id_q < num_kv_cache_blocks)
        slot_q = tl.where(
            valid_block_id,
            block_id_q * block_size + (query_cache_pos % block_size),
            -1,
        )
        tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

        if q_idx == 0:
            bonus_token = tl.load(next_token_ids_ptr + req_idx)
            tl.store(out_input_ids_ptr + query_out_idx, bonus_token)
        else:
            tl.store(out_input_ids_ptr + query_out_idx, parallel_drafting_token_id)

        if SAMPLE_FROM_ANCHOR:
            sample_out_idx = req_idx * num_speculative_tokens + q_idx
            tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)
        else:
            if q_idx > 0:
                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)


@triton.jit
def build_dspark_context_slots_kernel(
    positions_ptr,  # [num_context_tokens] flattened retained target positions
    req_row_map_ptr,  # [num_reqs] compact subbatch row -> full batch row
    req_start_loc_ptr,  # [num_reqs + 1] per-request token start in the flat array
    block_table_ptr,  # [max_reqs, max_blocks] full-batch block table (one per group)
    block_table_stride,  # stride of block_table dim 0 (in elements)
    num_kv_cache_blocks,  # maximum logical block id (exclusive)
    out_slots_ptr,  # [num_context_tokens] output slots for this group
    block_size,  # tl.int32, KV group block size
    num_reqs,  # tl.int32
):
    # Rebuild DSpark context slots from retained positions and the *current*
    # block table. Used by DSpark decode-only lazy init; staged slot mappings
    # are never trusted because sliding-window managers may recycle blocks.
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    token_start = tl.load(req_start_loc_ptr + req_idx)
    token_end = tl.load(req_start_loc_ptr + req_idx + 1)
    full_row = tl.load(req_row_map_ptr + req_idx).to(tl.int64)

    for i in range(token_start, token_end):
        pos = tl.load(positions_ptr + i)
        block_num = pos // block_size
        valid_block_num = (block_num >= 0) & (block_num < block_table_stride)
        block_id = tl.load(
            block_table_ptr + full_row * block_table_stride + block_num,
            mask=valid_block_num,
            other=-1,
        ).to(tl.int64)
        valid_block_id = valid_block_num & (block_id >= 0) & (block_id < num_kv_cache_blocks)
        slot = tl.where(
            valid_block_id,
            block_id * block_size + (pos % block_size),
            -1,
        )
        tl.store(out_slots_ptr + i, slot)


def build_dspark_context_slots(
    positions: "torch.Tensor",
    req_row_map: "torch.Tensor",
    req_start_loc: "torch.Tensor",
    block_table: "torch.Tensor",
    block_size: int,
    out_slots: "torch.Tensor",
    num_kv_cache_blocks: int | None = None,
) -> None:
    """Batch-rebuild per-group DSpark context slots on device.

    Args:
        positions: flattened retained target positions [num_context_tokens].
        req_row_map: compact-row -> full-batch-row mapping [num_reqs], int32.
        req_start_loc: per-request token starts [num_reqs + 1], int32.
        block_table: current full-batch block table of one draft KV group.
        block_size: block size of that draft KV group.
        out_slots: output slot buffer [num_context_tokens], int32/int64.
        num_kv_cache_blocks: maximum logical block id (exclusive). If omitted,
            only block-table bounds and negative block IDs are guarded.
    """
    num_reqs = req_row_map.shape[0]
    if num_reqs == 0:
        return
    if num_kv_cache_blocks is None:
        num_kv_cache_blocks = torch.iinfo(torch.int32).max
    build_dspark_context_slots_kernel[num_reqs,](
        positions_ptr=positions,
        req_row_map_ptr=req_row_map.to(torch.int32),
        req_start_loc_ptr=req_start_loc.to(torch.int32),
        block_table_ptr=block_table,
        block_table_stride=block_table.stride(0),
        num_kv_cache_blocks=num_kv_cache_blocks,
        out_slots_ptr=out_slots,
        block_size=block_size,
        num_reqs=num_reqs,
    )
