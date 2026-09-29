# SPDX-License-Identifier: Apache-2.0
"""Measure MRV2 DFlash input preparation, including capacity padding.

Run this same script on the baseline and changed revisions on an Ascend NPU:
    python benchmarks/prepare_dflash_inputs.py --batch-sizes 1 8 32

Timing uses NPU events over repeated launches after warmup. It includes launch
gaps; it does not measure model throughput or compilation. Compare identical
device, software and arguments, and inspect a profiler trace for kernel time.
"""

import argparse
import itertools
import math

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import _prepare_dflash_inputs_kernel_ascend


def make_launch(batch, context_len, steps, capacity, max_tokens):
    device = "npu"
    width = steps + 1
    block_size = 128
    history = 256
    num_blocks = math.ceil((history + context_len + width) / block_size)
    tile = min(256, triton.next_power_of_2(context_len + width))
    grid = (batch, triton.cdiv(context_len + width, tile))

    def empty(size, dtype=torch.int32):
        return torch.empty(size, dtype=dtype, device=device)

    args = dict(
        out_input_ids_ptr=empty(max_tokens),
        out_query_positions_ptr=empty(max_tokens, torch.int64),
        out_query_start_loc_ptr=empty(capacity + 1),
        out_seq_lens_ptr=empty(capacity),
        out_query_slot_mapping_ptr=empty(max_tokens, torch.int64),
        out_context_positions_ptr=empty(batch * context_len, torch.int64),
        out_context_slot_mapping_ptr=empty(batch * context_len),
        out_sample_indices_ptr=empty(capacity * steps),
        out_sample_pos_ptr=empty(capacity * steps),
        out_sample_idx_mapping_ptr=empty(capacity * steps),
        out_temperature_ptr=empty(capacity, torch.float32),
        out_seeds_ptr=empty(capacity, torch.int64),
        target_positions_ptr=(torch.arange(context_len, device=device) + history).repeat(batch),
        target_query_start_loc_ptr=torch.arange(batch + 1, dtype=torch.int32, device=device) * context_len,
        idx_mapping_ptr=torch.arange(batch, dtype=torch.int32, device=device),
        last_sampled_ptr=torch.full((capacity,), 42, dtype=torch.int32, device=device),
        next_prefill_tokens_ptr=torch.full((capacity,), 43, dtype=torch.int32, device=device),
        num_sampled_ptr=torch.ones(batch, dtype=torch.int32, device=device),
        num_rejected_ptr=torch.zeros(batch, dtype=torch.int32, device=device),
        temperature_ptr=torch.ones(capacity, device=device),
        seeds_ptr=torch.zeros(capacity, dtype=torch.int64, device=device),
        block_table_ptr=torch.arange(1, batch * num_blocks + 1, dtype=torch.int32, device=device).view(
            batch, num_blocks
        ),
        block_table_stride=num_blocks,
        parallel_drafting_token_id=151643,
        block_size=block_size,
        num_query_per_req=width,
        num_speculative_steps=steps,
        max_num_reqs=capacity,
        max_num_tokens=max_tokens,
        max_model_len=history + context_len + width,
        cp_rank=0,
        SAMPLE_FROM_ANCHOR=False,
        PAD_SLOT_ID=-1,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        BLOCK_SIZE=tile,
    )
    return lambda: _prepare_dflash_inputs_kernel_ascend[grid](**args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[1, 1024])
    parser.add_argument("--max-num-tokens", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--max-num-reqs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    values = args.batch_sizes + args.context_lengths + args.max_num_tokens
    values += [args.max_num_reqs, args.steps, args.warmup, args.repeat]
    if min(values) <= 0:
        parser.error("All sizes and iteration counts must be positive")
    torch.npu.set_device(0)
    init_device_properties_triton()
    print("batch,context_len,steps,max_num_reqs,max_num_tokens,mean_us")
    for batch, context_len, max_tokens in itertools.product(
        args.batch_sizes, args.context_lengths, args.max_num_tokens
    ):
        if batch > args.max_num_reqs or batch * max(context_len, args.steps + 1) > max_tokens:
            continue
        launch = make_launch(batch, context_len, args.steps, args.max_num_reqs, max_tokens)
        for _ in range(args.warmup):
            launch()
        torch.npu.synchronize()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(args.repeat):
            launch()
        end.record()
        end.synchronize()
        mean_us = begin.elapsed_time(end) * 1000 / args.repeat
        print(f"{batch},{context_len},{args.steps},{args.max_num_reqs},{max_tokens},{mean_us:.3f}")


if __name__ == "__main__":
    main()
