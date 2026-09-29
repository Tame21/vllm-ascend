import gc

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel_ascend,
)


def test_prepare_dflash_inputs_clamps_seq_len_to_max_model_len():
    init_device_properties_triton()
    device = "npu"
    max_model_len = 16
    num_query_per_req = 3
    num_speculative_steps = 2

    outputs = {
        "out_input_ids_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_query_positions_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_query_start_loc_ptr": torch.empty(2, dtype=torch.int32, device=device),
        "out_seq_lens_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_query_slot_mapping_ptr": torch.empty(num_query_per_req, dtype=torch.int32, device=device),
        "out_context_positions_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_context_slot_mapping_ptr": torch.empty(1, dtype=torch.int32, device=device),
        "out_sample_indices_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_sample_pos_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_sample_idx_mapping_ptr": torch.empty(num_speculative_steps, dtype=torch.int32, device=device),
        "out_temperature_ptr": torch.empty(1, dtype=torch.float32, device=device),
        "out_seeds_ptr": torch.empty(1, dtype=torch.int64, device=device),
    }
    inputs = {
        "target_positions_ptr": torch.tensor([max_model_len - 1], dtype=torch.int32, device=device),
        "target_query_start_loc_ptr": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "idx_mapping_ptr": torch.tensor([0], dtype=torch.int32, device=device),
        "last_sampled_ptr": torch.tensor([42], dtype=torch.int32, device=device),
        "next_prefill_tokens_ptr": torch.tensor([43], dtype=torch.int32, device=device),
        "num_sampled_ptr": torch.tensor([1], dtype=torch.int32, device=device),
        "num_rejected_ptr": torch.tensor([0], dtype=torch.int32, device=device),
        "temperature_ptr": torch.tensor([1.0], dtype=torch.float32, device=device),
        "seeds_ptr": torch.tensor([0], dtype=torch.int64, device=device),
        "block_table_ptr": torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device),
    }
    kwargs = {
        **outputs,
        **inputs,
        "block_table_stride": 3,
        "parallel_drafting_token_id": 151643,
        "block_size": 8,
        "num_query_per_req": num_query_per_req,
        "num_speculative_steps": num_speculative_steps,
        "max_num_reqs": 1,
        "max_num_tokens": num_query_per_req,
        "max_model_len": max_model_len,
        "SAMPLE_FROM_ANCHOR": False,
        "PAD_SLOT_ID": -1,
        "BLOCK_SIZE": 1,
    }
    kwargs.update(cp_rank=0, CP_SIZE=1, CP_INTERLEAVE=1)

    _prepare_dflash_inputs_kernel_ascend[(1, 1)](**kwargs)

    torch.testing.assert_close(
        outputs["out_seq_lens_ptr"],
        torch.tensor([max_model_len], dtype=torch.int32, device=device),
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _make_inputs(ctx_lens, num_spec, max_num_reqs, max_num_tokens, cp_size, cp_rank, from_anchor):
    """Sentinel-filled CPU buffers, including guards beyond each writable span."""
    num_reqs = len(ctx_lens)
    width = num_spec if from_anchor else num_spec + 1
    query_start = [0]
    positions = []
    for i, length in enumerate(ctx_lens):
        query_start.append(query_start[-1] + length)
        positions.extend(range(127 + i, 127 + i + length))
    num_states = max_num_reqs + 3
    guard = 7

    def output(size, dtype=torch.int32):
        return torch.full((size + guard,), -999, dtype=dtype)

    outputs = {
        "out_input_ids_ptr": output(max_num_reqs * width),
        "out_query_positions_ptr": output(max_num_reqs * width),
        "out_query_start_loc_ptr": output(max_num_reqs + 1),
        "out_seq_lens_ptr": output(max_num_reqs),
        "out_query_slot_mapping_ptr": output(max_num_tokens, torch.int64),
        "out_context_positions_ptr": output(len(positions)),
        "out_context_slot_mapping_ptr": output(len(positions)),
        "out_sample_indices_ptr": output(max_num_reqs * num_spec),
        "out_sample_pos_ptr": output(max_num_reqs * num_spec),
        "out_sample_idx_mapping_ptr": output(max_num_reqs * num_spec),
        "out_temperature_ptr": output(num_states, torch.float32),
        "out_seeds_ptr": output(num_states, torch.int64),
    }
    # Null blocks model evicted SWA pages.
    block_table = torch.arange(1, num_reqs * 64 + 1, dtype=torch.int32).view(num_reqs, 64)[:, ::2]
    block_table[:, ::3] = 0
    inputs = {
        "target_positions_ptr": torch.tensor(positions, dtype=torch.int64),
        "target_query_start_loc_ptr": torch.tensor(query_start, dtype=torch.int32),
        "idx_mapping_ptr": torch.arange(num_reqs, 0, -1, dtype=torch.int32),
        "last_sampled_ptr": torch.arange(num_states, dtype=torch.int32) + 42,
        "next_prefill_tokens_ptr": torch.arange(num_states, dtype=torch.int32) + 142,
        "num_sampled_ptr": torch.arange(num_reqs, dtype=torch.int32) % 2,
        "num_rejected_ptr": torch.tensor([min(i + 1, n - 1) for i, n in enumerate(ctx_lens)], dtype=torch.int32),
        "temperature_ptr": torch.arange(num_states, dtype=torch.float32) / 10,
        "seeds_ptr": torch.arange(num_states, dtype=torch.int64) + 2**40,
        # The kernel contract requires a unit stride within each row.
        "block_table_ptr": block_table.clone(),
    }
    return {
        **outputs,
        **inputs,
        "block_table_stride": inputs["block_table_ptr"].stride(0),
        "parallel_drafting_token_id": 151643,
        "block_size": 16,
        "num_query_per_req": width,
        "num_speculative_steps": num_spec,
        "max_num_reqs": max_num_reqs,
        "max_num_tokens": max_num_tokens,
        "max_model_len": 256,
        "cp_rank": cp_rank,
        "CP_SIZE": cp_size,
        "CP_INTERLEAVE": 4,
        "SAMPLE_FROM_ANCHOR": from_anchor,
        "PAD_SLOT_ID": -1,
    }


def _reference(inputs, num_reqs):
    """Scalar reference, independent of the kernel's tile and grid layout."""
    result = {k: v.clone() for k, v in inputs.items() if k.startswith("out_")}
    width, steps = inputs["num_query_per_req"], inputs["num_speculative_steps"]
    capacity = inputs["max_num_reqs"]
    block_size, cp_size = inputs["block_size"], inputs["CP_SIZE"]
    interleave = inputs["CP_INTERLEAVE"]

    def slot(req, pos):
        table = inputs["block_table_ptr"]
        block = int(table[req, min(pos // (block_size * cp_size), table.shape[1] - 1)])
        offset = pos % (block_size * cp_size)
        if block == 0 or (offset // interleave) % cp_size != inputs["cp_rank"]:
            return -1
        local = (offset // (interleave * cp_size)) * interleave + offset % interleave
        return block * block_size + local

    for req in range(num_reqs):
        start, end = inputs["target_query_start_loc_ptr"][req : req + 2].tolist()
        valid_end = end - int(inputs["num_rejected_ptr"][req])
        last = int(inputs["target_positions_ptr"][valid_end - 1])
        state = int(inputs["idx_mapping_ptr"][req])
        bonus_source = "last_sampled_ptr" if inputs["num_sampled_ptr"][req] > 0 else "next_prefill_tokens_ptr"
        for idx in range(start, end):
            pos = int(inputs["target_positions_ptr"][idx]) if idx < valid_end else 0
            result["out_context_positions_ptr"][idx] = pos
            result["out_context_slot_mapping_ptr"][idx] = slot(req, pos) if idx < valid_end else -1
        for offset in range(width):
            idx, pos = req * width + offset, last + 1 + offset
            result["out_input_ids_ptr"][idx] = (
                inputs[bonus_source][state] if offset == 0 else inputs["parallel_drafting_token_id"]
            )
            result["out_query_positions_ptr"][idx] = min(pos, inputs["max_model_len"] - 1)
            result["out_query_slot_mapping_ptr"][idx] = slot(req, pos)
            sample_offset = 0 if inputs["SAMPLE_FROM_ANCHOR"] else 1
            if offset >= sample_offset:
                sample = req * steps + offset - sample_offset
                result["out_sample_indices_ptr"][sample] = idx
                result["out_sample_pos_ptr"][sample] = pos + int(inputs["SAMPLE_FROM_ANCHOR"])
                result["out_sample_idx_mapping_ptr"][sample] = state
        result["out_query_start_loc_ptr"][req] = req * width
        result["out_seq_lens_ptr"][req] = min(last + 1 + width, inputs["max_model_len"])
        result["out_temperature_ptr"][state] = inputs["temperature_ptr"][state]
        result["out_seeds_ptr"][state] = inputs["seeds_ptr"][state]

    result["out_query_start_loc_ptr"][num_reqs : capacity + 1] = num_reqs * width
    result["out_seq_lens_ptr"][num_reqs:capacity] = 0
    for key in ("out_sample_indices_ptr", "out_sample_pos_ptr", "out_sample_idx_mapping_ptr"):
        result[key][num_reqs * steps : capacity * steps] = -1 if key == "out_sample_idx_mapping_ptr" else 0
    result["out_query_slot_mapping_ptr"][num_reqs * width : inputs["max_num_tokens"]] = -1
    return result


@pytest.mark.parametrize(
    "ctx_lens,num_spec,capacity,max_tokens,tile,num_blocks",
    [
        ([1], 1, 1, 2, 4, 1),  # No padding except the terminal query offset.
        ([1, 7, 3], 3, 17, 8193, 16, 1),  # Decode tile is much smaller than the padding tile.
        ([300, 1, 513], 15, 32, 4099, 256, 3),  # Multiple context tiles and rejected suffixes.
        ([3, 19], 129, 9, 4096, 64, 4),  # Query/sample outputs span several tiles.
        ([300, 17], 3, 7, 1025, 32, 1),  # Grid-stride loop must cover an undersized grid.
    ],
)
@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("from_anchor", [False, True])
def test_prepare_dflash_inputs_matches_reference(
    ctx_lens, num_spec, capacity, max_tokens, tile, num_blocks, cp_size, cp_rank, from_anchor
):
    init_device_properties_triton()
    args = _make_inputs(ctx_lens, num_spec, capacity, max_tokens, cp_size, cp_rank, from_anchor)
    expected = _reference(args, len(ctx_lens))
    device_args = {k: v.to("npu") if isinstance(v, torch.Tensor) else v for k, v in args.items()}
    _prepare_dflash_inputs_kernel_ascend[(len(ctx_lens), num_blocks)](**device_args, BLOCK_SIZE=tile)
    for name, value in expected.items():
        torch.testing.assert_close(device_args[name].cpu(), value, msg=name)


def test_prepare_dflash_inputs_clears_padding_when_batch_shrinks():
    init_device_properties_triton()
    host_outputs, device_outputs = {}, {}
    for num_reqs in (6, 2, 5, 1):
        args = _make_inputs([7] * num_reqs, 3, 6, 1025, 2, 1, False)
        args.update(host_outputs)
        expected = _reference(args, num_reqs)
        device_args = {k: v.to("npu") if isinstance(v, torch.Tensor) else v for k, v in args.items()}
        device_args.update(device_outputs)
        _prepare_dflash_inputs_kernel_ascend[(num_reqs, 1)](**device_args, BLOCK_SIZE=16)
        for name, value in expected.items():
            torch.testing.assert_close(device_args[name].cpu(), value, msg=name)
        host_outputs = expected
        device_outputs = {name: device_args[name] for name in expected}
