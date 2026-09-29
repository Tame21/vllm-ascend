# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Mapping
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cp_utils import cp_local_slot
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
)

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper

logger = logging.getLogger(__name__)


class AscendDFlashSpeculator(DFlashSpeculator):
    def _build_uniform_attn_metadata(
        self,
        batch_desc: BatchExecutionDescriptor,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        # The input kernel has applied rejection and model-length clamping.
        # Share this exact snapshot across KV groups and the graph replay rebuild.
        if self._draft_seq_lens_cpu_list is None:
            self._draft_seq_lens_cpu_list = self.input_buffers.seq_lens[:num_reqs].tolist()
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        seq_lens_cpu_list = self._draft_seq_lens_cpu_list + [0] * (num_reqs_padded - num_reqs)
        with build_attn_metadata_wrapper(seq_lens_cpu_list=seq_lens_cpu_list):
            return super()._build_uniform_attn_metadata(
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        with build_attn_metadata_wrapper():
            # vLLM main (#56181) replaced _build_draft_attn_metadata with
            # _build_uniform_attn_metadata (BatchExecutionDescriptor).
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
            )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return [attn_metadata]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        Upstream ``Speculator._build_draft_attn_metadata`` clamps
        ``query_start_loc`` at the real ``num_reqs`` to keep the cumulative
        series non-decreasing, so when a batch is padded to a capture size
        (``num_reqs_padded > num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._draft_seq_lens_cpu_list: list[int] | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # npu needs attn_backends to update full graph params in run_fullgraph.
        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        self._draft_seq_lens_cpu_list = None
        sync_state = dp_sync
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
        try:
            with build_attn_metadata_wrapper():
                return super().propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    sync_state,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    mm_inputs,
                    is_profile=is_profile,
                )
        finally:
            self._draft_seq_lens_cpu_list = None


# Share upstream's CP slot ownership and local-offset rules.
@triton.jit
def _prepare_dflash_inputs_kernel_ascend(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAD_BLOCK_SIZE: tl.constexpr = 256,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    nrejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - nrejected
    num_valid_ctx = valid_ctx_end - ctx_start

    nsampled = tl.load(num_sampled_ptr + req_idx)
    if nsampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    # Each program handles a tile of the context/query span. Query rows
    # overlap the rejected context suffix, but write to separate buffers.
    span = tl.maximum(num_ctx, num_valid_ctx + num_query_per_req)
    stride = tl.num_programs(1) * BLOCK_SIZE
    for start in range(block_idx * BLOCK_SIZE, span, stride):
        j = start + tl.arange(0, BLOCK_SIZE)
        is_ctx = j < num_ctx
        is_valid_ctx = j < num_valid_ctx
        query_off = j - num_valid_ctx
        is_query = (query_off >= 0) & (query_off < num_query_per_req)

        ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
        ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
        ctx_block_num = ctx_pos // (block_size * CP_SIZE)
        ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
        ctx_block_id = tl.load(
            block_table_ptr + req_idx * block_table_stride + ctx_block_num,
            mask=is_valid_ctx,
            other=0,
        ).to(tl.int64)
        local_ctx_slot = cp_local_slot(ctx_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID)
        ctx_slot = tl.where(is_valid_ctx & (ctx_block_id != 0), local_ctx_slot, PAD_SLOT_ID)
        # Rejected rows must be initialized too, so a later replay cannot
        # read positions or writable slots left by a previous batch.
        tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
        tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

        query_pos = last_valid_pos + 1 + query_off
        query_idx = query_base + query_off
        input_id = tl.where(query_off == 0, bonus_token, parallel_drafting_token_id)

        q_block_num = query_pos // (block_size * CP_SIZE)
        q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
        q_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + q_block_num, mask=is_query, other=0).to(
            tl.int64
        )
        local_q_slot = cp_local_slot(query_pos, q_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID)
        q_slot = tl.where(q_block_id != 0, local_q_slot, PAD_SLOT_ID)

        tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
        clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
        tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
        tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

        sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
        is_sample = is_query & (query_off >= sample_off)
        sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
        sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
        tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
        tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
        tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(out_seq_lens_ptr + req_idx, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))
        tl.store(out_temperature_ptr + req_state_idx, tl.load(temperature_ptr + req_state_idx))
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))

    # Distribute padding across ALL programs, independently of the request
    # tile size (which can be tiny during decode). Padding and active writes
    # are disjoint, so this needs neither a second launch nor a barrier.
    last_query_end = num_reqs * num_query_per_req
    sample_start = num_reqs * num_speculative_steps
    sample_end = max_num_reqs * num_speculative_steps
    pad_count = tl.maximum(max_num_reqs - num_reqs + 1, sample_end - sample_start)
    pad_count = tl.maximum(pad_count, max_num_tokens - last_query_end)
    pad_program = req_idx * tl.num_programs(1) + block_idx
    pad_stride = num_reqs * tl.num_programs(1) * PAD_BLOCK_SIZE
    for start in range(pad_program * PAD_BLOCK_SIZE, pad_count, pad_stride):
        offsets = start + tl.arange(0, PAD_BLOCK_SIZE)
        req = num_reqs + offsets
        tl.store(out_query_start_loc_ptr + req, last_query_end, mask=req <= max_num_reqs)
        tl.store(out_seq_lens_ptr + req, 0, mask=req < max_num_reqs)
        # Padded sample indices refer to valid query row 0; mapping -1
        # prevents the sampler from writing these rows to draft logits.
        sample = sample_start + offsets
        tl.store(out_sample_indices_ptr + sample, 0, mask=sample < sample_end)
        tl.store(out_sample_pos_ptr + sample, 0, mask=sample < sample_end)
        tl.store(out_sample_idx_mapping_ptr + sample, -1, mask=sample < sample_end)
        query = last_query_end + offsets
        tl.store(out_query_slot_mapping_ptr + query, PAD_SLOT_ID, mask=query < max_num_tokens)
