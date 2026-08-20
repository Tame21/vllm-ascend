# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer, _compute_num_programs
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler


@dataclass
class _PendingDSparkContext:
    """Target hidden states collected while a request is prefilling."""

    hidden_state_chunks: list[torch.Tensor] = field(default_factory=list)
    position_chunks: list[torch.Tensor] = field(default_factory=list)
    slot_mapping_chunks: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    num_tokens: int = 0


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's ``mtp`` method in user config, but its execution shape is
    closer to DFlash: target hidden states prepopulate draft K/V, then one
    anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DSpark probabilistic draft sampling is not supported on the v1 "
                "model runner; use greedy (the default) instead."
            )
        self.sample_from_anchor = getattr(self.draft_model_config.hf_config, "sample_from_anchor", True)
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_tokens
        else:
            self.num_query_per_req = 1 + self.num_speculative_tokens

        blk = 1 + self.num_speculative_tokens
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # DSpark is not supported in vllm v1, so related property needs to be reset here.
        del self.hidden_size, self.hidden_states, self._dflash_hidden_states  # type: ignore[has-type]
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        ascend_config = get_ascend_config()
        dynamic_spec_config = ascend_config.dynamic_spec_config
        self.dynamic_spec = None

        dspark_config = getattr(ascend_config, "dspark_config", None)
        self.decode_only = getattr(dspark_config, "execution_phase", "prefill_tail") == "decode_only"
        self.max_staged_tokens = getattr(dspark_config, "max_staged_tokens", 32768)
        self.overflow_policy = getattr(dspark_config, "overflow_policy", "fallback_prefill_tail")
        self._pending_contexts: dict[str, _PendingDSparkContext] = {}
        self._ready_request_ids: set[str] = set()
        self._disabled_request_ids: set[str] = set()
        self._pending_contexts_for_next_proposal: list[str] | None = None
        self._prefill_fallback_request_ids: set[str] = set()
        self._decode_only_active_request_indices: list[int] | None = None
        self._decode_only_empty_request_ids: set[str] = set()
        self._decode_only_force_full_batch = False
        self._decode_only_compact_block_tables: dict[int, torch.Tensor] | None = None
        self._decode_only_compact_slot_mappings: dict[int, torch.Tensor] | None = None
        self._decode_only_num_verify_tokens: torch.Tensor | None = None

        if self.decode_only and getattr(self, "use_async_scheduling", False):
            raise ValueError(
                "DSpark decode_only is not supported with async scheduling; "
                "use execution_phase='prefill_tail' or disable async scheduling."
            )
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        if self.decode_only and getattr(kv_transfer_config, "is_kv_transfer_instance", False):
            raise ValueError(
                "DSpark decode_only is not supported with PD KV transfer; "
                "use execution_phase='prefill_tail' until staged context transfer is implemented."
            )
        cache_config = getattr(vllm_config, "cache_config", None)
        if self.decode_only and getattr(cache_config, "enable_prefix_caching", False):
            raise ValueError(
                "DSpark decode_only is not supported with prefix caching; "
                "disable prefix caching or use execution_phase='prefill_tail'."
            )

        if dynamic_spec_config.method == "dspark":
            self.dynamic_spec = DynamicSpecScheduler(
                method="dspark",
                method_params=dynamic_spec_config.method_params,
                max_batch_size=self.max_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
                device=device,
            )
        # DSpark runs eager only (Ascend cudagraph unsupported on this path).
        self.use_cuda_graph = False
        # Max query tokens depend on whether sampling from anchor or not.
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
        # Position ids for the draft query block [max_query_tokens].
        # Overrides dflash:49; v2 uses input_buffers.positions.
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Primary-group query slot mapping buffer [max_query_tokens].
        # Overrides dflash:37; v2 uses BlockTables.slot_mappings. Per-non-
        # primary-gid buffers live in _per_group_query_slot_mapping_buffers.
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        # TODO simplify these comments
        # block_table / slot_mapping bookkeeping (10 dicts below). v1 self-
        # manages per kv_cache_group_id / per layer because it lacks v2's
        # BlockTables scaffold; v2 injects a single self.block_tables
        # (BlockTables, with .slot_mappings) + build_slot_mappings_by_layer,
        # so the speculator holds none of these. P2 refactor target (move to
        # runner).

        # per-gid block_table from runner (just read)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        # per-gid slot_mapping from runner (just read)
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}

        # per-gid block_table (use in proposer)
        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        # per-gid query slot_mapping buffer
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        # per-gid context slot_mapping buffer
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}

        # per-layer context slot mappings as a flat list
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None

    def clear_decode_only_contexts(self, request_ids: list[str] | set[str] | tuple[str, ...]) -> None:
        """Drop staged/ready state for requests that were preempted or reset."""
        request_id_set = set(request_ids)
        for request_id in request_id_set:
            self._pending_contexts.pop(request_id, None)
            self._ready_request_ids.discard(request_id)
            self._disabled_request_ids.discard(request_id)
            self._prefill_fallback_request_ids.discard(request_id)
            self._decode_only_empty_request_ids.discard(request_id)
        if self._pending_contexts_for_next_proposal is not None:
            self._pending_contexts_for_next_proposal = [
                request_id
                for request_id in self._pending_contexts_for_next_proposal
                if request_id not in request_id_set
            ]

    def clear_finished_decode_only_contexts(self, requests: dict[str, Any] | None) -> None:
        """Release state for finished requests still retained by the runner."""
        if not requests:
            return
        finished_request_ids = []
        for request_id, request in requests.items():
            is_finished = getattr(request, "is_finished", None)
            if callable(is_finished):
                is_finished = is_finished()
            if is_finished:
                finished_request_ids.append(request_id)
        if finished_request_ids:
            self.clear_decode_only_contexts(finished_request_ids)

    @staticmethod
    def _prefill_mask(common_attn_metadata: CommonAttentionMetadata) -> list[bool]:
        """Return the request-level prefill mask without device synchronization."""
        is_prefilling = getattr(common_attn_metadata, "is_prefilling", None)
        if is_prefilling is None:
            return [False] * common_attn_metadata.num_reqs
        if not torch.is_tensor(is_prefilling):
            return [bool(value) for value in is_prefilling[: common_attn_metadata.num_reqs]]
        if is_prefilling.device.type != "cpu":
            is_prefilling = is_prefilling.cpu()
        return [bool(value) for value in is_prefilling[: common_attn_metadata.num_reqs].tolist()]

    @staticmethod
    def _query_offsets(common_attn_metadata: CommonAttentionMetadata) -> list[int]:
        query_start_loc = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc is None:
            query_start_loc = common_attn_metadata.query_start_loc.cpu()
        elif query_start_loc.device.type != "cpu":
            query_start_loc = query_start_loc.cpu()
        return [int(value) for value in query_start_loc[: common_attn_metadata.num_reqs + 1].tolist()]

    def _stage_prefill_context(
        self,
        request_ids: list[str],
        target_hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        prefill_mask: list[bool],
    ) -> bool:
        """Save raw target states; DSpark projection is deferred to decode."""
        if target_positions.ndim != 1:
            raise RuntimeError("DSpark decode_only currently requires one-dimensional target positions.")

        offsets = self._query_offsets(common_attn_metadata)
        num_computed_tokens = getattr(common_attn_metadata, "num_computed_tokens_cpu", None)
        if num_computed_tokens is None:
            num_computed_tokens = getattr(common_attn_metadata, "_num_computed_tokens_cpu", None)
        if num_computed_tokens is not None:
            if num_computed_tokens.device.type != "cpu":
                num_computed_tokens = num_computed_tokens.cpu()
            num_computed_tokens = [
                int(value) for value in num_computed_tokens[: common_attn_metadata.num_reqs].tolist()
            ]
        else:
            num_computed_tokens = [0] * common_attn_metadata.num_reqs

        fallback_required = False
        for request_index, is_prefill in enumerate(prefill_mask):
            if not is_prefill:
                continue
            if request_index >= len(request_ids):
                raise RuntimeError("DSpark decode_only received fewer request IDs than attention metadata.")

            request_id = request_ids[request_index]
            context = self._pending_contexts.get(request_id)
            if num_computed_tokens[request_index] == 0:
                self._disabled_request_ids.discard(request_id)
                if context is not None and context.num_tokens > 0:
                    self._pending_contexts.pop(request_id, None)
                    context = None
                self._ready_request_ids.discard(request_id)
            if context is None and request_id in self._ready_request_ids:
                # An overflowed request has already fallen back to the
                # original prefill-tail path. Keep it on that path for every
                # remaining prefill chunk instead of trying to restage a
                # prefix that is already in the draft KV cache.
                fallback_required = True
                continue
            if context is None and num_computed_tokens[request_index] > 0:
                raise RuntimeError(
                    "DSpark decode_only cannot stage a request with an already-computed prefix. "
                    f"request_id={request_id!r}; disable prefix caching or use execution_phase='prefill_tail'."
                )

            start, end = offsets[request_index], offsets[request_index + 1]
            if end < start or end > target_hidden_states.shape[0] or end > target_positions.shape[0]:
                raise RuntimeError(
                    "DSpark decode_only received inconsistent prefill metadata: "
                    f"request_id={request_id!r}, start={start}, end={end}, "
                    f"hidden_tokens={target_hidden_states.shape[0]}, position_tokens={target_positions.shape[0]}"
                )
            chunk_tokens = end - start
            if chunk_tokens == 0:
                continue

            previous_tokens = 0 if context is None else context.num_tokens
            if previous_tokens + chunk_tokens > self.max_staged_tokens:
                if self.overflow_policy == "reject":
                    raise RuntimeError(
                        "DSpark decode_only staged context exceeds max_staged_tokens: "
                        f"request_id={request_id!r}, tokens={previous_tokens + chunk_tokens}, "
                        f"max_staged_tokens={self.max_staged_tokens}"
                    )
                logger.warning_once(
                    "DSpark decode_only will use the prefill-tail fallback for request %s because "
                    "staged context exceeds max_staged_tokens=%d.",
                    request_id,
                    self.max_staged_tokens,
                )
                fallback_required = True
                if context is None:
                    # The current target prefill contains the complete context
                    # when there is no previously staged chunk. Mark it only
                    # after the normal DSpark prefill path has run.
                    self._prefill_fallback_request_ids.add(request_id)
                continue

            if context is None:
                context = _PendingDSparkContext()
                self._pending_contexts[request_id] = context

            context.hidden_state_chunks.append(target_hidden_states[start:end].detach().clone())
            context.position_chunks.append(target_positions[start:end].detach().clone())
            for gid, slot_mapping in self._per_group_slot_mappings.items():
                if end > slot_mapping.shape[0]:
                    raise RuntimeError(
                        "DSpark decode_only received a slot mapping shorter than the prefill chunk: "
                        f"request_id={request_id!r}, kv_cache_gid={gid}, end={end}, "
                        f"slot_mapping_tokens={slot_mapping.shape[0]}"
                    )
                context.slot_mapping_chunks.setdefault(gid, []).append(slot_mapping[start:end].detach().clone())
            context.num_tokens += chunk_tokens
            self._ready_request_ids.discard(request_id)
        return fallback_required

    @staticmethod
    def _select_request_values(values: Any, request_indices: list[int]) -> Any:
        if values is None:
            return None
        if torch.is_tensor(values):
            index = torch.tensor(request_indices, dtype=torch.long, device=values.device)
            return values.index_select(0, index)
        try:
            return values[request_indices]
        except (IndexError, TypeError):
            return [values[index] for index in request_indices]

    @staticmethod
    def _select_token_values(values: torch.Tensor | None, token_indices: list[int]) -> torch.Tensor | None:
        if values is None:
            return None
        index = torch.tensor(token_indices, dtype=torch.long, device=values.device)
        axis = 0 if values.ndim == 1 else values.ndim - 1
        return values.index_select(axis, index)

    def _clear_decode_only_compact_metadata(self) -> None:
        self._decode_only_compact_block_tables = None
        self._decode_only_compact_slot_mappings = None

    def _compact_common_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        request_indices: list[int],
        token_indices: list[int],
        query_lengths: list[int],
    ) -> CommonAttentionMetadata:
        """Build a request-compact metadata view for mixed decode batches."""
        query_start_loc_cpu = torch.zeros(
            len(request_indices) + 1,
            dtype=common_attn_metadata.query_start_loc_cpu.dtype,
        )
        if query_lengths:
            query_start_loc_cpu[1:] = torch.cumsum(
                torch.tensor(query_lengths, dtype=query_start_loc_cpu.dtype), dim=0
            )
        query_start_loc = query_start_loc_cpu.to(
            device=common_attn_metadata.query_start_loc.device,
            dtype=common_attn_metadata.query_start_loc.dtype,
        )
        request_index_device = torch.tensor(
            request_indices,
            dtype=torch.long,
            device=common_attn_metadata.seq_lens.device,
        )
        token_index_device = torch.tensor(
            token_indices,
            dtype=torch.long,
            device=common_attn_metadata.slot_mapping.device,
        )

        dataclass_fields = getattr(common_attn_metadata, "__dataclass_fields__", {})
        updates: dict[str, Any] = {
            "query_start_loc": query_start_loc,
            "query_start_loc_cpu": query_start_loc_cpu,
            "seq_lens": common_attn_metadata.seq_lens.index_select(0, request_index_device),
            "num_reqs": len(request_indices),
            "num_actual_tokens": len(token_indices),
            "max_query_len": max(query_lengths, default=0),
            "block_table_tensor": common_attn_metadata.block_table_tensor.index_select(0, request_index_device),
            "slot_mapping": common_attn_metadata.slot_mapping.index_select(0, token_index_device),
            "num_input_tokens": len(token_indices),
            "actual_seq_lengths_q": torch.cumsum(
                torch.tensor(query_lengths, dtype=torch.int32), dim=0
            ).tolist(),
        }

        request_fields = (
            "seq_lens_cpu",
            "num_computed_tokens_cpu",
            "_seq_lens_cpu",
            "_num_computed_tokens_cpu",
            "seq_lens_cpu_upper_bound",
            "encoder_seq_lens",
            "encoder_seq_lens_cpu",
            "dcp_local_seq_lens",
            "dcp_local_seq_lens_cpu",
            "is_prefilling",
            "rswa_prefix_lens",
            "replayssm_decode_base_cpu",
            "group_len",
            "group_key_idx",
            "group_key_cache_idx",
            "req_ids_tensor",
        )
        for field_name in request_fields:
            if field_name in dataclass_fields:
                updates[field_name] = self._select_request_values(
                    getattr(common_attn_metadata, field_name, None), request_indices
                )

        token_fields = ("positions", "positions_cpu")
        for field_name in token_fields:
            if field_name in dataclass_fields:
                updates[field_name] = self._select_token_values(
                    getattr(common_attn_metadata, field_name, None), token_indices
                )

        if "causal" in dataclass_fields and torch.is_tensor(common_attn_metadata.causal):
            updates["causal"] = common_attn_metadata.causal.index_select(0, request_index_device)
        if "token_to_req" in dataclass_fields:
            token_to_req = getattr(common_attn_metadata, "token_to_req", None)
            if token_to_req is not None:
                token_to_req = token_to_req.index_select(0, token_index_device)
                remap = torch.full(
                    (common_attn_metadata.num_reqs,),
                    -1,
                    dtype=token_to_req.dtype,
                    device=token_to_req.device,
                )
                remap[request_index_device] = torch.arange(
                    len(request_indices), dtype=token_to_req.dtype, device=token_to_req.device
                )
                updates["token_to_req"] = remap[token_to_req]

        if "mm_req_doc_ranges" in dataclass_fields:
            mm_ranges = getattr(common_attn_metadata, "mm_req_doc_ranges", None)
            if mm_ranges is not None:
                updates["mm_req_doc_ranges"] = {
                    compact_index: mm_ranges.get(original_index, [])
                    for compact_index, original_index in enumerate(request_indices)
                }

        if "_num_computed_tokens_cache" in dataclass_fields:
            updates["_num_computed_tokens_cache"] = None
        if "_token_to_req_indices_cache" in dataclass_fields:
            updates["_token_to_req_indices_cache"] = None

        return common_attn_metadata.replace(
            **{field_name: value for field_name, value in updates.items() if field_name in dataclass_fields}
        )

    def compact_decode_only_inputs(
        self,
        request_ids: list[str],
        request_indices: list[int],
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        CommonAttentionMetadata,
        torch.Tensor | None,
        Any,
    ]:
        """Compact only the requests that are allowed to run DSpark."""
        offsets = self._query_offsets(common_attn_metadata)
        query_lengths = [offsets[index + 1] - offsets[index] for index in request_indices]
        token_indices = [
            token_index
            for request_index in request_indices
            for token_index in range(offsets[request_index], offsets[request_index + 1])
        ]
        token_index_device = torch.tensor(token_indices, dtype=torch.long, device=target_token_ids.device)
        request_index_device = torch.tensor(request_indices, dtype=torch.long, device=next_token_ids.device)
        self._decode_only_compact_block_tables = {
            gid: block_table.index_select(0, request_index_device.to(block_table.device))
            for gid, block_table in self._per_group_block_tables.items()
        }
        self._decode_only_compact_slot_mappings = {
            gid: slot_mapping.index_select(
                0, torch.tensor(token_indices, dtype=torch.long, device=slot_mapping.device)
            )
            for gid, slot_mapping in self._per_group_slot_mappings.items()
        }
        compact_metadata = self._compact_common_attn_metadata(
            common_attn_metadata,
            request_indices,
            token_indices,
            query_lengths,
        )
        compact_req_scheduled_tokens = req_scheduled_tokens
        if isinstance(req_scheduled_tokens, dict):
            compact_req_scheduled_tokens = {
                request_ids[index]: req_scheduled_tokens[request_ids[index]]
                for index in request_indices
                if request_ids[index] in req_scheduled_tokens
            }
        return (
            target_token_ids.index_select(0, token_index_device),
            self._select_token_values(target_positions, token_indices),
            target_hidden_states.index_select(0, token_index_device),
            next_token_ids.index_select(0, request_index_device),
            self._select_request_values(token_indices_to_sample, request_indices),
            compact_metadata,
            self._select_request_values(num_rejected_tokens_gpu, request_indices),
            compact_req_scheduled_tokens,
        )

    def get_decode_only_active_request_indices(self) -> list[int] | None:
        return self._decode_only_active_request_indices

    def get_decode_only_prefill_mask(self, common_attn_metadata: CommonAttentionMetadata) -> list[bool]:
        """Read the request phase before generic speculative metadata rewrites it."""
        return self._prefill_mask(common_attn_metadata)

    def scatter_decode_only_draft_token_ids(
        self,
        draft_token_ids: torch.Tensor,
        request_indices: list[int],
        request_ids: list[str],
        num_reqs: int,
    ) -> torch.Tensor:
        """Restore compact draft rows and record which rows must stay empty."""
        self._decode_only_empty_request_ids = {
            request_ids[index]
            for index in range(min(num_reqs, len(request_ids)))
            if index not in request_indices
        }
        if len(request_indices) == num_reqs:
            self._clear_decode_only_compact_metadata()
            return draft_token_ids
        scattered = torch.zeros(
            (num_reqs, draft_token_ids.shape[1]),
            dtype=draft_token_ids.dtype,
            device=draft_token_ids.device,
        )
        if request_indices:
            request_index_device = torch.tensor(request_indices, dtype=torch.long, device=draft_token_ids.device)
            scattered.index_copy_(0, request_index_device, draft_token_ids)
        dynamic_spec = self.dynamic_spec
        compact_num_verify_tokens = None if dynamic_spec is None else dynamic_spec.num_verify_tokens
        if compact_num_verify_tokens is not None:
            full_num_verify_tokens = torch.zeros(
                num_reqs,
                dtype=compact_num_verify_tokens.dtype,
                device=compact_num_verify_tokens.device,
            )
            if request_indices:
                request_index_device = torch.tensor(
                    request_indices,
                    dtype=torch.long,
                    device=compact_num_verify_tokens.device,
                )
                full_num_verify_tokens.index_copy_(
                    0, request_index_device, compact_num_verify_tokens[: len(request_indices)]
                )
            self._decode_only_num_verify_tokens = full_num_verify_tokens
        self._clear_decode_only_compact_metadata()
        return scattered

    def _initialize_pending_contexts_in_forward(self) -> None:
        """Write staged prompt KV while the draft forward context is active."""
        request_ids = self._pending_contexts_for_next_proposal
        if not request_ids:
            return

        try:
            contexts = [self._pending_contexts[request_id] for request_id in request_ids]
            raw_hidden_states = torch.cat(
                [chunk for context in contexts for chunk in context.hidden_state_chunks], dim=0
            )
            context_positions = torch.cat(
                [chunk for context in contexts for chunk in context.position_chunks], dim=0
            )
            combined_hidden_states = self.model.combine_hidden_states(raw_hidden_states)

            unique_group_ids = list(dict.fromkeys(self._layer_group_idx))
            slot_mapping_by_gid: dict[int, torch.Tensor] = {}
            for gid in unique_group_ids:
                slot_chunks = [
                    slot_chunk
                    for context in contexts
                    for slot_chunk in context.slot_mapping_chunks.get(gid, [])
                ]
                if not slot_chunks:
                    raise RuntimeError(
                        "DSpark decode_only has no staged slot mapping for a draft KV cache group: "
                        f"kv_cache_gid={gid}"
                    )
                slot_mapping_by_gid[gid] = torch.cat(slot_chunks, dim=0)

            context_slot_mapping = [slot_mapping_by_gid[gid] for gid in self._layer_group_idx]
            self.model.precompute_and_store_context_kv(
                combined_hidden_states,
                context_positions,
                context_slot_mapping,
            )
        except Exception:
            for request_id in request_ids:
                self._pending_contexts.pop(request_id, None)
                self._ready_request_ids.discard(request_id)
            self._pending_contexts_for_next_proposal = None
            raise

        for request_id in request_ids:
            self._pending_contexts.pop(request_id, None)
            self._ready_request_ids.add(request_id)
        self._pending_contexts_for_next_proposal = None

    def prepare_decode_only_context(
        self,
        request_ids: list[str],
        target_hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        preempted_request_ids: list[str] | set[str] | tuple[str, ...] | None = None,
        prefill_mask: list[bool] | None = None,
    ) -> bool:
        """Prepare request state and report whether the draft pass should be skipped.

        Prefill rows are staged and excluded from the draft batch. Pure decode
        batches run the normal proposer; mixed batches compact only eligible
        decode rows and scatter their results back. Pending prompt KV is
        initialized inside the draft forward context.
        """
        if not self.decode_only:
            return False

        if preempted_request_ids:
            self.clear_decode_only_contexts(preempted_request_ids)
        self.clear_finished_decode_only_contexts(getattr(self.runner, "requests", None))
        self._pending_contexts_for_next_proposal = None
        self._prefill_fallback_request_ids.clear()
        self._decode_only_active_request_indices = None
        self._decode_only_empty_request_ids.clear()
        self._decode_only_force_full_batch = False
        self._decode_only_num_verify_tokens = None
        self._clear_decode_only_compact_metadata()

        prefill_mask = prefill_mask or self._prefill_mask(common_attn_metadata)
        active_request_count = min(common_attn_metadata.num_reqs, len(request_ids))
        if any(prefill_mask):
            fallback_required = self._stage_prefill_context(
                request_ids,
                target_hidden_states,
                target_positions,
                common_attn_metadata,
                prefill_mask,
            )
            if fallback_required:
                self._decode_only_force_full_batch = True
                self._pending_contexts_for_next_proposal = [
                    request_id
                    for request_id in request_ids[: common_attn_metadata.num_reqs]
                    if request_id in self._pending_contexts
                ] or None
                return False
        decode_indices = [index for index in range(active_request_count) if not prefill_mask[index]]
        if decode_indices and len(decode_indices) < active_request_count and getattr(self, "dcp_size", 1) > 1:
            # The DCP metadata carries rank-local request state that cannot be
            # safely remapped by the MVP compact path. Let the requests make a
            # normal target-only step and retry DSpark on the next batch.
            self._decode_only_active_request_indices = []
            self._decode_only_empty_request_ids = set(request_ids[:active_request_count])
            return True
        active_request_ids = [request_ids[index] for index in decode_indices]
        unknown_request_ids = [
            request_id
            for request_id in active_request_ids
            if request_id not in self._ready_request_ids
            and request_id not in self._pending_contexts
            and request_id not in self._disabled_request_ids
        ]
        if unknown_request_ids:
            logger.warning_once(
                "DSpark decode_only has no staged context for request %s; "
                "DSpark will be skipped for this batch.",
                unknown_request_ids[0],
            )
            self._disabled_request_ids.update(unknown_request_ids)
        selected_indices = [
            index
            for index in decode_indices
            if request_ids[index] not in self._disabled_request_ids
        ]
        self._pending_contexts_for_next_proposal = [
            request_ids[index] for index in selected_indices if request_ids[index] in self._pending_contexts
        ] or None
        self._decode_only_active_request_indices = selected_indices
        self._decode_only_empty_request_ids = {
            request_ids[index]
            for index in range(active_request_count)
            if index not in selected_indices
        }
        return not selected_indices

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
        context_slots: torch.Tensor | list[torch.Tensor],
    ) -> None:
        self._initialize_pending_contexts_in_forward()
        super().build_model_inputs_first_pass(num_input_tokens, context_slots)
        if self._prefill_fallback_request_ids:
            self._ready_request_ids.update(self._prefill_fallback_request_ids)
            self._prefill_fallback_request_ids.clear()

    def _compute_confidence_logits(
        self,
        last_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        num_tokens = num_reqs * self.num_speculative_tokens
        flat_hidden = last_hidden_states.reshape(num_tokens, last_hidden_states.shape[-1])
        # Markov embeddings of the draft input tokens (cheap lookup, so they
        # are recomputed here instead of being captured in the drafting loop).
        markov_embs = self.model.markov_embed(draft_token_ids[:, : self.num_speculative_tokens])
        # The confidence head concatenates both inputs, so their dtypes must
        # match; it upcasts to float32 internally.
        flat_markov = markov_embs.reshape(num_tokens, markov_embs.shape[-1]).to(flat_hidden.dtype)
        conf_raw = self.model.confidence_logits(flat_hidden, flat_markov)
        confidence_logits = self._dspark_confidence_logits_buffer[:num_reqs]
        confidence_logits.copy_(conf_raw.reshape(num_reqs, self.num_speculative_tokens))
        return confidence_logits

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        attention_groups_list: list[dict[tuple[str, str], AttentionGroup]] = []
        # the draft layers have multiple kv_cache_groups
        if not hasattr(self.model, "get_draft_kv_cache_layer_names"):
            raise RuntimeError(
                "DSpark standard-cache path requires the draft model to expose get_draft_kv_cache_layer_names"
            )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))

        # there are many kv groups other than one
        for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            draft_layer_names_in_group = set(kv_cache_group_spec.layer_names) & self._draft_attn_layer_names
            if not draft_layer_names_in_group:
                continue

            attention_groups: dict[tuple[str, Any], AttentionGroup] = {}
            # iterate in a way like vllm's llm_base_proposer
            for layer_name in draft_layer_names_in_group:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (attn_backend.full_cls_name(), layer_kv_cache_spec)

                if key not in attention_groups:
                    attn_group = AttentionGroup(
                        attn_backend,
                        [layer_name],
                        layer_kv_cache_spec,
                        kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(self.vllm_config, self.device)
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

            attention_groups_list.append(attention_groups)

        self.draft_attn_groups = [
            attention_group
            for attention_groups in attention_groups_list
            for attention_group in attention_groups.values()
        ]
        self.kv_cache_gid = 0
        if not self.draft_attn_groups:
            raise RuntimeError(
                "DSpark standard-cache path requires registered draft attention "
                f"groups. Missing layers: {self.attn_layer_names}"
            )

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)

        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names
            if ln in self.attn_layer_names
        }
        self._layer_group_idx = [name_to_gid[name] for name in self.attn_layer_names]

        # some buffers need information of groups
        self._per_group_query_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_query_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        self._per_group_context_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        block_tables = self._decode_only_compact_block_tables or self._per_group_block_tables
        slot_mappings = self._decode_only_compact_slot_mappings or self._per_group_slot_mappings
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: block_tables[attn_group.kv_cache_group_id]
            for attn_group in self.draft_attn_groups
        }
        self._context_slot_mapping_buffers = None
        self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

        token_indices_to_sample = torch.empty(
            num_sample_total,
            dtype=torch.int32,
            device=self.device,
        )

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices.
        draft_attn_groups = getattr(self, "draft_attn_groups", [])
        for attn_group in draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers.get(gid)
            if gid_block_table is None:
                continue
            kv_block_size = int(attn_group.kv_cache_spec.block_size)
            copy_and_expand_dflash_and_dspark_inputs_kernel[
                (_compute_num_programs(self._dflash_num_context, num_query_total),)
            ](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kv_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
            )
        # to compute self._context_slot_mapping_buffers from dict to list
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
        ]

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = self.num_query_per_req
        cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions  # this would be sliced in attention backend
        if hasattr(self.model, "get_draft_attn_causal"):
            # Currently, attention causality across draft layers are uniform.
            cad.causal = self.model.get_draft_attn_causal()[0]
        else:
            cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        num_query_total = num_reqs * self.num_query_per_req
        num_query_tokens = min(num_query_total if num_reqs > 0 else num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=[],
                    num_tokens=num_input_tokens,
                )
