# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import DSparkExecutionConfig, get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.spec_decode.utils import (
    build_dspark_context_slots,
    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid,
)
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

# Allocation errors that may be raised by staging clones. Falling back is
# allowed only because no DSpark KV write has happened yet at that point.
_STAGING_ALLOC_ERRORS = (
    RuntimeError,
    getattr(torch, "OutOfMemoryError", RuntimeError),
)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class DSparkRequestPhase(Enum):
    """Per-request lifecycle of the DSpark decode-only state machine."""

    COLLECTING = auto()
    PENDING_INIT = auto()
    READY = auto()
    FALLBACK_PREFILL = auto()
    INVALID = auto()


@dataclass
class StagedDSparkChunk:
    """One staged prefill chunk of raw target auxiliary hidden states.

    The hidden states are stored *before* ``combine_hidden_states()`` so no
    DSpark computation enters the prefill critical path. Slot mappings are
    intentionally not retained: lazy init rebuilds them from the current
    block tables.
    """

    raw_hidden_states: torch.Tensor
    positions: torch.Tensor
    num_tokens: int
    num_bytes: int
    hidden_row_bytes: int
    position_row_bytes: int


@dataclass
class PendingDSparkContext:
    """Decode-only staging state of a single request (keyed by request_id)."""

    request_id: str
    generation: int
    phase: DSparkRequestPhase = DSparkRequestPhase.COLLECTING
    chunks: list[StagedDSparkChunk] = field(default_factory=list)
    num_staged_tokens: int = 0
    num_staged_bytes: int = 0
    prompt_len: int = 0
    retained_context_tokens: int | None = None
    fallback_reason: str | None = None
    lazy_init_done: bool = False


@dataclass
class DSparkDecodeOnlyRequestMeta:
    """Scheduler-derived per-request step classification (CPU only)."""

    request_rows: list[int]
    request_ids: list[str]
    is_prefill: list[bool]
    finishes_prefill: list[bool]
    prompt_lens: list[int]
    num_computed_tokens: list[int] = field(default_factory=list)


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
        self.sample_from_anchor = not getattr(self.draft_model_config.hf_config, "dspark_bonus_anchor", False)
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
        # Ascend may split a physical KV block into smaller logical/kernel
        # blocks. Slot arithmetic must use the actual per-group kernel size,
        # not KVCacheSpec.block_size.
        self._per_group_kernel_block_sizes: dict[int, int] = {}
        self._per_group_num_kv_cache_blocks: dict[int, int] = {}

        # ---------------- decode-only state ----------------
        self._init_decode_only_state()

    # ------------------------------------------------------------------
    # Decode-only: configuration, request state and lifecycle
    # ------------------------------------------------------------------

    def _resolve_decode_only_exec_config(self) -> DSparkExecutionConfig | None:
        """Resolve the decode-only exec config, never silently dropping it.

        Normal path: the AscendConfig singleton (initialized by the platform
        before any worker/drafter construction). Fallback: parse
        ``additional_config.dspark_config`` directly so a request for
        decode_only is honored (or fails validation loudly) even if the
        singleton is not initialized yet.
        """
        try:
            dspark_config = getattr(get_ascend_config(), "dspark_config", None)
            if dspark_config is not None and dspark_config.decode_only:
                return dspark_config
            return None
        except RuntimeError:
            pass
        vllm_config = getattr(self, "vllm_config", None)
        additional_config = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional_config, dict):
            return None
        if not isinstance(additional_config.get("dspark_config"), dict):
            return None
        return DSparkExecutionConfig(additional_config["dspark_config"], vllm_config)

    def _init_decode_only_state(self) -> None:
        exec_config = self._resolve_decode_only_exec_config()

        self.decode_only = exec_config is not None
        self._dspark_exec_config = exec_config
        # request_id -> staging context
        self._pending_contexts: dict[str, PendingDSparkContext] = {}
        # request_id -> generation counter (bumped on release) so an aborted
        # and immediately resubmitted request never reuses stale state.
        self._request_generations: dict[str, int] = {}
        self._total_staged_bytes = 0
        # Retained context window across all draft KV layers:
        #   None                -> keep the full prompt (any full-attention layer)
        #   max(sliding_window) -> keep only the last N positions
        self._dspark_retained_context_tokens: int | None = None
        # Per-group slot buffers used by lazy init / context-only fallback.
        self._lazy_init_slot_buffers: dict[int, torch.Tensor] | None = None
        # CPU-only counters/gauges; never synchronize with the device.
        self._dspark_stats: dict[str, int] = {
            "pending_requests": 0,
            "staged_tokens": 0,
            "staged_bytes": 0,
            "fallbacks_per_request_tokens": 0,
            "fallbacks_total_bytes": 0,
            "fallbacks_alloc": 0,
            "invalidated_resumed": 0,
            "invalidated_rewind": 0,
            "invalidated_recompute": 0,
        }
        # Per-step valid mask of the last decode-only proposal, aligned with
        # the full persistent batch rows. None means "all rows valid" (or no
        # decode-only proposal has run yet).
        self._last_draft_valid_mask: list[bool] | None = None

        if exec_config is not None:
            logger.info(
                "[dspark/decode_only] enabled: max_staged_tokens_per_request=%d, "
                "max_staged_bytes_total=%d, lazy_init_chunk_tokens=%d, "
                "overflow_policy=%s",
                exec_config.max_staged_tokens_per_request,
                exec_config.max_staged_bytes_total,
                exec_config.lazy_init_chunk_tokens,
                exec_config.overflow_policy,
            )

    def _decode_only_limits(self) -> tuple[int, int, int]:
        cfg = self._dspark_exec_config
        assert cfg is not None
        return (
            cfg.max_staged_tokens_per_request,
            cfg.max_staged_bytes_total,
            cfg.lazy_init_chunk_tokens,
        )

    def take_last_draft_valid_mask(self) -> list[bool] | None:
        """Return the per-row valid mask of the last decode-only proposal."""
        return self._last_draft_valid_mask

    def release_requests(self, request_ids) -> None:
        """Drop all decode-only state for finished/cancelled requests."""
        for req_id in request_ids:
            ctx = self._pending_contexts.pop(req_id, None)
            if ctx is None:
                continue
            self._release_staging(ctx)
            self._request_generations[req_id] = ctx.generation + 1

    def invalidate_requests(self, request_ids, reason: str) -> None:
        """Drop state after preemption/resume/rewind; re-collect from scratch."""
        stats_key = f"invalidated_{reason}"
        for req_id in set(request_ids):
            ctx = self._pending_contexts.pop(req_id, None)
            if ctx is None:
                continue
            self._release_staging(ctx)
            self._request_generations[req_id] = ctx.generation + 1
            self._dspark_stats[stats_key] = self._dspark_stats.get(stats_key, 0) + 1
            logger.warning(
                "[dspark/decode_only] invalidated request %s (generation=%d, reason=%s)",
                req_id,
                ctx.generation,
                reason,
            )

    def _release_staging(self, ctx: PendingDSparkContext) -> None:
        if ctx.chunks:
            ctx.chunks.clear()
        self._total_staged_bytes = max(0, self._total_staged_bytes - ctx.num_staged_bytes)
        ctx.num_staged_tokens = 0
        ctx.num_staged_bytes = 0
        ctx.lazy_init_done = False
        self._refresh_staging_stats()

    def _refresh_staging_stats(self) -> None:
        self._dspark_stats["pending_requests"] = sum(
            1 for c in self._pending_contexts.values() if c.phase == DSparkRequestPhase.PENDING_INIT
        )
        self._dspark_stats["staged_tokens"] = sum(c.num_staged_tokens for c in self._pending_contexts.values())
        self._dspark_stats["staged_bytes"] = self._total_staged_bytes

    def _get_or_create_context(self, request_id: str, prompt_len: int) -> PendingDSparkContext:
        ctx = self._pending_contexts.get(request_id)
        if ctx is not None:
            return ctx
        ctx = PendingDSparkContext(
            request_id=request_id,
            generation=self._request_generations.get(request_id, 0),
            prompt_len=prompt_len,
            retained_context_tokens=self._dspark_retained_context_tokens,
        )
        self._pending_contexts[request_id] = ctx
        return ctx

    # ------------------------------------------------------------------
    # Decode-only: request classification (CPU metadata only, no .item())
    # ------------------------------------------------------------------

    def classify_decode_only_requests(
        self,
        req_ids: list[str],
        num_computed_tokens_cpu,
        num_prompt_tokens,
        num_scheduled_tokens: dict[str, int],
    ) -> DSparkDecodeOnlyRequestMeta:
        num_reqs = len(req_ids)
        is_prefill: list[bool] = []
        finishes_prefill: list[bool] = []
        prompt_lens: list[int] = []
        num_computed_tokens: list[int] = []
        for row in range(num_reqs):
            num_computed_before = int(num_computed_tokens_cpu[row])
            prompt_len = int(num_prompt_tokens[row])
            scheduled = num_scheduled_tokens.get(req_ids[row], 0)
            row_is_prefill = num_computed_before < prompt_len
            is_prefill.append(row_is_prefill)
            finishes_prefill.append(row_is_prefill and num_computed_before + scheduled >= prompt_len)
            prompt_lens.append(prompt_len)
            num_computed_tokens.append(num_computed_before)
        return DSparkDecodeOnlyRequestMeta(
            request_rows=list(range(num_reqs)),
            request_ids=list(req_ids),
            is_prefill=is_prefill,
            finishes_prefill=finishes_prefill,
            prompt_lens=prompt_lens,
            num_computed_tokens=num_computed_tokens,
        )

    def _collect_eligible_rows(self, meta: DSparkDecodeOnlyRequestMeta) -> tuple[list[int], list[int], list[int]]:
        """Rows eligible for a DSpark proposal after staging, by category.

        Returns (pending_first_decode_rows, ready_decode_rows,
        fallback_final_prefill_rows).
        """
        pending_rows: list[int] = []
        ready_rows: list[int] = []
        fallback_final_rows: list[int] = []
        for row, req_id in enumerate(meta.request_ids):
            ctx = self._pending_contexts.get(req_id)
            phase = ctx.phase if ctx is not None else DSparkRequestPhase.COLLECTING
            if phase == DSparkRequestPhase.INVALID:
                raise RuntimeError(
                    f"[dspark/decode_only] request {req_id} is INVALID; "
                    "refusing to continue with corrupted DSpark state."
                )
            if meta.is_prefill[row]:
                if phase == DSparkRequestPhase.FALLBACK_PREFILL and meta.finishes_prefill[row]:
                    fallback_final_rows.append(row)
                continue
            if phase == DSparkRequestPhase.PENDING_INIT:
                pending_rows.append(row)
            elif phase == DSparkRequestPhase.READY:
                ready_rows.append(row)
            elif phase == DSparkRequestPhase.FALLBACK_PREFILL:
                # Decode after a fallback final prefill: the final-prefill
                # proposal already moved it to READY; anything else means the
                # fallback never completed its final prefill.
                raise RuntimeError(
                    f"[dspark/decode_only] request {req_id} is in FALLBACK_PREFILL "
                    "at a decode step; state machine is inconsistent."
                )
            else:
                raise RuntimeError(
                    f"[dspark/decode_only] request {req_id} reached a decode step "
                    f"in phase {phase.name} without any staged prefill context; "
                    "DSpark decode-only requires the prefill to be observed."
                )
        return pending_rows, ready_rows, fallback_final_rows

    # ------------------------------------------------------------------
    # Decode-only: prefill staging (no DSpark model op is allowed here)
    # ------------------------------------------------------------------

    def stage_prefill_context(
        self,
        meta: DSparkDecodeOnlyRequestMeta,
        raw_target_hidden_states: torch.Tensor | None,
        target_positions: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        raw_aux_hidden_states: list[torch.Tensor] | None = None,
    ) -> None:
        """Stage raw target auxiliary hidden states for every prefill row.

        Must not call ``combine_hidden_states()``,
        ``precompute_and_store_context_kv()``, the draft forward or any
        draft head: those all belong to the lazy-init / proposal phases.
        """
        qsl = query_start_loc_cpu
        if self._dspark_retained_context_tokens is not None:
            logger.info_once(
                "[dspark/decode_only] suffix-only staging active: "
                "retained_context_tokens=%d; earlier prefill chunks are not copied",
                self._dspark_retained_context_tokens,
            )
        for row, req_id in enumerate(meta.request_ids):
            if not meta.is_prefill[row]:
                continue
            ctx = self._get_or_create_context(req_id, prompt_len=meta.prompt_lens[row])
            ctx.prompt_len = meta.prompt_lens[row]
            if ctx.phase in (DSparkRequestPhase.PENDING_INIT, DSparkRequestPhase.READY):
                # Prefill re-scheduled for an already-initialized request
                # without a resume/rewind hook: treat as recompute and
                # re-collect from scratch.
                previous_phase = ctx.phase
                self._release_staging(ctx)
                ctx.phase = DSparkRequestPhase.COLLECTING
                ctx.fallback_reason = None
                self._dspark_stats["invalidated_recompute"] += 1
                logger.warning(
                    "[dspark/decode_only] request %s re-entered prefill in phase %s; resetting to COLLECTING",
                    req_id,
                    previous_phase.name,
                )
            if ctx.phase == DSparkRequestPhase.INVALID:
                raise RuntimeError(f"[dspark/decode_only] request {req_id} is INVALID; refusing to stage.")

            start = int(qsl[row])
            end = int(qsl[row + 1])

            if ctx.phase == DSparkRequestPhase.FALLBACK_PREFILL:
                # Already fell back: project every live chunk directly, no
                # retained-suffix filtering is allowed in this mode.
                raw_slice = self._materialize_raw_hidden(
                    raw_target_hidden_states,
                    raw_aux_hidden_states,
                    start,
                    end,
                )
                self._project_live_chunk(ctx, row, raw_slice, target_positions[start:end])
                if meta.finishes_prefill[row]:
                    pass  # eligibility handled by _collect_eligible_rows
                continue

            # When all draft attention groups are sliding-window, only the
            # suffix can ever be read by DSpark. Avoid materializing/cloning
            # earlier prefill chunks at all; the previous implementation
            # cloned the full chunk and discarded it in _trim_staged_prefix.
            keep = ctx.retained_context_tokens
            computed_before = (
                meta.num_computed_tokens[row] if len(meta.num_computed_tokens) > row else 0
            )
            retain_start = max(0, ctx.prompt_len - keep) if keep is not None else computed_before
            local_offset = max(0, retain_start - computed_before)
            if keep is not None and local_offset >= end - start:
                if meta.finishes_prefill[row]:
                    ctx.phase = DSparkRequestPhase.PENDING_INIT
                    self._refresh_staging_stats()
                continue

            stage_start = start + local_offset
            pos_slice = target_positions[stage_start:end]

            try:
                hidden = self._materialize_raw_hidden(
                    raw_target_hidden_states,
                    raw_aux_hidden_states,
                    stage_start,
                    end,
                )
                positions = pos_slice.detach().to(torch.int32).clone()
            except _STAGING_ALLOC_ERRORS:
                self._dspark_stats["fallbacks_alloc"] += 1
                logger.warning(
                    "[dspark/decode_only] staging allocation failed for request "
                    "%s; falling back to prefill_tail for this request.",
                    req_id,
                )
                raw_slice = self._materialize_raw_hidden(
                    raw_target_hidden_states,
                    raw_aux_hidden_states,
                    start,
                    end,
                )
                self._fallback_prefill_tail(
                    ctx,
                    row,
                    live_raw=raw_slice,
                    live_pos=target_positions[start:end],
                )
                continue

            chunk = StagedDSparkChunk(
                raw_hidden_states=hidden,
                positions=positions,
                num_tokens=hidden.shape[0],
                num_bytes=_tensor_bytes(hidden) + _tensor_bytes(positions),
                hidden_row_bytes=hidden[0].numel() * hidden.element_size() if hidden.shape[0] else 0,
                position_row_bytes=positions[0].numel() * positions.element_size() if positions.shape[0] else 0,
            )
            ctx.chunks.append(chunk)
            ctx.num_staged_tokens += chunk.num_tokens
            ctx.num_staged_bytes += chunk.num_bytes
            self._total_staged_bytes += chunk.num_bytes

            self._trim_staged_prefix(ctx)
            if self._staging_limits_exceeded(ctx):
                self._fallback_prefill_tail(ctx, row, live_raw=None, live_pos=None)
                continue

            if meta.finishes_prefill[row]:
                ctx.phase = DSparkRequestPhase.PENDING_INIT
                self._refresh_staging_stats()

    @staticmethod
    def _materialize_raw_hidden(
        raw_target_hidden_states: torch.Tensor | None,
        raw_aux_hidden_states: list[torch.Tensor] | None,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Build an owned raw-hidden slice without copying unused prefill rows."""
        if raw_aux_hidden_states is not None:
            if len(raw_aux_hidden_states) == 1:
                return raw_aux_hidden_states[0][start:end].detach().clone()
            # torch.cat allocates independent storage, so an additional clone
            # would duplicate the entire selected slice unnecessarily.
            return torch.cat([hidden[start:end] for hidden in raw_aux_hidden_states], dim=-1).detach()
        if raw_target_hidden_states is None:
            raise RuntimeError("DSpark decode-only requires raw auxiliary hidden states.")
        return raw_target_hidden_states[start:end].detach().clone()

    def _trim_staged_prefix(self, ctx: PendingDSparkContext) -> None:
        """Keep only the last ``retained_context_tokens`` staged positions."""
        keep = ctx.retained_context_tokens
        if keep is None or ctx.num_staged_tokens <= keep:
            return
        dropped_rows = 0
        dropped_bytes = 0
        while ctx.chunks and ctx.num_staged_tokens - ctx.chunks[0].num_tokens >= keep:
            chunk = ctx.chunks.pop(0)
            ctx.num_staged_tokens -= chunk.num_tokens
            ctx.num_staged_bytes -= chunk.num_bytes
            dropped_rows += chunk.num_tokens
            dropped_bytes += chunk.num_bytes
        overflow = ctx.num_staged_tokens - keep
        if overflow > 0 and ctx.chunks:
            head = ctx.chunks[0]
            # Slice at token granularity; the narrow view keeps the original
            # storage but accounting uses the retained rows only.
            ctx.chunks[0] = StagedDSparkChunk(
                raw_hidden_states=head.raw_hidden_states[overflow:],
                positions=head.positions[overflow:],
                num_tokens=head.num_tokens - overflow,
                num_bytes=head.num_bytes - overflow * (head.hidden_row_bytes + head.position_row_bytes),
                hidden_row_bytes=head.hidden_row_bytes,
                position_row_bytes=head.position_row_bytes,
            )
            ctx.num_staged_tokens -= overflow
            head_row_bytes = head.hidden_row_bytes + head.position_row_bytes
            ctx.num_staged_bytes -= overflow * head_row_bytes
            dropped_rows += overflow
            dropped_bytes += overflow * head_row_bytes
        self._total_staged_bytes = max(0, self._total_staged_bytes - dropped_bytes)

    def _staging_limits_exceeded(self, ctx: PendingDSparkContext) -> bool:
        max_tokens, max_bytes_total, _ = self._decode_only_limits()
        if ctx.num_staged_tokens > max_tokens:
            ctx.fallback_reason = "per_request_tokens"
            self._dspark_stats["fallbacks_per_request_tokens"] += 1
            return True
        if self._total_staged_bytes > max_bytes_total:
            ctx.fallback_reason = "total_bytes"
            self._dspark_stats["fallbacks_total_bytes"] += 1
            return True
        return False

    def _fallback_prefill_tail(
        self,
        ctx: PendingDSparkContext,
        request_row: int,
        live_raw: torch.Tensor | None,
        live_pos: torch.Tensor | None,
    ) -> None:
        """Per-request overflow fallback: move DSpark context work back to prefill.

        Projects everything staged so far (plus the chunk that triggered the
        overflow) into the DSpark KV cache, releases staging and switches the
        request to ``FALLBACK_PREFILL`` so later chunks write incrementally.
        """
        with record_function_or_nullcontext("dspark_context_only_fallback"):
            if live_raw is not None and live_pos is not None:
                hidden = live_raw.detach().clone()
                positions = live_pos.detach().to(torch.int32).clone()
                ctx.chunks.append(
                    StagedDSparkChunk(
                        raw_hidden_states=hidden,
                        positions=positions,
                        num_tokens=hidden.shape[0],
                        num_bytes=_tensor_bytes(hidden) + _tensor_bytes(positions),
                        hidden_row_bytes=(hidden[0].numel() * hidden.element_size() if hidden.shape[0] else 0),
                        position_row_bytes=(
                            positions[0].numel() * positions.element_size() if positions.shape[0] else 0
                        ),
                    )
                )
                ctx.num_staged_tokens += hidden.shape[0]
                ctx.num_staged_bytes += _tensor_bytes(hidden) + _tensor_bytes(positions)
                self._total_staged_bytes += _tensor_bytes(hidden) + _tensor_bytes(positions)
            reason = ctx.fallback_reason or "overflow"
            max_tokens, max_bytes_total, _ = self._decode_only_limits()
            # Keep this as one fully-rendered line.  Besides identifying the
            # violated limit, it makes it explicit that context projection is
            # being put back on the prefill critical path for this request.
            logger.warning(
                f"[dspark/decode_only] fallback_prefill_tail=true "
                f"context_projection_in_prefill=true request={ctx.request_id} "
                f"reason={reason} prompt_tokens={ctx.prompt_len} "
                f"staged_tokens={ctx.num_staged_tokens} "
                f"max_staged_tokens_per_request={max_tokens} "
                f"staged_bytes={ctx.num_staged_bytes} "
                f"total_staged_bytes={self._total_staged_bytes} "
                f"max_staged_bytes_total={max_bytes_total}"
            )
            self._project_staged_contexts([ctx], [request_row])
            self._release_staging(ctx)
            ctx.phase = DSparkRequestPhase.FALLBACK_PREFILL
            ctx.fallback_reason = reason

    def _project_live_chunk(
        self,
        ctx: PendingDSparkContext,
        request_row: int,
        raw: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Context-only projection of one live chunk for FALLBACK_PREFILL.

        A prefill chunk may be larger than ``lazy_init_chunk_tokens``; split
        it so every projection pack fits the preallocated slot buffers.
        """
        del ctx
        with record_function_or_nullcontext("dspark_context_only_fallback"):
            _, _, chunk_tokens = self._decode_only_limits()
            num_tokens = raw.shape[0]
            offset = 0
            while offset < num_tokens:
                take = min(chunk_tokens, num_tokens - offset)
                qsl = torch.tensor([0, take], dtype=torch.int32)
                row_map = torch.tensor([request_row], dtype=torch.int32)
                self._project_context_kv(
                    raw[offset : offset + take].detach().clone(),
                    positions[offset : offset + take].detach().to(torch.int32).clone(),
                    row_map,
                    qsl,
                )
                offset += take

    # ------------------------------------------------------------------
    # Decode-only: lazy init and context slot rebuild
    # ------------------------------------------------------------------

    def initialize_pending_contexts(
        self,
        request_ids: list[str],
        full_batch_rows: list[int],
    ) -> None:
        """Project staged prompt context into the current DSpark KV blocks.

        Called after the first ordinary decode's target forward. Staged
        prompt hidden states are combined, their context slots are rebuilt
        from the *current* per-group block tables, and
        ``precompute_and_store_context_kv`` writes them in
        ``lazy_init_chunk_tokens``-sized packs.
        """
        contexts = [self._pending_contexts[req_id] for req_id in request_ids]
        try:
            self._project_staged_contexts(contexts, full_batch_rows)
        except Exception as exc:
            for ctx in contexts:
                ctx.phase = DSparkRequestPhase.INVALID
                self._release_staging(ctx)
            raise RuntimeError(
                "[dspark/decode_only] lazy init failed midway for requests "
                f"{request_ids}; DSpark KV state is unrecoverable for them."
            ) from exc
        for ctx in contexts:
            ctx.lazy_init_done = True

    def _project_staged_contexts(
        self,
        contexts: list[PendingDSparkContext],
        full_batch_rows: list[int],
    ) -> None:
        _, _, chunk_tokens = self._decode_only_limits()
        # Each segment: (full_batch_row, hidden_view, position_view, num_tokens)
        segments: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal segments, pending_tokens
            if not segments:
                return
            raw = torch.cat([seg[1] for seg in segments], dim=0)
            pos = torch.cat([seg[2] for seg in segments], dim=0)
            qsl_values = [0]
            row_values: list[int] = []
            for seg_row, _, _, seg_tokens in segments:
                row_values.append(seg_row)
                qsl_values.append(qsl_values[-1] + seg_tokens)
            row_map = torch.tensor(row_values, dtype=torch.int32)
            qsl = torch.tensor(qsl_values, dtype=torch.int32)
            self._project_context_kv(raw, pos, row_map, qsl)
            segments = []
            pending_tokens = 0

        for ctx, row in zip(contexts, full_batch_rows):
            for chunk in ctx.chunks:
                offset = 0
                remaining = chunk.num_tokens
                while remaining > 0:
                    if pending_tokens >= chunk_tokens:
                        flush()
                    take = min(remaining, chunk_tokens - pending_tokens)
                    segments.append(
                        (
                            row,
                            chunk.raw_hidden_states[offset : offset + take],
                            chunk.positions[offset : offset + take],
                            take,
                        )
                    )
                    pending_tokens += take
                    offset += take
                    remaining -= take
        flush()

    def _ensure_draft_fused_buffers(self) -> None:
        """Ensure the draft model's fused context-KV buffer metadata is valid.

        The fused-projection buffers (``_num_attn_layers`` etc.) are normally
        built during weight loading or on the first prefill-side
        ``precompute_and_store_context_kv`` call. Decode-only defers that
        first call to lazy init, so a missing or invalid (e.g. -1 sentinel)
        value must be rebuilt here instead of crashing inside the projection.
        """
        draft_inner = getattr(self.model, "model", None)
        if draft_inner is None:
            return
        num_attn_layers = getattr(draft_inner, "_num_attn_layers", None)
        if num_attn_layers is not None and num_attn_layers > 0:
            return
        builder = getattr(draft_inner, "_build_fused_kv_buffers", None)
        if callable(builder):
            logger.info(
                "[dspark/decode_only] draft fused context-KV buffers missing "
                "or invalid (_num_attn_layers=%r); rebuilding now.",
                num_attn_layers,
            )
            builder()

    def _project_context_kv(
        self,
        raw_hidden: torch.Tensor,
        positions: torch.Tensor,
        req_row_map_cpu: torch.Tensor,
        req_start_loc_cpu: torch.Tensor,
    ) -> None:
        """Combine raw aux hidden and write DSpark context KV for one pack."""
        num_tokens = raw_hidden.shape[0]
        if num_tokens == 0:
            return
        self._ensure_draft_fused_buffers()
        if positions.dim() != 1:
            raise RuntimeError(
                "[dspark/decode_only] lazy-init expects flat 1-D token "
                "positions, got shape {} (mrope/xdrope layout leaked into "
                "staging); this is a proposer bug.".format(tuple(positions.shape))
            )
        combined = self.model.combine_hidden_states(raw_hidden)
        if combined.shape[0] != positions.shape[0]:
            raise RuntimeError(
                "[dspark/decode_only] lazy-init pack shape mismatch: "
                "combined hidden rows={} vs positions rows={} "
                "(raw_hidden.shape={}, positions.shape={})".format(
                    combined.shape[0],
                    positions.shape[0],
                    tuple(raw_hidden.shape),
                    tuple(positions.shape),
                )
            )
        draft_inner = getattr(self.model, "model", None)
        num_attn_layers = getattr(draft_inner, "_num_attn_layers", None)
        if num_attn_layers is not None and num_attn_layers <= 0:
            raise RuntimeError(
                "[dspark/decode_only] draft fused context-KV buffers are "
                "invalid (_num_attn_layers={}, num_ctx={}): the draft model "
                "reports no attention layers; refusing to project context KV.".format(num_attn_layers, num_tokens)
            )
        req_start_loc_gpu = req_start_loc_cpu.to(self.device, non_blocking=True)
        req_row_map_gpu = req_row_map_cpu.to(self.device, non_blocking=True)
        per_group_slots: dict[int, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            block_table = self._per_group_block_tables.get(gid)
            if block_table is None:
                raise RuntimeError(
                    f"[dspark/decode_only] missing current block table for draft "
                    f"KV group {gid}; cannot rebuild context slots."
                )
            out_slots = self._lazy_init_slot_buffers[gid][:num_tokens]
            if out_slots.shape[0] < num_tokens:
                raise RuntimeError(
                    "[dspark/decode_only] lazy-init pack ({} tokens) exceeds the "
                    "preallocated slot buffer ({}); check lazy_init_chunk_tokens.".format(
                        num_tokens, self._lazy_init_slot_buffers[gid].shape[0]
                    )
                )
            build_dspark_context_slots(
                positions=positions,
                req_row_map=req_row_map_gpu,
                req_start_loc=req_start_loc_gpu,
                block_table=block_table,
                block_size=self._per_group_kernel_block_sizes.get(
                    gid, int(attn_group.kv_cache_spec.block_size)
                ),
                num_kv_cache_blocks=self._per_group_num_kv_cache_blocks.get(
                    gid, torch.iinfo(torch.int32).max
                ),
                out_slots=out_slots,
            )
            self._debug_validate_context_slots(
                gid=gid,
                block_table=block_table,
                block_size=self._per_group_kernel_block_sizes.get(
                    gid, int(attn_group.kv_cache_spec.block_size)
                ),
                positions=positions,
                req_row_map_cpu=req_row_map_cpu,
                out_slots=out_slots,
            )
            per_group_slots[gid] = out_slots
        slots_by_layer = [per_group_slots[gidx] for gidx in self._layer_group_idx]
        self.model.precompute_and_store_context_kv(combined, positions, slots_by_layer)

    def _debug_validate_context_slots(
        self,
        gid: int,
        block_table: torch.Tensor,
        block_size: int,
        positions: torch.Tensor,
        req_row_map_cpu: torch.Tensor,
        out_slots: torch.Tensor,
    ) -> None:
        """Debug-only validation of rebuilt context slots (env-gated).

        Enabled with VLLM_ASCEND_DSPARK_DEBUG=1. Intentionally synchronizes
        with the device: it turns an async device-side fault (aivec 'MTE
        accesses an invalid GM address') into an immediate Python error that
        carries the offending values, so a bad block table / block size is
        diagnosable from the engine log instead of a device reset.
        """
        import os

        if os.getenv("VLLM_ASCEND_DSPARK_DEBUG", "0") != "1":
            return
        max_pos = int(positions.max().item())
        min_pos = int(positions.min().item())
        max_block_num = max_pos // block_size
        if max_block_num >= block_table.shape[1]:
            raise RuntimeError(
                "[dspark/decode_only][debug] position {} needs block column {} "
                "but group {} block table has only {} columns "
                "(block_size={}, block_table.shape={}, rows={}).".format(
                    max_pos,
                    max_block_num,
                    gid,
                    block_table.shape[1],
                    block_size,
                    tuple(block_table.shape),
                    req_row_map_cpu.tolist(),
                )
            )
        rows = req_row_map_cpu.to(torch.int64)
        needed_blocks = block_table[rows, : max_block_num + 1]
        bad_mask = needed_blocks < 0
        slots_min = int(out_slots.min().item())
        slots_max = int(out_slots.max().item())
        if bool(bad_mask.any().item()):
            raise RuntimeError(
                "[dspark/decode_only][debug] group {} block table contains "
                "invalid (negative) block ids for required columns 0..{}: "
                "{}; positions=[{}, {}], block_size={}, rows={}.".format(
                    gid,
                    max_block_num,
                    needed_blocks.tolist(),
                    min_pos,
                    max_pos,
                    block_size,
                    req_row_map_cpu.tolist(),
                )
            )
        logger.warning(
            "[dspark/decode_only][debug] gid=%d block_size=%d positions=[%d, %d] "
            "needed_blocks(row0)=%s slots=[%d, %d] (max block id=%d)",
            gid,
            block_size,
            min_pos,
            max_pos,
            needed_blocks[0].tolist() if needed_blocks.shape[0] else [],
            slots_min,
            slots_max,
            int(needed_blocks.max().item()),
        )

    def _mark_rows_proposed(
        self,
        meta: DSparkDecodeOnlyRequestMeta,
        pending_rows: list[int],
        fallback_final_rows: list[int],
    ) -> None:
        """Atomically switch requests to READY after a successful proposal."""
        for row in pending_rows:
            ctx = self._pending_contexts.get(meta.request_ids[row])
            if ctx is None or ctx.phase != DSparkRequestPhase.PENDING_INIT:
                continue
            self._release_staging(ctx)
            ctx.phase = DSparkRequestPhase.READY
        for row in fallback_final_rows:
            ctx = self._pending_contexts.get(meta.request_ids[row])
            if ctx is None or ctx.phase != DSparkRequestPhase.FALLBACK_PREFILL:
                continue
            ctx.phase = DSparkRequestPhase.READY
        self._refresh_staging_stats()

    # ------------------------------------------------------------------
    # Decode-only: orchestration entry point
    # ------------------------------------------------------------------

    def propose_decode_only(
        self,
        *,
        scheduler_output,
        common_attn_metadata: CommonAttentionMetadata,
        token_indices: torch.Tensor | None,
        token_indices_to_sample: torch.Tensor | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
        next_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        raw_target_hidden_states: torch.Tensor | None,
        num_scheduled_tokens: dict[str, int],
        req_ids: list[str],
        num_computed_tokens_cpu,
        num_prompt_tokens,
        target_model_batch_desc,
        sampling_metadata,
        raw_aux_hidden_states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """DSpark decode-only orchestration: stage, lazy-init, propose, scatter."""
        with record_function_or_nullcontext("dspark_decode_proposal"):
            return self._propose_decode_only_impl(
                scheduler_output=scheduler_output,
                common_attn_metadata=common_attn_metadata,
                token_indices=token_indices,
                token_indices_to_sample=token_indices_to_sample,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                next_token_ids=next_token_ids,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                raw_target_hidden_states=raw_target_hidden_states,
                raw_aux_hidden_states=raw_aux_hidden_states,
                num_scheduled_tokens=num_scheduled_tokens,
                req_ids=req_ids,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_prompt_tokens=num_prompt_tokens,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
            )

    @staticmethod
    def _flatten_target_positions(target_positions: torch.Tensor) -> torch.Tensor:
        """Flatten [rope_dim, num_tokens] (mrope/xdrope) positions to 1-D.

        DSpark context staging / lazy init / slot rebuild index positions by
        token; for 2-D layouts the first rope dim is the token position, the
        same row AscendSpecDecodeBaseProposer.set_inputs_first_pass feeds to
        the draft model.
        """
        if target_positions.dim() > 1:
            return target_positions[0]
        return target_positions

    def _propose_decode_only_impl(
        self,
        *,
        scheduler_output,
        common_attn_metadata: CommonAttentionMetadata,
        token_indices: torch.Tensor | None,
        token_indices_to_sample: torch.Tensor | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
        next_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        raw_target_hidden_states: torch.Tensor | None,
        raw_aux_hidden_states: list[torch.Tensor] | None,
        num_scheduled_tokens: dict[str, int],
        req_ids: list[str],
        num_computed_tokens_cpu,
        num_prompt_tokens,
        target_model_batch_desc,
        sampling_metadata,
    ) -> torch.Tensor | None:
        del token_indices  # target tensors are already gathered by the runner.
        num_reqs = len(req_ids)
        meta = self.classify_decode_only_requests(
            req_ids, num_computed_tokens_cpu, num_prompt_tokens, num_scheduled_tokens
        )
        # mrope/xdrope targets hand positions as [rope_dim, num_tokens].
        # The DFlash/DSpark first-pass kernel indexes positions by token and
        # has no 2-D layout support: a [3, N] tensor would interleave rope
        # coords into the context/query positions (observed as ~2% draft
        # acceptance and eventual device-side faults). Flatten to the first
        # rope dim — the same row AscendSpecDecodeBaseProposer's EAGLE path
        # feeds the draft — for staging, lazy init AND _propose.
        flat_positions = self._flatten_target_positions(target_positions)

        if any(meta.is_prefill):
            with record_function_or_nullcontext("dspark_stage_context"):
                self.stage_prefill_context(
                    meta,
                    raw_target_hidden_states,
                    flat_positions,
                    common_attn_metadata.query_start_loc_cpu,
                    raw_aux_hidden_states=raw_aux_hidden_states,
                )

        pending_rows, ready_rows, fallback_final_rows = self._collect_eligible_rows(meta)
        if pending_rows:
            with record_function_or_nullcontext("dspark_lazy_init"):
                self.initialize_pending_contexts([req_ids[row] for row in pending_rows], pending_rows)

        eligible_rows = sorted(pending_rows + ready_rows + fallback_final_rows)
        valid_mask = [False] * num_reqs

        if not eligible_rows:
            # Pure prefill: raw auxiliary hidden states have been staged, but
            # no request is eligible for a draft proposal yet. Returning None
            # lets ModelRunner skip the draft-token D2H/event path entirely;
            # the target sampled token can be returned immediately. A
            # fallback final-prefill row is included in eligible_rows above,
            # so fallback_prefill_tail keeps its historical proposal behavior.
            self._last_draft_valid_mask = valid_mask
            logger.info_once(
                "[dspark/decode_only] prefill-only path active: DSpark proposal "
                "is skipped before the first decode"
            )
            logger.debug(
                "[dspark/decode_only] prefill-only step: skipped DSpark proposal "
                "(num_reqs=%d, scheduled_tokens=%d)",
                num_reqs,
                sum(num_scheduled_tokens.values()),
            )
            return None

        full_drafts = torch.zeros((num_reqs, self.num_speculative_tokens), dtype=torch.int64, device=self.device)
        if raw_target_hidden_states is None:
            if raw_aux_hidden_states is None:
                raise RuntimeError("DSpark decode-only proposal has no raw hidden states.")
            raw_target_hidden_states = torch.cat(
                [hidden[: target_positions.shape[-1]] for hidden in raw_aux_hidden_states], dim=-1
            ).detach()
        prefill_rows = sum(meta.is_prefill)
        if prefill_rows:
            logger.info_once(
                "[dspark/decode_only] mixed-batch path active: DSpark proposal "
                "is limited to decode-eligible rows"
            )
            logger.debug(
                "[dspark/decode_only] mixed batch: skipped DSpark proposal "
                "for prefill rows (prefill_rows=%d, proposal_rows=%d)",
                prefill_rows,
                len(eligible_rows),
            )
        sub_drafts = self._propose_compact(
            eligible_rows=eligible_rows,
            common_attn_metadata=common_attn_metadata,
            token_indices_to_sample=token_indices_to_sample,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            next_token_ids=next_token_ids,
            target_token_ids=target_token_ids,
            target_positions=flat_positions,
            raw_target_hidden_states=raw_target_hidden_states,
            target_model_batch_desc=target_model_batch_desc,
            sampling_metadata=sampling_metadata,
            scheduler_output=scheduler_output,
        )
        if len(eligible_rows) == num_reqs:
            full_drafts = sub_drafts
        else:
            rows_gpu = torch.tensor(eligible_rows, dtype=torch.int64, device=self.device)
            full_drafts.index_copy_(0, rows_gpu, sub_drafts)
        for row in eligible_rows:
            valid_mask[row] = True
        self._mark_rows_proposed(meta, pending_rows, fallback_final_rows)

        self._last_draft_valid_mask = valid_mask
        return full_drafts

    def _propose_compact(
        self,
        *,
        eligible_rows: list[int],
        common_attn_metadata: CommonAttentionMetadata,
        token_indices_to_sample: torch.Tensor | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
        next_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        raw_target_hidden_states: torch.Tensor,
        target_model_batch_desc,
        sampling_metadata,
        scheduler_output,
    ) -> torch.Tensor:
        """Run the existing ``_propose`` path on the eligible subbatch."""
        num_reqs = common_attn_metadata.num_reqs
        identity = eligible_rows == list(range(num_reqs))
        if identity:
            return self._propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=raw_target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
                req_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )

        m = len(eligible_rows)
        qsl_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].to(torch.int32)
        qsl_np = qsl_cpu.numpy()
        eligible_np = np.asarray(eligible_rows, dtype=np.int64)
        counts_np = (qsl_np[eligible_np + 1] - qsl_np[eligible_np]).astype(np.int32)
        total_tokens = int(counts_np.sum())
        new_qsl_np = np.zeros(m + 1, dtype=np.int32)
        np.cumsum(counts_np, out=new_qsl_np[1:])
        # Compact-token index: old flat index of every retained token.
        old_starts = np.repeat(qsl_np[eligible_np], counts_np)
        new_starts = np.repeat(new_qsl_np[:-1], counts_np)
        offsets = np.arange(total_tokens, dtype=np.int32) - new_starts
        compact_idx_np = old_starts + offsets

        compact_idx_gpu = torch.from_numpy(compact_idx_np).to(self.device, non_blocking=True)
        compact_idx_long = compact_idx_gpu.to(torch.int64)
        rows_gpu = torch.from_numpy(eligible_np).to(self.device, non_blocking=True)

        sub_token_ids = target_token_ids.index_select(0, compact_idx_long)
        # The caller flattens mrope/xdrope positions to 1-D before _propose
        # (the first-pass kernel has no 2-D layout support); flatten again
        # defensively so this helper is safe for any input layout.
        sub_positions = self._flatten_target_positions(target_positions).index_select(0, compact_idx_long)
        sub_hidden = raw_target_hidden_states.index_select(0, compact_idx_long)
        sub_next = next_token_ids.index_select(0, rows_gpu)
        sub_rejected = (
            num_rejected_tokens_gpu.index_select(0, rows_gpu) if num_rejected_tokens_gpu is not None else None
        )
        sub_tits = token_indices_to_sample.index_select(0, rows_gpu) if token_indices_to_sample is not None else None

        cad_sub = copy.copy(common_attn_metadata)
        cad_sub.num_reqs = m
        new_qsl_cpu = torch.from_numpy(new_qsl_np)
        cad_sub.query_start_loc_cpu = new_qsl_cpu
        cad_sub.query_start_loc = new_qsl_cpu.to(self.device, non_blocking=True)
        if getattr(cad_sub, "seq_lens", None) is not None:
            cad_sub.seq_lens = common_attn_metadata.seq_lens.index_select(0, rows_gpu)
        if getattr(cad_sub, "block_table_tensor", None) is not None:
            cad_sub.block_table_tensor = common_attn_metadata.block_table_tensor.index_select(0, rows_gpu)
        if getattr(cad_sub, "num_computed_tokens_cpu", None) is not None:
            cad_sub.num_computed_tokens_cpu = common_attn_metadata.num_computed_tokens_cpu[eligible_np]

        # Per-group runner tensors must follow the compact layout: the DSpark
        # first-pass kernel indexes them by subbatch row / compact token.
        saved_block_tables = self._per_group_block_tables
        saved_slot_mappings = self._per_group_slot_mappings
        try:
            self._per_group_block_tables = {gid: bt.index_select(0, rows_gpu) for gid, bt in saved_block_tables.items()}
            self._per_group_slot_mappings = {
                gid: sm.index_select(0, compact_idx_gpu) for gid, sm in saved_slot_mappings.items()
            }
            return self._propose(
                target_token_ids=sub_token_ids,
                target_positions=sub_positions,
                target_hidden_states=sub_hidden,
                next_token_ids=sub_next,
                token_indices_to_sample=sub_tits,
                common_attn_metadata=cad_sub,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
                req_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                num_rejected_tokens_gpu=sub_rejected,
            )
        finally:
            self._per_group_block_tables = saved_block_tables
            self._per_group_slot_mappings = saved_slot_mappings

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

        # ModelRunner computes the authoritative per-group kernel block sizes
        # before recreating its NPUInputBatch. Resolve from that list first;
        # the argument is retained as a compatibility fallback for older
        # callers, and the KV spec is the final fallback for unsplit blocks.
        runner_kernel_block_sizes = getattr(self.runner, "kernel_block_sizes", None)

        def resolve_kernel_block_size(gid: int, spec) -> int:
            for sizes in (runner_kernel_block_sizes, kernel_block_sizes):
                if sizes is None:
                    continue
                if isinstance(sizes, int):
                    candidate = sizes if gid == 0 else 0
                elif gid < len(sizes):
                    candidate = sizes[gid]
                else:
                    continue
                if isinstance(candidate, (list, tuple)):
                    candidate = candidate[0] if candidate else 0
                if isinstance(candidate, int) and candidate > 0:
                    return int(candidate)
            return int(spec.block_size)

        configured_num_blocks = getattr(kv_cache_config, "num_blocks", None)
        if configured_num_blocks is not None:
            configured_num_blocks = int(configured_num_blocks)
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            spec_block_size = int(attn_group.kv_cache_spec.block_size)
            kernel_block_size = resolve_kernel_block_size(gid, attn_group.kv_cache_spec)
            if spec_block_size % kernel_block_size != 0:
                raise RuntimeError(
                    "DSpark draft KV block size must be divisible by the Ascend "
                    f"kernel block size: gid={gid}, spec_block_size={spec_block_size}, "
                    f"kernel_block_size={kernel_block_size}."
                )
            blocks_per_physical_block = spec_block_size // kernel_block_size
            self._per_group_kernel_block_sizes[gid] = kernel_block_size
            # BlockTable expands physical IDs into logical IDs when virtual
            # kernel blocks are used, so include that expansion in the bound.
            self._per_group_num_kv_cache_blocks[gid] = (
                configured_num_blocks * blocks_per_physical_block
                if configured_num_blocks is not None
                else torch.iinfo(torch.int32).max
            )

        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]

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

        if self.decode_only:
            self._init_decode_only_attn_state()

    def _init_decode_only_attn_state(self) -> None:
        """Derive the retained-context window and allocate lazy-init buffers.

        Runs after ``load_model`` and after the draft attention groups are
        registered, so draft model capabilities can be verified here.
        """
        for method_name in ("combine_hidden_states", "precompute_and_store_context_kv"):
            if not callable(getattr(self.model, method_name, None)):
                raise RuntimeError(
                    "DSpark decode_only requires the draft model to expose "
                    f"`{method_name}()`; the loaded draft model does not."
                )
        # Validate (and rebuild) the fused context-KV buffers now: decode-only
        # defers the first precompute call to lazy init, so an invalid value
        # must surface at startup rather than mid-serving.
        self._ensure_draft_fused_buffers()
        _, _, chunk_tokens = self._decode_only_limits()
        windows: list[int] = []
        has_full_attention = False
        for attn_group in self.draft_attn_groups:
            spec = attn_group.kv_cache_spec
            sliding_window = getattr(spec, "sliding_window", None)
            if isinstance(sliding_window, int) and sliding_window > 0:
                windows.append(sliding_window)
            else:
                has_full_attention = True
        if has_full_attention:
            self._dspark_retained_context_tokens = None
        else:
            self._dspark_retained_context_tokens = max(windows)
        self._lazy_init_slot_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(chunk_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        logger.info(
            "[dspark/decode_only] retained_context_tokens=%s",
            self._dspark_retained_context_tokens,
        )

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
        # The initial input token of markovHead is the next token.
        # mrope/xdrope targets hand positions as [rope_dim, num_tokens];
        # the first-pass kernel indexes positions by token, so flatten to
        # the first rope dim (mirrors AscendDflashProposer and the EAGLE
        # default path). Decode-only flattens earlier; this covers the
        # prefill_tail path.
        if target_positions.dim() > 1:
            target_positions = target_positions[0]
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
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
            kv_block_size = self._per_group_kernel_block_sizes.get(
                gid, int(attn_group.kv_cache_spec.block_size)
            )
            if batch_size > 0:
                copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[batch_size,](
                    # Inputs
                    next_token_ids_ptr=next_token_ids,
                    target_positions_ptr=target_positions,
                    context_slot_mapping_ptr=self._per_group_slot_mappings[gid],
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
                    num_kv_cache_blocks=self._per_group_num_kv_cache_blocks.get(
                        gid, torch.iinfo(torch.int32).max
                    ),
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
