#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for the dspark speculative-decoding proposer."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.spec_decode.dspark_proposer import (
    AscendDSparkProposer,
    DSparkDecodeOnlyRequestMeta,
    DSparkRequestPhase,
)
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# 0 = single-DP (no padding); >0 = multi-DP where num_input_tokens >
# num_query_total, the out-of-bounds regime.
MULTI_DP_PADDING_SIZES = [0, 8, 32]
_NUM_SPECULATIVE_TOKENS = 3
_MAX_BATCH_SIZE = 2
_MAX_NUM_TOKENS = 8
_HIDDEN_SIZE = 16


class _DSparkProposerTestBase:
    """Shared helpers for ``AscendDSparkProposer`` tests."""

    @staticmethod
    def _make_vllm_config(hf_config: SimpleNamespace) -> SimpleNamespace:
        """Build the minimal config consumed by the DSpark initializer."""
        draft_model_config = SimpleNamespace(hf_config=hf_config, get_hidden_size=lambda: _HIDDEN_SIZE)
        return SimpleNamespace(
            speculative_config=SimpleNamespace(draft_sample_method="greedy", draft_model_config=draft_model_config)
        )

    @classmethod
    def _make_proposer(
        cls,
        *,
        max_num_tokens: int,
        num_reqs: int,
        block_size: int,
        hf_config: SimpleNamespace | None = None,
    ):
        device = torch.device("cpu")
        vllm_config = cls._make_vllm_config(hf_config or SimpleNamespace())

        def mock_parent_init(
            proposer: AscendDSparkProposer,
            vllm_config: SimpleNamespace,
            device: torch.device,
            runner: object | None = None,
        ) -> None:
            del runner
            proposer.draft_model_config = vllm_config.speculative_config.draft_model_config
            proposer.num_speculative_tokens = block_size
            proposer.max_batch_size = num_reqs
            proposer.max_num_tokens = max_num_tokens
            proposer.dtype = torch.float32
            proposer.device = device
            proposer.hidden_size = _HIDDEN_SIZE
            proposer.hidden_states = torch.empty(0)
            proposer._dflash_hidden_states = torch.empty(0)

        with patch.object(AscendDSparkProposer.__base__, "__init__", mock_parent_init):
            proposer = AscendDSparkProposer(vllm_config, device)

        num_query_total = num_reqs * proposer.num_query_per_req
        proposer.positions = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer.positions[:num_query_total] = torch.arange(num_query_total, dtype=torch.int32)
        proposer.parallel_drafting_token_id = 0
        proposer.kv_cache_gid = 0
        proposer._dflash_num_context = 0

        proposer.input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._context_positions_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._slot_mapping_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._dspark_seed_buffer = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._dflash_hidden_states = torch.zeros((max_num_tokens, 8), dtype=torch.float32, device=device)
        proposer.arange_dflash = torch.arange(max_num_tokens + 1, dtype=torch.int32, device=device)
        proposer.token_arange_np = np.arange(max_num_tokens + 1, dtype=np.int32)

        gid = 0
        proposer.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_group_id=gid,
                kv_cache_spec=SimpleNamespace(block_size=block_size),
                layer_names=["L0"],
            )
        ]
        proposer._layer_group_idx = [gid]
        block_table = torch.zeros((num_reqs, 16), dtype=torch.int32, device=device)
        proposer._per_group_block_tables = {gid: block_table}
        proposer._per_group_block_table_buffers = {gid: block_table}
        slot = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._per_group_slot_mappings = {gid: slot}
        proposer._per_group_query_slot_mapping_buffers = {gid: slot.clone()}
        proposer._per_group_context_slot_mapping_buffers = {gid: slot.clone()}
        return proposer


class TestDSparkPositionsFullUnderMultiDp(_DSparkProposerTestBase):
    """Guard: under multi-DP the dspark draft proposer must hand DSA attention a
    full-length positions buffer so ``positions[:num_input_tokens]`` never reads
    out of bounds (the slice is DP-padded and may exceed the local query size)."""

    @staticmethod
    def _call_set_inputs_first_pass(proposer, *, num_reqs, block_size):
        # query_start_loc_cpu[num_reqs] is 0 so _dflash_num_context becomes 0.
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=torch.zeros(num_reqs + 1, dtype=torch.int32),
            seq_lens=torch.full((num_reqs,), 128, dtype=torch.int32),
            max_seq_len=128,
        )
        proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            next_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            target_positions=torch.zeros(num_reqs, dtype=torch.int32),
            target_hidden_states=torch.zeros((num_reqs, 8), dtype=torch.float32),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )
        return cad

    @pytest.mark.parametrize("dp_padding", MULTI_DP_PADDING_SIZES)
    def test_positions_not_pre_sliced(self, monkeypatch, dp_padding):
        """``cad.positions`` must be the full buffer, not ``[:num_query_total]``."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)

        # DSA attention slices positions[:num_input_tokens] (DP-padded); a
        # pre-slice to num_query_total reads out of bounds under multi-DP.
        assert cad.positions.shape[0] == max_num_tokens
        assert cad.positions[:num_input_tokens].shape[0] == num_input_tokens

    @pytest.mark.parametrize("dp_padding", [8, 32])
    def test_positions_full_and_padded_for_dsa(self, monkeypatch, dp_padding):
        """After set_inputs_first_pass + _pad_draft_buffers, positions[:num_input]
        is full-length and zero-padded in the DP region."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_query_total:num_input_tokens] = -999
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)
        proposer._pad_draft_buffers(num_query_total, num_input_tokens)

        dsa_slice = cad.positions[:num_input_tokens]
        assert dsa_slice.shape[0] == num_input_tokens
        assert torch.all(dsa_slice[num_query_total:] == 0)


class TestPadDraftBuffersBeforeBuild(_DSparkProposerTestBase):
    """Guard: ``_pad_draft_buffers`` must zero the DP-padding region of positions
    and run before ``build_draft_attn_metadata``, so the attention backend reads
    valid (zero) padding instead of stale values."""

    def test_zeros_dp_padding_region(self):
        """``_pad_draft_buffers`` zeros positions / input_ids / slot_mapping in
        the DP-padding region."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_actual:num_input] = -999
        proposer.input_ids[num_actual:num_input] = -999
        proposer._slot_mapping_buffer[num_actual:num_input] = -999
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            buf[num_actual:num_input] = -999

        proposer._pad_draft_buffers(num_actual, num_input)

        assert torch.all(proposer.positions[num_actual:num_input] == 0)
        assert torch.all(proposer.input_ids[num_actual:num_input] == proposer.parallel_drafting_token_id)
        assert torch.all(proposer._slot_mapping_buffer[num_actual:num_input] == -1)
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            assert torch.all(buf[num_actual:num_input] == -1)
        assert torch.all(proposer.positions[:num_actual] != -999)

    def test_noop_without_dp_padding(self):
        """Single-DP (num_input <= num_actual) leaves buffers untouched."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        snapshot = proposer.positions.clone()
        proposer._pad_draft_buffers(num_actual, num_actual)
        assert torch.equal(proposer.positions, snapshot)

    def test_must_precede_build(self):
        """build_draft_attn_metadata reads positions but does not zero it, so
        _pad_draft_buffers must run first."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        def capture_build():
            captured = {}

            def fake_build(common_attn_metadata, num_input_tokens, num_actual_tokens):
                captured["region"] = common_attn_metadata.positions[num_actual:num_input].clone()
                return None, common_attn_metadata

            return captured, fake_build

        ok = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        ok.positions[num_actual:num_input] = -999
        cap_ok, build_ok = capture_build()
        ok.build_draft_attn_metadata = build_ok
        ok._pad_draft_buffers(num_actual, num_input)
        ok.build_draft_attn_metadata(SimpleNamespace(positions=ok.positions), num_input, num_actual)
        assert torch.all(cap_ok["region"] == 0)

        bug = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        bug.positions[num_actual:num_input] = -999
        cap_bug, build_bug = capture_build()
        bug.build_draft_attn_metadata = build_bug
        bug.build_draft_attn_metadata(SimpleNamespace(positions=bug.positions), num_input, num_actual)
        bug._pad_draft_buffers(num_actual, num_input)
        assert torch.all(cap_bug["region"] == -999)

    def test_called_before_build_in_propose(self):
        """In ``_propose`` the ``_pad_draft_buffers`` call must precede
        ``build_draft_attn_metadata``."""
        src = inspect.getsource(AscendSpecDecodeBaseProposer._propose)
        pad_idx = src.find("self._pad_draft_buffers(")
        build_idx = src.find("self.build_draft_attn_metadata(")
        # Only assert when both calls live directly in _propose; a refactor that
        # extracts them elsewhere leaves this guard inert rather than brittle.
        if pad_idx != -1 and build_idx != -1:
            assert pad_idx < build_idx, (
                "_pad_draft_buffers must be called before build_draft_attn_metadata "
                "in _propose, otherwise the attention backend reads un-zeroed "
                "positions in the DP-padding region."
            )


class TestDSparkInitialization(_DSparkProposerTestBase):
    """Tests for DSpark initialization configuration."""

    @pytest.mark.parametrize(
        ("hf_config", "expected_sample_from_anchor", "expected_num_query_per_req"),
        [
            pytest.param(SimpleNamespace(), True, _NUM_SPECULATIVE_TOKENS),
            pytest.param(SimpleNamespace(dspark_bonus_anchor=True), False, 1 + _NUM_SPECULATIVE_TOKENS),
        ],
    )
    def test_configures_anchor_sampling(
        self,
        hf_config: SimpleNamespace,
        expected_sample_from_anchor: bool,
        expected_num_query_per_req: int,
    ) -> None:
        """Verify the bonus-anchor flag selects the expected query layout."""
        proposer = self._make_proposer(
            max_num_tokens=_MAX_NUM_TOKENS,
            num_reqs=_MAX_BATCH_SIZE,
            block_size=_NUM_SPECULATIVE_TOKENS,
            hf_config=hf_config,
        )
        expected_max_query_tokens = _MAX_BATCH_SIZE * expected_num_query_per_req
        assert proposer.sample_from_anchor is expected_sample_from_anchor
        assert proposer.num_query_per_req == expected_num_query_per_req
        assert proposer.max_query_tokens == expected_max_query_tokens


class _DSparkDecodeOnlyTestBase(_DSparkProposerTestBase):
    """Shared helpers for DSpark decode-only state-machine tests."""

    @staticmethod
    def _make_exec_config(
        max_tokens: int = 1024,
        max_bytes: int = 10**9,
        lazy_chunk: int = 4096,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            decode_only=True,
            max_staged_tokens_per_request=max_tokens,
            max_staged_bytes_total=max_bytes,
            lazy_init_chunk_tokens=lazy_chunk,
            overflow_policy="fallback_prefill_tail",
        )

    @classmethod
    def _make_decode_only_proposer(
        cls,
        *,
        max_num_tokens: int = 256,
        num_reqs: int = 8,
        block_size: int = 4,
        exec_config: SimpleNamespace | None = None,
    ) -> AscendDSparkProposer:
        exec_config = exec_config or cls._make_exec_config()
        fake_ascend_config = SimpleNamespace(dspark_config=exec_config)
        with patch(
            "vllm_ascend.spec_decode.dspark_proposer.get_ascend_config",
            return_value=fake_ascend_config,
        ):
            proposer = cls._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.model = MagicMock()
        return proposer

    @staticmethod
    def _make_meta(
        req_ids: list[str],
        is_prefill: list[bool],
        finishes_prefill: list[bool] | None = None,
        prompt_lens: list[int] | None = None,
    ) -> DSparkDecodeOnlyRequestMeta:
        num_reqs = len(req_ids)
        return DSparkDecodeOnlyRequestMeta(
            request_rows=list(range(num_reqs)),
            request_ids=list(req_ids),
            is_prefill=list(is_prefill),
            finishes_prefill=finishes_prefill or [False] * num_reqs,
            prompt_lens=prompt_lens or [16] * num_reqs,
        )

    @staticmethod
    def _make_pending_context(
        proposer: AscendDSparkProposer,
        req_id: str,
        *,
        phase: DSparkRequestPhase,
        num_tokens: int = 0,
        hidden_width: int = 8,
    ) -> None:
        from vllm_ascend.spec_decode.dspark_proposer import PendingDSparkContext, StagedDSparkChunk

        chunks = []
        if num_tokens:
            hidden = torch.randn(num_tokens, hidden_width)
            positions = torch.arange(num_tokens, dtype=torch.int32)
            chunk = StagedDSparkChunk(
                raw_hidden_states=hidden,
                positions=positions,
                num_tokens=num_tokens,
                num_bytes=hidden.numel() * hidden.element_size() + positions.numel() * positions.element_size(),
                hidden_row_bytes=hidden_width * hidden.element_size(),
                position_row_bytes=positions.element_size(),
            )
            chunks.append(chunk)
        ctx = PendingDSparkContext(
            request_id=req_id,
            generation=proposer._request_generations.get(req_id, 0),
            phase=phase,
            chunks=chunks,
            num_staged_tokens=sum(c.num_tokens for c in chunks),
            num_staged_bytes=sum(c.num_bytes for c in chunks),
            retained_context_tokens=proposer._dspark_retained_context_tokens,
        )
        proposer._pending_contexts[req_id] = ctx
        proposer._total_staged_bytes += ctx.num_staged_bytes


class TestDSparkDecodeOnlyClassification(_DSparkDecodeOnlyTestBase):
    def test_classify_prefill_and_final_prefill_from_cpu_metadata(self):
        proposer = self._make_decode_only_proposer()
        req_ids = ["r0", "r1", "r2"]
        num_computed = np.array([0, 8, 20], dtype=np.int64)
        prompt_lens = np.array([10, 10, 20], dtype=np.int64)
        scheduled = {"r0": 4, "r1": 2, "r2": 1}
        meta = proposer.classify_decode_only_requests(req_ids, num_computed, prompt_lens, scheduled)
        assert meta.is_prefill == [True, True, False]
        # r0: 0 + 4 < 10 (unfinished chunk); r1: 8 + 2 >= 10 (final prefill).
        assert meta.finishes_prefill == [False, True, False]

    def test_collect_eligible_rows_mixed_batch(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.COLLECTING)
        self._make_pending_context(proposer, "r1", phase=DSparkRequestPhase.PENDING_INIT)
        self._make_pending_context(proposer, "r2", phase=DSparkRequestPhase.READY)
        self._make_pending_context(proposer, "r3", phase=DSparkRequestPhase.FALLBACK_PREFILL)
        meta = self._make_meta(
            ["r0", "r1", "r2", "r3"],
            is_prefill=[True, False, False, True],
            finishes_prefill=[False, False, False, True],
        )
        pending, ready, fallback_final = proposer._collect_eligible_rows(meta)
        assert pending == [1]
        assert ready == [2]
        assert fallback_final == [3]

    def test_collect_eligible_rejects_decode_without_prefill(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.COLLECTING)
        meta = self._make_meta(["r0"], is_prefill=[False])
        with pytest.raises(RuntimeError, match="decode step"):
            proposer._collect_eligible_rows(meta)

    def test_collect_eligible_rejects_invalid(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.INVALID)
        meta = self._make_meta(["r0"], is_prefill=[False])
        with pytest.raises(RuntimeError, match="INVALID"):
            proposer._collect_eligible_rows(meta)


class TestDSparkDecodeOnlyStaging(_DSparkDecodeOnlyTestBase):
    def test_stage_prefill_never_calls_dspark_model(self):
        proposer = self._make_decode_only_proposer()
        meta = self._make_meta(["r0"], is_prefill=[True], finishes_prefill=[True], prompt_lens=[4])
        raw = torch.randn(4, 8)
        positions = torch.arange(4, dtype=torch.int32)
        qsl = torch.tensor([0, 4], dtype=torch.int32)
        proposer.stage_prefill_context(meta, raw, positions, qsl)
        model = proposer.model
        assert not model.combine_hidden_states.called
        assert not model.precompute_and_store_context_kv.called
        assert not model.called  # draft forward
        assert not model.compute_logits.called

    def test_chunked_prefill_accumulates_per_request_id(self):
        proposer = self._make_decode_only_proposer()
        meta = self._make_meta(["r0", "r1"], is_prefill=[True, True])
        raw = torch.randn(6, 8)
        positions = torch.arange(6, dtype=torch.int32)
        qsl = torch.tensor([0, 4, 6], dtype=torch.int32)
        proposer.stage_prefill_context(meta, raw, positions, qsl)
        assert proposer._pending_contexts["r0"].num_staged_tokens == 4
        assert proposer._pending_contexts["r1"].num_staged_tokens == 2
        # Second chunk for r0 accumulates on top of the first.
        meta2 = self._make_meta(["r0"], is_prefill=[True], finishes_prefill=[True], prompt_lens=[10])
        raw2 = torch.randn(2, 8)
        positions2 = torch.arange(4, 6, dtype=torch.int32)
        qsl2 = torch.tensor([0, 2], dtype=torch.int32)
        proposer.stage_prefill_context(meta2, raw2, positions2, qsl2)
        assert proposer._pending_contexts["r0"].num_staged_tokens == 6
        assert len(proposer._pending_contexts["r0"].chunks) == 2
        # Staged tensors are detached copies with independent storage.
        staged = proposer._pending_contexts["r0"].chunks[0].raw_hidden_states
        raw.zero_()
        assert not torch.all(staged == 0)

    def test_final_prefill_transitions_to_pending_init(self):
        proposer = self._make_decode_only_proposer()
        meta = self._make_meta(["r0"], is_prefill=[True], finishes_prefill=[True], prompt_lens=[4])
        proposer.stage_prefill_context(
            meta, torch.randn(4, 8), torch.arange(4, dtype=torch.int32), torch.tensor([0, 4], dtype=torch.int32)
        )
        assert proposer._pending_contexts["r0"].phase == DSparkRequestPhase.PENDING_INIT

    def test_sliding_window_trims_to_last_n_positions(self):
        proposer = self._make_decode_only_proposer()
        proposer._dspark_retained_context_tokens = 4
        meta = self._make_meta(["r0"], is_prefill=[True], prompt_lens=[10])
        positions = torch.arange(6, dtype=torch.int32)
        proposer.stage_prefill_context(meta, torch.randn(6, 8), positions, torch.tensor([0, 6], dtype=torch.int32))
        ctx = proposer._pending_contexts["r0"]
        assert ctx.num_staged_tokens == 4
        assert ctx.chunks[0].positions.tolist() == [2, 3, 4, 5]
        assert ctx.num_staged_bytes == 4 * (8 * 4 + 4)

    def test_full_attention_keeps_entire_prompt(self):
        proposer = self._make_decode_only_proposer()
        assert proposer._dspark_retained_context_tokens is None
        meta = self._make_meta(["r0"], is_prefill=[True], prompt_lens=[10])
        proposer.stage_prefill_context(
            meta, torch.randn(64, 8), torch.arange(64, dtype=torch.int32), torch.tensor([0, 64], dtype=torch.int32)
        )
        assert proposer._pending_contexts["r0"].num_staged_tokens == 64

    def test_per_request_token_limit_falls_back_only_that_request(self):
        exec_config = self._make_exec_config(max_tokens=4)
        proposer = self._make_decode_only_proposer(exec_config=exec_config)
        proposer._project_staged_contexts = MagicMock()
        meta = self._make_meta(["r0", "r1"], is_prefill=[True, True])
        raw = torch.randn(10, 8)
        positions = torch.arange(10, dtype=torch.int32)
        qsl = torch.tensor([0, 6, 10], dtype=torch.int32)
        proposer.stage_prefill_context(meta, raw, positions, qsl)
        ctx0 = proposer._pending_contexts["r0"]
        assert ctx0.phase == DSparkRequestPhase.FALLBACK_PREFILL
        assert ctx0.num_staged_tokens == 0  # staging released after fallback
        assert ctx0.fallback_reason == "per_request_tokens"
        # r1 unaffected: still staging.
        assert proposer._pending_contexts["r1"].phase == DSparkRequestPhase.COLLECTING
        assert proposer._pending_contexts["r1"].num_staged_tokens == 4

    def test_global_byte_limit_falls_back(self):
        exec_config = self._make_exec_config(max_bytes=64)
        proposer = self._make_decode_only_proposer(exec_config=exec_config)
        proposer._project_staged_contexts = MagicMock()
        meta = self._make_meta(["r0"], is_prefill=[True], prompt_lens=[10])
        proposer.stage_prefill_context(
            meta, torch.randn(4, 8), torch.arange(4, dtype=torch.int32), torch.tensor([0, 4], dtype=torch.int32)
        )
        ctx = proposer._pending_contexts["r0"]
        assert ctx.phase == DSparkRequestPhase.FALLBACK_PREFILL
        assert ctx.fallback_reason == "total_bytes"
        assert proposer._total_staged_bytes == 0

    def test_fallback_projects_staged_context(self):
        exec_config = self._make_exec_config(max_tokens=4)
        proposer = self._make_decode_only_proposer(exec_config=exec_config)
        proposer._project_staged_contexts = MagicMock()
        meta = self._make_meta(["r0"], is_prefill=[True], prompt_lens=[10])
        proposer.stage_prefill_context(
            meta, torch.randn(6, 8), torch.arange(6, dtype=torch.int32), torch.tensor([0, 6], dtype=torch.int32)
        )
        proposer._project_staged_contexts.assert_called_once()
        (contexts, rows), _ = proposer._project_staged_contexts.call_args
        assert rows == [0]

    def test_fallback_prefill_projects_later_chunks_directly(self):
        exec_config = self._make_exec_config(max_tokens=4)
        proposer = self._make_decode_only_proposer(exec_config=exec_config)
        proposer._project_staged_contexts = MagicMock()
        meta = self._make_meta(["r0"], is_prefill=[True], prompt_lens=[10])
        proposer.stage_prefill_context(
            meta, torch.randn(6, 8), torch.arange(6, dtype=torch.int32), torch.tensor([0, 6], dtype=torch.int32)
        )
        assert proposer._pending_contexts["r0"].phase == DSparkRequestPhase.FALLBACK_PREFILL
        # Next chunk for the fallback request is projected, not staged.
        with patch.object(proposer, "_project_context_kv") as mock_project:
            meta2 = self._make_meta(["r0"], is_prefill=[True], prompt_lens=[10])
            proposer.stage_prefill_context(
                meta2, torch.randn(2, 8), torch.arange(6, 8, dtype=torch.int32), torch.tensor([0, 2], dtype=torch.int32)
            )
            mock_project.assert_called_once()
            args, _ = mock_project.call_args
            assert args[0].shape == (2, 8)
            assert args[1].tolist() == [6, 7]
        assert proposer._pending_contexts["r0"].num_staged_tokens == 0

    def test_prefill_after_init_resets_to_collecting(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.READY, num_tokens=2)
        meta = self._make_meta(["r0"], is_prefill=[True], prompt_lens=[10])
        proposer.stage_prefill_context(
            meta, torch.randn(2, 8), torch.arange(2, dtype=torch.int32), torch.tensor([0, 2], dtype=torch.int32)
        )
        ctx = proposer._pending_contexts["r0"]
        assert ctx.phase == DSparkRequestPhase.COLLECTING
        assert ctx.num_staged_tokens == 2  # fresh staging for the new chunk


class TestDSparkDecodeOnlyLifecycle(_DSparkDecodeOnlyTestBase):
    def test_release_drops_state_and_bumps_generation(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.PENDING_INIT, num_tokens=4)
        staged_bytes = proposer._total_staged_bytes
        assert staged_bytes > 0
        proposer.release_requests(["r0"])
        assert "r0" not in proposer._pending_contexts
        assert proposer._total_staged_bytes == 0
        assert proposer._request_generations["r0"] == 1
        # A resubmitted request with the same ID starts a fresh context.
        proposer._get_or_create_context("r0", prompt_len=8)
        assert proposer._pending_contexts["r0"].generation == 1
        assert proposer._pending_contexts["r0"].phase == DSparkRequestPhase.COLLECTING

    def test_invalidate_drops_state_and_counts_reason(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.READY, num_tokens=4)
        proposer.invalidate_requests(["r0"], reason="resumed")
        assert "r0" not in proposer._pending_contexts
        assert proposer._dspark_stats["invalidated_resumed"] == 1
        assert proposer._total_staged_bytes == 0

    def test_mark_rows_proposed_releases_staging_and_sets_ready(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.PENDING_INIT, num_tokens=4)
        self._make_pending_context(proposer, "r1", phase=DSparkRequestPhase.FALLBACK_PREFILL)
        meta = self._make_meta(["r0", "r1"], is_prefill=[False, True])
        proposer._mark_rows_proposed(meta, pending_rows=[0], fallback_final_rows=[1])
        assert proposer._pending_contexts["r0"].phase == DSparkRequestPhase.READY
        assert proposer._pending_contexts["r0"].num_staged_tokens == 0
        assert proposer._pending_contexts["r1"].phase == DSparkRequestPhase.READY
        assert proposer._total_staged_bytes == 0

    def test_lazy_init_failure_invalidates_and_raises(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.PENDING_INIT, num_tokens=4)
        with (
            patch.object(proposer, "_project_staged_contexts", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="lazy init failed"),
        ):
            proposer.initialize_pending_contexts(["r0"], [0])
        ctx = proposer._pending_contexts["r0"]
        assert ctx.phase == DSparkRequestPhase.INVALID
        assert ctx.num_staged_tokens == 0
        assert proposer._total_staged_bytes == 0

    def test_lazy_init_success_marks_done(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.PENDING_INIT, num_tokens=4)
        with patch.object(proposer, "_project_staged_contexts") as mock_project:
            proposer.initialize_pending_contexts(["r0"], [3])
        mock_project.assert_called_once()
        assert proposer._pending_contexts["r0"].lazy_init_done


class TestDSparkDecodeOnlySlotRebuild(_DSparkDecodeOnlyTestBase):
    def test_project_context_kv_requires_current_block_table(self):
        proposer = self._make_decode_only_proposer()
        proposer.model.combine_hidden_states = lambda h: h
        proposer.model.model = SimpleNamespace(_num_attn_layers=4)
        proposer._lazy_init_slot_buffers = {0: torch.zeros(16, dtype=torch.int32)}
        proposer._per_group_block_tables = {}  # runner metadata missing
        with pytest.raises(RuntimeError, match="block table"):
            proposer._project_context_kv(
                torch.randn(4, 8),
                torch.arange(4, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 4], dtype=torch.int32),
            )

    def test_ensure_draft_fused_buffers_rebuilds_invalid_sentinel(self):
        proposer = self._make_decode_only_proposer()
        draft_inner = MagicMock()
        draft_inner._num_attn_layers = -1  # invalid sentinel left by a partial build
        proposer.model.model = draft_inner

        proposer._ensure_draft_fused_buffers()

        draft_inner._build_fused_kv_buffers.assert_called_once()

    def test_ensure_draft_fused_buffers_skips_when_valid(self):
        proposer = self._make_decode_only_proposer()
        draft_inner = MagicMock()
        draft_inner._num_attn_layers = 4
        proposer.model.model = draft_inner

        proposer._ensure_draft_fused_buffers()

        draft_inner._build_fused_kv_buffers.assert_not_called()

    def test_project_context_kv_rejects_invalid_num_attn_layers(self):
        proposer = self._make_decode_only_proposer()
        proposer.model.combine_hidden_states = lambda h: h
        draft_inner = SimpleNamespace(_num_attn_layers=-1)
        draft_inner._build_fused_kv_buffers = MagicMock()
        proposer.model.model = draft_inner
        proposer._per_group_block_tables = {}  # unreachable: assert fires first

        with pytest.raises(RuntimeError, match="_num_attn_layers"):
            proposer._project_context_kv(
                torch.randn(2, 8),
                torch.arange(2, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 2], dtype=torch.int32),
            )

    def test_project_context_kv_rejects_2d_positions(self):
        proposer = self._make_decode_only_proposer()
        proposer.model.combine_hidden_states = lambda h: h
        proposer.model.model = SimpleNamespace(_num_attn_layers=4)
        with pytest.raises(RuntimeError, match="1-D token positions"):
            proposer._project_context_kv(
                torch.randn(4, 8),
                torch.zeros(3, 4, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 4], dtype=torch.int32),
            )

    def test_project_context_kv_rejects_shape_mismatch(self):
        proposer = self._make_decode_only_proposer()
        proposer.model.combine_hidden_states = lambda h: h
        proposer.model.model = SimpleNamespace(_num_attn_layers=4)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            proposer._project_context_kv(
                torch.randn(4, 8),
                torch.arange(3, dtype=torch.int32),  # 3 positions vs 4 hidden rows
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 4], dtype=torch.int32),
            )

    def test_build_dspark_context_slots_zero_reqs_is_noop(self):
        from vllm_ascend.ops.triton.spec_decode.utils import build_dspark_context_slots

        # Empty subbatch must not launch any kernel.
        build_dspark_context_slots(
            positions=torch.zeros(0, dtype=torch.int32),
            req_row_map=torch.zeros(0, dtype=torch.int32),
            req_start_loc=torch.tensor([0], dtype=torch.int32),
            block_table=torch.zeros(2, 8, dtype=torch.int32),
            block_size=16,
            out_slots=torch.zeros(0, dtype=torch.int32),
        )

    def test_flatten_target_positions(self):
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer as P

        flat = torch.arange(7, dtype=torch.int32)
        assert torch.equal(P._flatten_target_positions(flat), flat)
        mrope = torch.stack([torch.arange(7, dtype=torch.int32), torch.full((7,), 5), torch.zeros(7)])
        assert torch.equal(P._flatten_target_positions(mrope), mrope[0])
        from vllm_ascend.ops.triton.spec_decode.utils import build_dspark_context_slots

        # Empty subbatch must not launch any kernel.
        build_dspark_context_slots(
            positions=torch.zeros(0, dtype=torch.int32),
            req_row_map=torch.zeros(0, dtype=torch.int32),
            req_start_loc=torch.tensor([0], dtype=torch.int32),
            block_table=torch.zeros(2, 8, dtype=torch.int32),
            block_size=16,
            out_slots=torch.zeros(0, dtype=torch.int32),
        )


class TestDSparkDecodeOnlyProposal(_DSparkDecodeOnlyTestBase):
    def _make_cad(self, num_reqs: int, qsl: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.tensor(qsl, dtype=torch.int32),
            query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
            seq_lens=torch.full((num_reqs,), 32, dtype=torch.int32),
            block_table_tensor=torch.arange(num_reqs * 4, dtype=torch.int32).view(num_reqs, 4),
            num_computed_tokens_cpu=torch.arange(num_reqs, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int32),
            max_seq_len=32,
        )

    def test_propose_compact_keeps_mrope_positions_layout(self):
        """mrope/xdrope positions [3, N] must be compacted along the token
        dim (last), not the rope dim, so set_inputs_first_pass can still
        flatten them inside _propose."""
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r1", phase=DSparkRequestPhase.READY)
        # r0: prefill rows 0..1; r1: decode rows 2..4.
        num_computed = np.array([4, 20], dtype=np.int64)
        prompt_lens = np.array([12, 12], dtype=np.int64)
        scheduled = {"r0": 2, "r1": 3}
        cad = self._make_cad(2, [0, 2, 5])
        rope = torch.arange(3).unsqueeze(1) * 100 + torch.arange(5)  # [3, 5]
        k = proposer.num_speculative_tokens
        captured = {}

        def fake_propose(**kwargs):
            captured.update(kwargs)
            # Only r1 (one decode row) is eligible here.
            return torch.ones(1, k, dtype=torch.int64)

        with (
            patch.object(proposer, "_propose", side_effect=fake_propose),
            patch.object(proposer, "_project_staged_contexts"),
        ):
            proposer.propose_decode_only(
                scheduler_output=SimpleNamespace(num_scheduled_tokens=scheduled),
                common_attn_metadata=cad,
                token_indices=None,
                token_indices_to_sample=None,
                num_rejected_tokens_gpu=None,
                next_token_ids=torch.arange(2, dtype=torch.int64),
                target_token_ids=torch.arange(5, dtype=torch.int64),
                target_positions=rope.clone(),
                raw_target_hidden_states=torch.randn(5, 8),
                num_scheduled_tokens=scheduled,
                req_ids=["r0", "r1"],
                num_computed_tokens_cpu=num_computed,
                num_prompt_tokens=prompt_lens,
                target_model_batch_desc=SimpleNamespace(uniform=False),
                sampling_metadata=None,
            )

        sub_positions = captured["target_positions"]
        assert sub_positions.shape == (3, 3)
        # Token dim compacted: columns 2..4 of the original rope tensor.
        assert torch.equal(sub_positions, rope[:, 2:5])
        # Staged positions are flat 1-D from the first rope dim.
        ctx0 = proposer._pending_contexts["r0"]
        assert ctx0.chunks[0].positions.dim() == 1

    def test_prefill_only_step_publishes_empty_mask_without_proposal(self):
        proposer = self._make_decode_only_proposer()
        num_computed = np.array([0, 8], dtype=np.int64)
        prompt_lens = np.array([12, 12], dtype=np.int64)
        scheduled = {"r0": 4, "r1": 4}
        cad = self._make_cad(2, [0, 4, 8])
        with patch.object(proposer, "_propose") as mock_propose:
            drafts = proposer.propose_decode_only(
                scheduler_output=SimpleNamespace(num_scheduled_tokens=scheduled),
                common_attn_metadata=cad,
                token_indices=None,
                token_indices_to_sample=None,
                num_rejected_tokens_gpu=None,
                next_token_ids=torch.zeros(2, dtype=torch.int64),
                target_token_ids=torch.zeros(8, dtype=torch.int64),
                target_positions=torch.arange(8, dtype=torch.int32),
                raw_target_hidden_states=torch.randn(8, 8),
                num_scheduled_tokens=scheduled,
                req_ids=["r0", "r1"],
                num_computed_tokens_cpu=num_computed,
                num_prompt_tokens=prompt_lens,
                target_model_batch_desc=SimpleNamespace(uniform=False),
                sampling_metadata=None,
            )
        mock_propose.assert_not_called()
        assert drafts.shape == (2, proposer.num_speculative_tokens)
        assert proposer.take_last_draft_valid_mask() == [False, False]
        assert proposer._pending_contexts["r1"].phase == DSparkRequestPhase.PENDING_INIT

    def test_mixed_batch_compacts_eligible_and_scatters(self):
        proposer = self._make_decode_only_proposer()
        # r0: collecting prefill (2 tokens); r1: first decode (3 tokens);
        # r2: ready decode (2 tokens).
        self._make_pending_context(proposer, "r1", phase=DSparkRequestPhase.PENDING_INIT, num_tokens=12)
        self._make_pending_context(proposer, "r2", phase=DSparkRequestPhase.READY)
        num_computed = np.array([4, 20, 30], dtype=np.int64)
        prompt_lens = np.array([12, 12, 12], dtype=np.int64)
        scheduled = {"r0": 2, "r1": 3, "r2": 2}
        qsl = [0, 2, 5, 7]
        cad = self._make_cad(3, qsl)
        k = proposer.num_speculative_tokens
        sub_drafts = torch.arange(2 * k, dtype=torch.int64).view(2, k) + 1

        captured = {}

        def fake_propose(**kwargs):
            captured.update(kwargs)
            return sub_drafts

        with (
            patch.object(proposer, "_propose", side_effect=fake_propose),
            patch.object(proposer, "_project_staged_contexts") as mock_project,
        ):
            drafts = proposer.propose_decode_only(
                scheduler_output=SimpleNamespace(num_scheduled_tokens=scheduled),
                common_attn_metadata=cad,
                token_indices=None,
                token_indices_to_sample=None,
                num_rejected_tokens_gpu=None,
                next_token_ids=torch.arange(3, dtype=torch.int64),
                target_token_ids=torch.arange(7, dtype=torch.int64),
                target_positions=torch.arange(7, dtype=torch.int32),
                raw_target_hidden_states=torch.randn(7, 8),
                num_scheduled_tokens=scheduled,
                req_ids=["r0", "r1", "r2"],
                num_computed_tokens_cpu=num_computed,
                num_prompt_tokens=prompt_lens,
                target_model_batch_desc=SimpleNamespace(uniform=False),
                sampling_metadata=None,
            )

        # Lazy init ran for r1's staged prompt context only.
        mock_project.assert_called_once()
        # Proposal received the compact subbatch: rows 1..2, tokens 2..7.
        assert captured["target_token_ids"].tolist() == [2, 3, 4, 5, 6]
        assert captured["next_token_ids"].tolist() == [1, 2]
        assert captured["common_attn_metadata"].num_reqs == 2
        assert captured["common_attn_metadata"].query_start_loc_cpu.tolist() == [0, 3, 5]
        # Scatter: row 0 stays zero (invalid), rows 1-2 carry the sub drafts.
        assert torch.equal(drafts[0], torch.zeros(k, dtype=torch.int64))
        assert torch.equal(drafts[1], sub_drafts[0])
        assert torch.equal(drafts[2], sub_drafts[1])
        assert proposer.take_last_draft_valid_mask() == [False, True, True]
        # r1 transitioned to READY after its first proposal.
        assert proposer._pending_contexts["r1"].phase == DSparkRequestPhase.READY
        assert proposer._pending_contexts["r1"].num_staged_tokens == 0

    def test_all_eligible_uses_identity_path(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.READY)
        self._make_pending_context(proposer, "r1", phase=DSparkRequestPhase.READY)
        num_computed = np.array([20, 30], dtype=np.int64)
        prompt_lens = np.array([12, 12], dtype=np.int64)
        scheduled = {"r0": 2, "r1": 2}
        cad = self._make_cad(2, [0, 2, 4])
        k = proposer.num_speculative_tokens
        with patch.object(
            proposer,
            "_propose",
            return_value=torch.ones(2, k, dtype=torch.int64),
        ) as mock_propose:
            drafts = proposer.propose_decode_only(
                scheduler_output=SimpleNamespace(num_scheduled_tokens=scheduled),
                common_attn_metadata=cad,
                token_indices=None,
                token_indices_to_sample=None,
                num_rejected_tokens_gpu=None,
                next_token_ids=torch.arange(2, dtype=torch.int64),
                target_token_ids=torch.arange(4, dtype=torch.int64),
                target_positions=torch.arange(4, dtype=torch.int32),
                raw_target_hidden_states=torch.randn(4, 8),
                num_scheduled_tokens=scheduled,
                req_ids=["r0", "r1"],
                num_computed_tokens_cpu=num_computed,
                num_prompt_tokens=prompt_lens,
                target_model_batch_desc=SimpleNamespace(uniform=False),
                sampling_metadata=None,
            )
        _, kwargs = mock_propose.call_args
        assert kwargs["target_token_ids"].tolist() == [0, 1, 2, 3]
        assert torch.equal(drafts, torch.ones(2, k, dtype=torch.int64))
        assert proposer.take_last_draft_valid_mask() == [True, True]


class TestDSparkDecodeOnlyStagingBytes(_DSparkDecodeOnlyTestBase):
    def test_byte_accounting_has_no_leak_across_lifecycle(self):
        proposer = self._make_decode_only_proposer()
        self._make_pending_context(proposer, "r0", phase=DSparkRequestPhase.PENDING_INIT, num_tokens=8)
        first = proposer._total_staged_bytes
        assert first == proposer._pending_contexts["r0"].num_staged_bytes
        proposer._trim_staged_prefix(proposer._pending_contexts["r0"])
        proposer.release_requests(["r0"])
        assert proposer._total_staged_bytes == 0
