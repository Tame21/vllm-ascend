"""Regression tests: the v2 Ascend speculator must skip GDN (SSM) metadata.

Hybrid GDN models run linear-attention layers alongside full-attention layers,
so the metadata dict flowing through ``AscendAutoRegressiveSpeculator`` can
contain ``GDNAttentionMetadata`` entries (e.g. draft prefill reuses the target
model's metadata). GDN metadata carries no ``seq_lens``/``seq_lens_cpu``/
``attn_state``; its recurrent state advances inside the GDN kernels. The
speculator's per-step seq-len bookkeeping must only touch full-attention
metadata (mirroring v1, where the drafter never mutates GDN metadata).
"""

from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def _make_speculator(**attrs) -> AscendAutoRegressiveSpeculator:
    speculator = object.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "GQA"
    for name, value in attrs.items():
        setattr(speculator, name, value)
    return speculator


def _make_gdn_metadata() -> GDNAttentionMetadata:
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
    )


def _make_full_attn_metadata(num_reqs: int = 4) -> AscendMetadata:
    return AscendMetadata(
        seq_lens=torch.ones(num_reqs, dtype=torch.int32),
        seq_lens_cpu=torch.ones(num_reqs, dtype=torch.int32),
    )


def _make_mixed_attn_metadata() -> dict:
    # GDN entry first so a `next(iter(...))`-style access also trips it.
    return {"linear_attn": _make_gdn_metadata(), "full_attn": _make_full_attn_metadata()}


def test_ascend_update_seq_lens_skips_gdn_metadata():
    speculator = _make_speculator()
    attn_metadata = _make_mixed_attn_metadata()

    speculator._ascend_update_seq_lens(attn_metadata)

    full_attn = attn_metadata["full_attn"]
    assert full_attn.seq_lens.tolist() == [2, 2, 2, 2]
    assert full_attn.seq_len_list == [2, 2, 2, 2]
    gdn = attn_metadata["linear_attn"]
    assert not hasattr(gdn, "seq_lens")
    assert not hasattr(gdn, "seq_len_list")


def test_update_decode_attn_metadata_skips_gdn_metadata():
    speculator = _make_speculator(
        input_batch=SimpleNamespace(seq_lens_np=np.array([5, 7, 0, 0], dtype=np.int32)),
        max_model_len=100,
    )
    attn_metadata = _make_mixed_attn_metadata()

    speculator._update_decode_attn_metadata(attn_metadata, step=1, num_reqs=2)

    full_attn = attn_metadata["full_attn"]
    assert full_attn.seq_lens_cpu.tolist() == [6, 8, 0, 0]
    assert full_attn.seq_lens_list == [6, 8, 0, 0]
    assert full_attn.actual_seq_lengths_q == [1, 2, 3, 4]
    gdn = attn_metadata["linear_attn"]
    assert not hasattr(gdn, "seq_lens_cpu")
    assert not hasattr(gdn, "seq_lens_list")
    assert not hasattr(gdn, "actual_seq_lengths_q")


def test_update_decode_attn_metadata_returns_without_full_attn_layers():
    speculator = _make_speculator(
        input_batch=SimpleNamespace(seq_lens_np=np.array([5], dtype=np.int32)),
        max_model_len=100,
    )

    speculator._update_decode_attn_metadata({"linear_attn": _make_gdn_metadata()}, step=1)


def test_init_decode_draft_attn_metadatas_keeps_gdn_entries_untouched():
    speculator = _make_speculator(
        input_buffers=SimpleNamespace(
            draft_seq_lens_cpus=[
                torch.zeros(8, dtype=torch.int32),
                torch.zeros(8, dtype=torch.int32),
            ]
        )
    )
    attn_metadata = _make_mixed_attn_metadata()

    draft_attn_metadatas = speculator._init_decode_draft_attn_metadatas(attn_metadata, num_reqs_padded=4)

    assert len(draft_attn_metadatas) == 2
    for per_step_attn_metadata in draft_attn_metadatas:
        # GDN entries are still copied per step: the draft forward looks its
        # layers up by name, but the speculator must not decorate them with
        # full-attention fields.
        assert set(per_step_attn_metadata) == {"linear_attn", "full_attn"}
        gdn = per_step_attn_metadata["linear_attn"]
        assert isinstance(gdn, GDNAttentionMetadata)
        assert not hasattr(gdn, "seq_lens_cpu")
        assert not hasattr(gdn, "attn_state")
        full_attn = per_step_attn_metadata["full_attn"]
        assert full_attn.attn_state is AscendAttentionState.DecodeOnly
        assert full_attn.seq_lens_cpu.shape[0] == 4
