"""Attention graph-update replacement for hybrid dense models."""

from typing import Any

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.attention.utils import using_paged_attention


ORIGINAL_UPDATE_GRAPH_PARAMS = AscendAttentionBackendImpl.update_graph_params


def filter_fia_metadata(attn_metadata: Any) -> Any:
    if not isinstance(attn_metadata, dict):
        return attn_metadata
    return {
        key: metadata
        for key, metadata in attn_metadata.items()
        if hasattr(metadata, "seq_lens_list")
        and hasattr(metadata, "actual_seq_lengths_q")
    }


def update_graph_params(
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    num_dcp_pcp_tokens=None,
    draft_attn_metadatas=None,
):
    should_filter = (
        not _EXTRA_CTX.is_draft_model
        and not using_paged_attention(num_tokens, vllm_config)
    )
    original_metadata = forward_context.attn_metadata
    if should_filter:
        forward_context.attn_metadata = filter_fia_metadata(original_metadata)
    try:
        return ORIGINAL_UPDATE_GRAPH_PARAMS(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            num_dcp_pcp_tokens,
            draft_attn_metadatas,
        )
    finally:
        forward_context.attn_metadata = original_metadata

