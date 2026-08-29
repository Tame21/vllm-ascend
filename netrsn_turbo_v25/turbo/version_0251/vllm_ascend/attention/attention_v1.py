# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 LoRA graph-parameter filtering for Ascend attention."""

from copy import copy
from functools import wraps

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.utils import using_paged_attention

from netrsn_turbo.turbo.version_0251.vllm_ascend.lora.punica_npu import (
    patch_applies,
)


def wrap_update_graph_params(original):
    @wraps(original)
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        if (
            patch_applies(vllm_config)
            and not _EXTRA_CTX.is_draft_model
            and not using_paged_attention(num_tokens, vllm_config)
            and isinstance(forward_context.attn_metadata, dict)
        ):
            filtered_context = copy(forward_context)
            filtered_context.attn_metadata = {
                key: metadata
                for key, metadata in forward_context.attn_metadata.items()
                if hasattr(metadata, "seq_lens_list")
                and hasattr(metadata, "actual_seq_lengths_q")
            }
            forward_context = filtered_context
        return original(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            draft_attn_metadatas=draft_attn_metadatas,
        )

    return update_graph_params
