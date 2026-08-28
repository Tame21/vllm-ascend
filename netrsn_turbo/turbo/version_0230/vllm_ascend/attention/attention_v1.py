# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime equivalent of the non-test change in lora_1.patch."""

from copy import copy
from functools import wraps

from vllm_ascend.attention import attention_v1


def _wrap_update_graph_params(original):
    @wraps(original)
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        # Match only the non-paged, non-sinks, target-model FIA branch.
        # Draft and sinks metadata follow their existing upstream paths.
        if (
            not attention_v1.using_paged_attention(num_tokens, vllm_config)
            and not attention_v1._EXTRA_CTX.sinks
            and not attention_v1._EXTRA_CTX.is_draft_model
        ):
            metadata = forward_context.attn_metadata
            if isinstance(metadata, dict):
                fia_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if hasattr(value, "seq_lens_list")
                }
                if len(fia_metadata) != len(metadata):
                    # The conv1d path still needs the original GDN metadata.
                    # Never mutate the shared ForwardContext, even temporarily.
                    forward_context = copy(forward_context)
                    forward_context.attn_metadata = fia_metadata

        return original(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            num_dcp_pcp_tokens,
            draft_attn_metadatas,
        )

    return update_graph_params
