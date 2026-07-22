#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""Exclude GDN/linear-attention metadata from FIA graph updates.

Hybrid models such as Qwen3.5 put both regular attention metadata and
``GDNAttentionMetadata`` in ``ForwardContext.attn_metadata``.  Ascend's FIA
graph-update path consumes only regular attention metadata and accesses fields
such as ``seq_lens_list``.  Passing GDN metadata into that path causes:

    AttributeError: 'GDNAttentionMetadata' object has no attribute
    'seq_lens_list'

This restores the capability-based filtering used by the v18 patch without
replacing v23's newer FIA/sinks/C8/layer-aware replay implementation.
"""

from functools import wraps
from typing import Any

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


def _is_fia_metadata(metadata: object) -> bool:
    """Return whether metadata provides the fields required by FIA replay."""
    return hasattr(metadata, "seq_lens_list")


def _filter_fia_metadata(metadata: Any) -> Any:
    """Filter one per-layer metadata mapping while preserving key order."""
    if not isinstance(metadata, dict):
        return metadata
    return {key: value for key, value in metadata.items() if _is_fia_metadata(value)}


def _filter_draft_fia_metadata(metadata: Any) -> Any:
    """Filter the list of per-step metadata mappings used by draft models."""
    if not isinstance(metadata, list):
        return metadata
    return [_filter_fia_metadata(per_step_metadata) for per_step_metadata in metadata]


if not hasattr(AscendAttentionBackendImpl, "_ascend_original_update_graph_params_with_gdn"):
    _original_update_graph_params = AscendAttentionBackendImpl.update_graph_params
    AscendAttentionBackendImpl._ascend_original_update_graph_params_with_gdn = (  # type: ignore[attr-defined]
        _original_update_graph_params
    )

    @wraps(_original_update_graph_params)
    def _patched_update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        # update_graph_params is synchronous. Temporarily expose only metadata
        # belonging to FIA, then restore the complete hybrid metadata mapping so
        # GDN layers continue to receive their own metadata during model forward.
        original_attn_metadata = forward_context.attn_metadata
        forward_context.attn_metadata = _filter_fia_metadata(original_attn_metadata)
        filtered_draft_metadata = _filter_draft_fia_metadata(draft_attn_metadatas)

        try:
            return _original_update_graph_params(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                filtered_draft_metadata,
            )
        finally:
            forward_context.attn_metadata = original_attn_metadata

    AscendAttentionBackendImpl.update_graph_params = staticmethod(  # type: ignore[method-assign]
        _patched_update_graph_params
    )
