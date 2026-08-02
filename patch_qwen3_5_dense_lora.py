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

"""Compatibility patches for Qwen3.5 dense LoRA on vLLM v0.23.

Qwen3.5 registers its language tower below ``language_model`` while common
text adapters use bare ``model.layers.*`` names.  The model also contains GDN
linear-attention layers and narrow packed projections.  This patch aligns the
adapter names, supplies a graph-safe expand-slice fallback for those packed
projections, and keeps GDN metadata out of FIA graph-task replay.
"""

from collections.abc import Iterable
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.lora.model_manager import LoRAModelManager
from vllm.lora.worker_manager import WorkerLoRAManager

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

logger = init_logger(__name__)

_PATCH_MARKER = "_vllm_ascend_qwen3_5_dense_lora_patch"


@torch.library.custom_op(
    "vllm_ascend::lora_bmm_expand_slice",
    mutates_args={"y"},
)
def _lora_bmm_expand_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    indices: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    """Apply a packed LoRA-B slice without the fused rank/width constraint."""
    rows = x.shape[0]
    if indices.shape[0] != rows:
        indices = indices[:rows]

    safe_indices = indices.clamp(min=0).to(torch.long)
    gathered = lora_b_stacked[safe_indices, 0].to(y.dtype)
    active_rank = gathered.shape[-1]
    x_active = x[..., :active_rank].to(y.dtype).contiguous()

    if x_active.shape[0] == 0 or gathered.shape[1] == 0:
        return

    delta = (x_active.unsqueeze(1) * gathered).sum(dim=-1)
    delta = torch.where(
        (indices >= 0).unsqueeze(-1),
        delta,
        torch.zeros_like(delta),
    )
    y_slice = y.narrow(1, y_offset, y_slice_size)
    if y_slice.shape[0] != delta.shape[0]:
        y_slice = y_slice[: delta.shape[0]]
    if add_inputs:
        y_slice.add_(delta)
    else:
        y_slice.copy_(delta)


@_lora_bmm_expand_slice.register_fake
def _lora_bmm_expand_slice_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    indices: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    return None


def _detect_language_model_prefix(
    lora_keys: Iterable[str],
    model_module_names: Iterable[str],
    packed_modules_mapping: dict[str, list[str]] | None = None,
) -> str | None:
    """Return the wrapper prefix that aligns adapter and model module names."""
    lora_keys = tuple(lora_keys)
    model_module_names = tuple(model_module_names)
    model_name_set = set(model_module_names)
    for lora_key in lora_keys:
        if lora_key in model_name_set:
            return ""

        candidates = [lora_key]
        module_prefix, separator, module_suffix = lora_key.rpartition(".")
        for packed_name, child_names in (packed_modules_mapping or {}).items():
            if module_suffix not in child_names:
                continue
            packed_key = packed_name
            if separator:
                packed_key = f"{module_prefix}.{packed_name}"
            candidates.append(packed_key)

        for candidate in candidates:
            if candidate in model_name_set:
                return ""
            suffix = "." + candidate
            for model_name in model_module_names:
                if model_name.endswith(suffix):
                    return model_name[: -len(candidate)]
    return None


def _enable_language_model_expand_slice(manager: Any) -> None:
    """Enable the compatible packed-LoRA path for a wrapped language tower."""
    if not getattr(manager, "supports_mm", False):
        return

    mm_mapping = getattr(manager, "mm_mapping", None)
    language_prefixes = getattr(mm_mapping, "language_model", ())
    wrapper_mapping = getattr(manager, "punica_wrapper_mapping", {})
    seen_wrappers: set[int] = set()
    for prefix in language_prefixes:
        if not prefix:
            continue
        wrapper = wrapper_mapping.get(prefix)
        if wrapper is None or id(wrapper) in seen_wrappers:
            continue
        enable = getattr(
            wrapper,
            "enable_compatible_lora_bmm_expand_slice",
            None,
        )
        if enable is None:
            continue
        enable()
        seen_wrappers.add(id(wrapper))


def _filter_fia_metadata(attn_metadata: Any) -> Any:
    """Remove GDN entries from target-model FIA graph-task replay."""
    if not isinstance(attn_metadata, dict):
        return attn_metadata
    return {
        key: metadata
        for key, metadata in attn_metadata.items()
        if hasattr(metadata, "seq_lens_list")
        and hasattr(metadata, "actual_seq_lengths_q")
    }


def apply_patch() -> None:
    if getattr(PunicaWrapperNPU, _PATCH_MARKER, False):
        return

    original_punica_init = PunicaWrapperNPU.__init__
    original_expand_slice_prefill = PunicaWrapperNPU._expand_slice_prefill
    original_expand_slice_decode = PunicaWrapperNPU._expand_slice_decode
    guarded_dense_methods = {
        method_name: getattr(PunicaWrapperNPU, method_name)
        for method_name in (
            "add_shrink",
            "add_expand",
            "add_lora_embedding",
            "add_lora_linear",
            "add_lora_logits",
        )
    }
    original_model_manager_init = LoRAModelManager.__init__
    original_load_adapter = WorkerLoRAManager._load_adapter
    original_update_graph_params = AscendAttentionBackendImpl.update_graph_params

    def patched_punica_init(self, *args, **kwargs) -> None:
        original_punica_init(self, *args, **kwargs)
        self._force_compatible_lora_expand_slice = False

    def enable_compatible_lora_bmm_expand_slice(self) -> None:
        self._force_compatible_lora_expand_slice = True

    def requires_compatible_lora_expand_slice(
        self,
        x: torch.Tensor,
        y_slice_size: int,
    ) -> bool:
        return (
            self._force_compatible_lora_expand_slice
            or x.shape[-1] > y_slice_size
        )

    def bmm_expand_slice(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ) -> None:
        _lora_bmm_expand_slice(
            y,
            x,
            lora_b_stacked,
            self._get_token_lora_indices(x),
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def patched_expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ) -> None:
        if self.no_lora:
            return
        if self._requires_compatible_lora_expand_slice(x, y_slice_size):
            self._compatible_lora_bmm_expand_slice(
                y,
                x,
                lora_b_stacked,
                y_offset,
                y_slice_size,
                add_inputs,
            )
            return
        original_expand_slice_prefill(
            self,
            y,
            x,
            lora_b_stacked,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def patched_expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ) -> None:
        if self.no_lora:
            return
        if self._requires_compatible_lora_expand_slice(x, y_slice_size):
            self._compatible_lora_bmm_expand_slice(
                y,
                x,
                lora_b_stacked,
                y_offset,
                y_slice_size,
                add_inputs,
            )
            return
        original_expand_slice_decode(
            self,
            y,
            x,
            lora_b_stacked,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def guard_dense_lora_method(original_method):
        def guarded(self, *args, **kwargs):
            if self.no_lora:
                return None
            return original_method(self, *args, **kwargs)

        return guarded

    def patched_model_manager_init(self, *args, **kwargs) -> None:
        original_model_manager_init(self, *args, **kwargs)
        _enable_language_model_expand_slice(self)

    def patched_load_adapter(self, lora_request):
        lora = original_load_adapter(self, lora_request)
        try:
            manager = self._adapter_manager
            model_module_names = tuple(getattr(manager, "modules", {}).keys())
            lora_keys = tuple(lora.loras.keys())
            prefix = _detect_language_model_prefix(
                lora_keys,
                model_module_names,
                getattr(manager, "packed_modules_mapping", None),
            )
            if not prefix:
                return lora

            lora.loras = {
                key if key.startswith(prefix) else prefix + key: weights
                for key, weights in lora.loras.items()
            }
            logger.debug(
                "Remapped %d LoRA module names with wrapper prefix %r",
                len(lora_keys),
                prefix,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Skipping Qwen3.5 LoRA module-name remap after %s: %s",
                type(error).__name__,
                error,
            )
        return lora

    def patched_update_graph_params(
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
            forward_context.attn_metadata = _filter_fia_metadata(original_metadata)
        try:
            return original_update_graph_params(
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

    PunicaWrapperNPU.__init__ = patched_punica_init
    PunicaWrapperNPU.enable_compatible_lora_bmm_expand_slice = (
        enable_compatible_lora_bmm_expand_slice
    )
    PunicaWrapperNPU._requires_compatible_lora_expand_slice = (
        requires_compatible_lora_expand_slice
    )
    PunicaWrapperNPU._compatible_lora_bmm_expand_slice = bmm_expand_slice
    PunicaWrapperNPU._expand_slice_prefill = patched_expand_slice_prefill
    PunicaWrapperNPU._expand_slice_decode = patched_expand_slice_decode
    for method_name, original_method in guarded_dense_methods.items():
        setattr(
            PunicaWrapperNPU,
            method_name,
            guard_dense_lora_method(original_method),
        )
    LoRAModelManager.__init__ = patched_model_manager_init
    WorkerLoRAManager._load_adapter = patched_load_adapter
    AscendAttentionBackendImpl.update_graph_params = staticmethod(
        patched_update_graph_params
    )
    setattr(PunicaWrapperNPU, _PATCH_MARKER, True)


apply_patch()
