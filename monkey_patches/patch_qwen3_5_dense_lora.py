"""External monkey patches for Qwen3.5 dense LoRA on vLLM-Ascend v0.23.

This module does not modify vLLM or vLLM-Ascend source files. Call
``install(register_func)`` before constructing the model. Existing functions
are installed through the project-provided ``register_func(function)`` API;
new helper methods are copied here and attached directly to their target class.
"""

from collections.abc import Callable, Iterable
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

RegisterFunc = Callable[[Callable[..., Any]], Any]
_PATCH_MARKER = "_external_qwen3_5_dense_lora_patch"


@torch.library.custom_op(
    "external_patch::lora_bmm_expand_slice",
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


def _register_replacement(
    register_func: RegisterFunc,
    replacement: Callable[..., Any],
    original: Callable[..., Any],
) -> None:
    """Give a replacement the target identity, then register it."""
    replacement.__name__ = original.__name__
    replacement.__qualname__ = original.__qualname__
    replacement.__module__ = original.__module__
    register_func(replacement)


def _detect_language_model_prefix(
    lora_keys: Iterable[str],
    model_module_names: Iterable[str],
    packed_modules_mapping: dict[str, list[str]] | None = None,
) -> str | None:
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
        if enable is not None:
            enable()
            seen_wrappers.add(id(wrapper))


def _filter_fia_metadata(attn_metadata: Any) -> Any:
    if not isinstance(attn_metadata, dict):
        return attn_metadata
    return {
        key: metadata
        for key, metadata in attn_metadata.items()
        if hasattr(metadata, "seq_lens_list")
        and hasattr(metadata, "actual_seq_lengths_q")
    }


# New Punica methods: copied into this repository and attached directly.
def enable_compatible_lora_bmm_expand_slice(self) -> None:
    self._force_compatible_lora_expand_slice = True


def _requires_compatible_lora_expand_slice(
    self,
    x: torch.Tensor,
    y_slice_size: int,
) -> bool:
    return (
        getattr(self, "_force_compatible_lora_expand_slice", False)
        or x.shape[-1] > y_slice_size
    )


def _compatible_lora_bmm_expand_slice(
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


_ORIGINAL_EXPAND_SLICE_PREFILL = PunicaWrapperNPU._expand_slice_prefill
_ORIGINAL_EXPAND_SLICE_DECODE = PunicaWrapperNPU._expand_slice_decode
_ORIGINAL_MODEL_MANAGER_INIT = LoRAModelManager.__init__
_ORIGINAL_LOAD_ADAPTER = WorkerLoRAManager._load_adapter
_ORIGINAL_UPDATE_GRAPH_PARAMS = AscendAttentionBackendImpl.update_graph_params


def _expand_slice_prefill(
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
    _ORIGINAL_EXPAND_SLICE_PREFILL(
        self,
        y,
        x,
        lora_b_stacked,
        y_offset,
        y_slice_size,
        add_inputs,
    )


def _expand_slice_decode(
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
    _ORIGINAL_EXPAND_SLICE_DECODE(
        self,
        y,
        x,
        lora_b_stacked,
        y_offset,
        y_slice_size,
        add_inputs,
    )


def _model_manager_init(self, *args, **kwargs) -> None:
    _ORIGINAL_MODEL_MANAGER_INIT(self, *args, **kwargs)
    _enable_language_model_expand_slice(self)


def _load_adapter(self, lora_request):
    lora = _ORIGINAL_LOAD_ADAPTER(self, lora_request)
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
        forward_context.attn_metadata = _filter_fia_metadata(original_metadata)
    try:
        return _ORIGINAL_UPDATE_GRAPH_PARAMS(
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


def _make_no_lora_guard(original_method: Callable[..., Any]):
    def guarded(self, *args, **kwargs):
        if self.no_lora:
            return None
        return original_method(self, *args, **kwargs)

    return guarded


def install(register_func: RegisterFunc) -> None:
    """Install the dense LoRA patch using the host project's registry."""
    if getattr(PunicaWrapperNPU, _PATCH_MARKER, False):
        return

    PunicaWrapperNPU.enable_compatible_lora_bmm_expand_slice = (
        enable_compatible_lora_bmm_expand_slice
    )
    PunicaWrapperNPU._requires_compatible_lora_expand_slice = (
        _requires_compatible_lora_expand_slice
    )
    PunicaWrapperNPU._compatible_lora_bmm_expand_slice = (
        _compatible_lora_bmm_expand_slice
    )

    _register_replacement(
        register_func,
        _expand_slice_prefill,
        _ORIGINAL_EXPAND_SLICE_PREFILL,
    )
    _register_replacement(
        register_func,
        _expand_slice_decode,
        _ORIGINAL_EXPAND_SLICE_DECODE,
    )
    _register_replacement(
        register_func,
        _model_manager_init,
        _ORIGINAL_MODEL_MANAGER_INIT,
    )
    _register_replacement(
        register_func,
        _load_adapter,
        _ORIGINAL_LOAD_ADAPTER,
    )
    _register_replacement(
        register_func,
        update_graph_params,
        _ORIGINAL_UPDATE_GRAPH_PARAMS,
    )

    for method_name in (
        "add_shrink",
        "add_expand",
        "add_lora_embedding",
        "add_lora_linear",
        "add_lora_logits",
    ):
        original_method = getattr(PunicaWrapperNPU, method_name)
        guarded_method = _make_no_lora_guard(original_method)
        _register_replacement(
            register_func,
            guarded_method,
            original_method,
        )

    setattr(PunicaWrapperNPU, _PATCH_MARKER, True)

