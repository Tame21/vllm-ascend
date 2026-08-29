# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 dense LoRA fixes for the vLLM 0.25.1 release lane."""

from copy import copy
from functools import wraps

import torch
from vllm.config import CUDAGraphMode
from vllm.lora.model_manager import LoRAModelManager
from vllm.lora.worker_manager import WorkerLoRAManager

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def patch_applies(config) -> bool:
    return bool(
        config.lora_config is not None
        and getattr(config.model_config.hf_text_config, "model_type", None) == "qwen3_5_text"
    )


def specialize_lora(config) -> bool:
    return bool(
        patch_applies(config)
        and not config.model_config.enforce_eager
        and config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and config.compilation_config.cudagraph_specialize_lora
    )


def validate_config(config):
    if not patch_applies(config):
        return
    if config.use_v2_model_runner:
        raise ValueError("The Qwen3.5 LoRA patch supports model runner v1 only")
    if config.lora_config.max_lora_rank > 64:
        raise ValueError("The Qwen3.5 LoRA patch currently supports max_lora_rank <= 64 only")
    speculative_config = config.speculative_config
    if speculative_config is not None and speculative_config.method != "mtp":
        raise ValueError("The Qwen3.5 LoRA patch supports MTP speculative decoding only")
    parallel = config.parallel_config
    if parallel.prefill_context_parallel_size > 1 or parallel.decode_context_parallel_size > 1:
        raise ValueError("The Qwen3.5 LoRA patch does not support context parallelism")
    if parallel.enable_dbo or parallel.ubatch_size > 1:
        raise ValueError("The Qwen3.5 LoRA patch does not support microbatching")
    if config.model_config.enable_sleep_mode:
        raise ValueError("The Qwen3.5 LoRA patch has not been validated with sleep mode")


@torch.library.custom_op("vllm_ascend::qwen3_5_lora_expand_slice", mutates_args={"y"})
def _lora_expand_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    offset: int,
    width: int,
    add_inputs: bool,
) -> None:
    # Bound the gathered [tokens, output_width, rank] temporary for prefill.
    rows_per_chunk = 128
    rank = weights.shape[-1]
    if rank > x.shape[-1] or weights.shape[-2] != width:
        raise ValueError("Incompatible Qwen3.5 LoRA-B slice shape")
    for start in range(0, x.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, x.shape[0])
        token_indices = indices[start:end]
        selected = weights[token_indices.clamp(min=0).long(), 0].to(y.dtype)
        inputs = x[start:end, :rank].to(y.dtype)
        delta = (inputs.unsqueeze(1) * selected).sum(dim=-1)
        delta = torch.where((token_indices >= 0).unsqueeze(-1), delta, torch.zeros_like(delta))
        target = y[start:end, offset : offset + width]
        if add_inputs:
            target.add_(delta)
        else:
            target.copy_(delta)


@_lora_expand_slice.register_fake
def _lora_expand_slice_fake(y, x, weights, indices, offset, width, add_inputs) -> None:
    return None


def _wrap_expand_slice(original):
    @wraps(original)
    def expand(self, y, x, w_t_all, y_offset, y_slice_size, add_inputs):
        if not getattr(self, "_ascend_qwen3_5_lora", False):
            return original(self, y, x, w_t_all, y_offset, y_slice_size, add_inputs)
        if getattr(self, "_ascend_specialize_lora", False) and self.no_lora:
            return None
        _lora_expand_slice(y, x, w_t_all, self._get_token_lora_indices(x), y_offset, y_slice_size, add_inputs)

    return expand


def _module_candidates(key, packed_modules_mapping):
    yield key
    prefix, separator, suffix = key.rpartition(".")
    for packed_name, children in packed_modules_mapping.items():
        if suffix in children:
            yield f"{prefix}.{packed_name}" if separator else packed_name


def _remap_lora_keys(loras, module_names, packed_modules_mapping, language_prefixes):
    """Only add a language prefix when it resolves to an existing module."""
    names = set(module_names)
    remapped = {}
    for key, weights in loras.items():
        candidates = tuple(_module_candidates(key, packed_modules_mapping))
        target = key
        if not any(candidate in names for candidate in candidates):
            matches = {
                prefix.rstrip(".") + "." + key
                for prefix in language_prefixes
                if prefix and any(prefix.rstrip(".") + "." + candidate in names for candidate in candidates)
            }
            if len(matches) > 1:
                raise ValueError(f"Ambiguous Qwen3.5 LoRA module prefix: {key}")
            if matches:
                target = matches.pop()
        if target in remapped:
            raise ValueError(f"Duplicate Qwen3.5 LoRA module after prefix mapping: {target}")
        remapped[target] = weights
    return remapped


_ORIGINAL_MANAGER_INIT = LoRAModelManager.__init__
_ORIGINAL_LOAD_ADAPTER = WorkerLoRAManager._load_adapter
_ORIGINAL_UPDATE_METADATA = PunicaWrapperNPU.update_metadata
_ORIGINAL_UPDATE_GRAPH_PARAMS = AscendAttentionBackendImpl.update_graph_params


@wraps(_ORIGINAL_MANAGER_INIT)
def _manager_init(self, model, max_num_seqs, max_num_batched_tokens, vocab_size, lora_config, device, vllm_config):
    enabled = patch_applies(vllm_config)
    validate_config(vllm_config)
    _ORIGINAL_MANAGER_INIT(
        self, model, max_num_seqs, max_num_batched_tokens, vocab_size, lora_config, device, vllm_config
    )
    self._ascend_qwen3_5_lora = enabled
    if not enabled:
        return
    prefixes = self.mm_mapping.language_model if self.supports_mm else tuple(self.punica_wrapper_mapping)
    for prefix in prefixes:
        wrapper = self.punica_wrapper_mapping[prefix]
        wrapper._ascend_qwen3_5_lora = True
        wrapper._ascend_specialize_lora = specialize_lora(vllm_config)


@wraps(_ORIGINAL_LOAD_ADAPTER)
def _load_adapter(self, lora_request):
    lora = _ORIGINAL_LOAD_ADAPTER(self, lora_request)
    manager = self._adapter_manager
    if getattr(manager, "_ascend_qwen3_5_lora", False) and manager.supports_mm:
        lora.loras = _remap_lora_keys(
            lora.loras, manager.modules, manager.packed_modules_mapping, manager.mm_mapping.language_model
        )
    return lora


@wraps(_ORIGINAL_UPDATE_METADATA)
def _update_metadata(self, mapping, *args, **kwargs):
    result = _ORIGINAL_UPDATE_METADATA(self, mapping, *args, **kwargs)
    if getattr(self, "_ascend_qwen3_5_lora", False):
        # The upstream decode path leaves no_lora unchanged. Use the CPU
        # mapping, avoiding a device synchronization and stale prefill state.
        self.no_lora = not any(adapter_id > 0 for adapter_id in mapping.index_mapping)
    return result


def _no_lora_guard(original):
    @wraps(original)
    def guarded(self, *args, **kwargs):
        if getattr(self, "_ascend_specialize_lora", False) and self.no_lora:
            return None
        return original(self, *args, **kwargs)

    return guarded


def update_graph_params(
    update_stream, forward_context, num_tokens, vllm_config, speculative_config=None, draft_attn_metadatas=None
):
    if (
        patch_applies(vllm_config)
        and not _EXTRA_CTX.is_draft_model
        and not using_paged_attention(num_tokens, vllm_config)
        and isinstance(forward_context.attn_metadata, dict)
    ):
        # Do not mutate the shared context: GDN still needs its own metadata.
        filtered_context = copy(forward_context)
        filtered_context.attn_metadata = {
            key: metadata
            for key, metadata in forward_context.attn_metadata.items()
            if hasattr(metadata, "seq_lens_list") and hasattr(metadata, "actual_seq_lengths_q")
        }
        forward_context = filtered_context
    return _ORIGINAL_UPDATE_GRAPH_PARAMS(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config,
        draft_attn_metadatas=draft_attn_metadatas,
    )


def _install():
    if getattr(PunicaWrapperNPU, "_ascend_qwen3_5_patch_installed", False):
        return
    if getattr(PunicaWrapperNPU, "_external_qwen3_5_dense_lora_patch", False):
        raise RuntimeError("Remove the external v0.23 Qwen3.5 LoRA patch before using this patch")
    PunicaWrapperNPU._expand_slice_prefill = _wrap_expand_slice(PunicaWrapperNPU._expand_slice_prefill)
    PunicaWrapperNPU._expand_slice_decode = _wrap_expand_slice(PunicaWrapperNPU._expand_slice_decode)
    PunicaWrapperNPU.update_metadata = _update_metadata
    for name in ("add_shrink", "add_expand", "add_lora_embedding", "add_lora_linear", "add_lora_logits"):
        setattr(PunicaWrapperNPU, name, _no_lora_guard(getattr(PunicaWrapperNPU, name)))
    LoRAModelManager.__init__ = _manager_init
    WorkerLoRAManager._load_adapter = _load_adapter
    AscendAttentionBackendImpl.update_graph_params = staticmethod(update_graph_params)
    PunicaWrapperNPU._ascend_qwen3_5_patch_installed = True


_install()
