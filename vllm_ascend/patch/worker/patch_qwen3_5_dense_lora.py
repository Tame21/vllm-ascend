# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 dense LoRA fixes for the vLLM 0.25.1 release lane."""

from functools import wraps

from vllm.config import CUDAGraphMode
from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA
from vllm.lora.model_manager import LoRAModelManager
from vllm.lora.worker_manager import WorkerLoRAManager

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

_NPU_COPY_BLOCK_BYTES = 32
_SUPPORTED_QWEN_LORA_RANKS = (8, 16)


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
    if config.lora_config.max_lora_rank not in _SUPPORTED_QWEN_LORA_RANKS:
        raise ValueError("The Qwen3.5 LoRA patch currently supports max_lora_rank 8 or 16 only")
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


def _aligned_copy_elements(elements: int, element_size: int) -> int:
    """Round an element count up to one NPU copy block."""
    if element_size <= 0 or _NPU_COPY_BLOCK_BYTES % element_size != 0:
        raise ValueError(f"Unsupported Qwen3.5 LoRA element size: {element_size}")
    elements_per_block = _NPU_COPY_BLOCK_BYTES // element_size
    return ((elements + elements_per_block - 1) // elements_per_block) * elements_per_block


def _wrap_expand_slice(original):
    @wraps(original)
    def expand(self, y, x, w_t_all, y_offset, y_slice_size, add_inputs):
        if not getattr(self, "_ascend_qwen3_5_lora", False):
            return original(self, y, x, w_t_all, y_offset, y_slice_size, add_inputs)
        if getattr(self, "_ascend_specialize_lora", False) and self.no_lora:
            return None

        if w_t_all.ndim not in (3, 4):
            raise ValueError("Incompatible Qwen3.5 LoRA-B slice shape")
        rank = w_t_all.shape[-1]
        if rank not in _SUPPORTED_QWEN_LORA_RANKS:
            raise ValueError(f"Unsupported Qwen3.5 LoRA-B rank: {rank}")
        if (
            w_t_all.shape[-2] != y_slice_size
            or x.shape[-1] < rank
            or y_slice_size <= 0
            or y_offset < 0
            or y_offset + y_slice_size > y.shape[-1]
        ):
            raise ValueError("Incompatible Qwen3.5 LoRA-B slice shape")

        element_size = y.element_size()
        aligned_width = _aligned_copy_elements(y_slice_size, element_size)
        output_is_aligned = (
            aligned_width == y_slice_size
            and y_offset * element_size % _NPU_COPY_BLOCK_BYTES == 0
            and y.shape[-1] * element_size % _NPU_COPY_BLOCK_BYTES == 0
        )
        kernel_input = x if x.shape[-1] == rank else x[..., :rank].contiguous()
        if output_is_aligned and y_slice_size >= rank and add_inputs:
            return original(self, y, kernel_input, w_t_all, y_offset, y_slice_size, True)

        # The native expand kernel supports rank 8/16, requires rank <=
        # output width, and copies output in complete 32-byte blocks. Isolate
        # the packed slice in an aligned temporary, padding LoRA-B rows with
        # zeros, then expose only the real slice to the caller.
        kernel_width = max(rank, aligned_width)
        target = y[..., y_offset : y_offset + y_slice_size]
        padded_output = y.new_zeros((*y.shape[:-1], kernel_width))
        if add_inputs:
            padded_output[..., :y_slice_size].copy_(target)

        kernel_weights = w_t_all
        if kernel_width != y_slice_size:
            padded_weight_shape = list(w_t_all.shape)
            padded_weight_shape[-2] = kernel_width
            kernel_weights = w_t_all.new_zeros(tuple(padded_weight_shape))
            kernel_weights[..., :y_slice_size, :].copy_(w_t_all)

        result = original(self, padded_output, kernel_input, kernel_weights, 0, kernel_width, True)
        target.copy_(padded_output[..., :y_slice_size])
        return result

    return expand


def _wrap_shrink_kernel(original):
    @wraps(original)
    def shrink(inputs, lora_a_weights, output_tensor, *args, **kwargs):
        rank = output_tensor.shape[-1]
        rank_bytes = rank * output_tensor.element_size()
        if rank_bytes % _NPU_COPY_BLOCK_BYTES == 0:
            return original(inputs, lora_a_weights, output_tensor, *args, **kwargs)
        if lora_a_weights.ndim not in (3, 4) or lora_a_weights.shape[-2] != rank:
            raise ValueError("Incompatible Qwen3.5 LoRA-A shrink shape")

        # AscendC DataCopy writes complete 32-byte blocks. Pad both the
        # LoRA-A rank dimension and the destination row so the native kernel
        # sees an aligned rank, then expose only the real rank downstream.
        aligned_rank = _aligned_copy_elements(rank, output_tensor.element_size())
        padded_output = output_tensor.new_zeros((*output_tensor.shape[:-1], aligned_rank))
        padded_output[..., :rank].copy_(output_tensor)

        padded_weight_shape = list(lora_a_weights.shape)
        padded_weight_shape[-2] = aligned_rank
        padded_weights = lora_a_weights.new_zeros(tuple(padded_weight_shape))
        padded_weights[..., :rank, :].copy_(lora_a_weights)

        result = original(inputs, padded_weights, padded_output, *args, **kwargs)
        output_tensor.copy_(padded_output[..., :rank])
        return result

    return shrink


def _install_shrink_padding(wrapper):
    if getattr(wrapper, "_ascend_qwen3_5_shrink_padding_installed", False):
        return
    wrapper.bgmv_shrink = _wrap_shrink_kernel(wrapper.bgmv_shrink)
    wrapper.sgmv_shrink = _wrap_shrink_kernel(wrapper.sgmv_shrink)
    wrapper._ascend_qwen3_5_shrink_padding_installed = True


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
_ORIGINAL_EXPAND_PACKED_LORA = MergedColumnParallelLinearWithLoRA.expand_packed_lora


@wraps(_ORIGINAL_EXPAND_PACKED_LORA)
def _expand_packed_lora(self, lora_a, lora_b):
    if not getattr(self, "_ascend_qwen3_5_qkvz_lora", False) or all(b is not None for b in lora_b):
        return _ORIGINAL_EXPAND_PACKED_LORA(self, lora_a, lora_b)

    # Qwen3.5 in_proj_qkvz has four slices, but the checkpoint groups them
    # as [qkv, z]. A partial adapter leaves either group as None. The upstream
    # expansion reads b_i.shape without checking for None, so preserve the
    # absent group as empty per-slice entries instead of materializing zeros.
    if len(lora_a) != 2 or len(lora_b) != 2 or self.n_slices != 4 or len(self.output_sizes) != 4:
        raise ValueError("Incompatible Qwen3.5 in_proj_qkvz LoRA groups")
    expanded_a = []
    expanded_b = []
    start = 0
    for a_i, b_i, count in zip(lora_a, lora_b, (3, 1)):
        sizes = self.output_sizes[start : start + count]
        if (a_i is None) != (b_i is None):
            raise ValueError("Qwen3.5 in_proj_qkvz LoRA-A/B groups must both be present or absent")
        if b_i is None:
            expanded_a.extend([None] * count)
            expanded_b.extend([None] * count)
        else:
            if b_i.shape[0] != sum(sizes):
                raise ValueError("Incompatible Qwen3.5 in_proj_qkvz LoRA-B group width")
            expanded_a.extend([a_i] * count)
            expanded_b.extend(b_i.split(sizes, dim=0))
        start += count
    return expanded_a, expanded_b


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
        _install_shrink_padding(wrapper)
    for name, module in self.modules.items():
        if name.rpartition(".")[-1] == "in_proj_qkvz" and isinstance(module, MergedColumnParallelLinearWithLoRA):
            module._ascend_qwen3_5_qkvz_lora = True


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
    MergedColumnParallelLinearWithLoRA.expand_packed_lora = _expand_packed_lora
    PunicaWrapperNPU._ascend_qwen3_5_patch_installed = True


_install()
