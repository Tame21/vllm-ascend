# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 dense LoRA behavior for ``PunicaWrapperNPU``."""

from functools import wraps

import torch.nn.functional as F
from vllm.config import CUDAGraphMode

_DIRECT_LORA_OPS_MODULE = "vllm_ascend.lora.lora_ops"
_LORA_RANK_ALIGNMENT = 8


def patch_applies(config) -> bool:
    return bool(
        config.lora_config is not None
        and getattr(
            config.model_config.hf_text_config,
            "model_type",
            None,
        )
        == "qwen3_5_text"
    )


def specialize_lora(config) -> bool:
    return bool(
        patch_applies(config)
        and not config.model_config.enforce_eager
        and config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and config.compilation_config.cudagraph_specialize_lora
    )


def validate_config(config) -> None:
    if not patch_applies(config):
        return
    if config.use_v2_model_runner:
        raise ValueError("The Qwen3.5 LoRA patch supports model runner v1 only")
    if config.lora_config.max_lora_rank > 64:
        raise ValueError(
            "The Qwen3.5 LoRA patch currently supports max_lora_rank <= 64 only"
        )
    speculative_config = config.speculative_config
    if speculative_config is not None and speculative_config.method != "mtp":
        raise ValueError(
            "The Qwen3.5 LoRA patch supports MTP speculative decoding only"
        )
    parallel = config.parallel_config
    if (
        parallel.prefill_context_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
    ):
        raise ValueError("The Qwen3.5 LoRA patch does not support context parallelism")
    if parallel.enable_dbo or parallel.ubatch_size > 1:
        raise ValueError("The Qwen3.5 LoRA patch does not support microbatching")
    if config.model_config.enable_sleep_mode:
        raise ValueError(
            "The Qwen3.5 LoRA patch has not been validated with sleep mode"
        )


def pad_lora_rank(rank):
    """Align the FP32 shrink/expand dimension to an AscendC data block."""
    return ((rank + _LORA_RANK_ALIGNMENT - 1) // _LORA_RANK_ALIGNMENT) * (
        _LORA_RANK_ALIGNMENT
    )


def wrap_shrink_with_padding(original):
    @wraps(original)
    def shrink(inputs, weights, output, *args, **kwargs):
        rank = output.shape[-1]
        padded_rank = pad_lora_rank(rank)
        if padded_rank == rank:
            return original(inputs, weights, output, *args, **kwargs)

        rank_padding = padded_rank - rank
        padded_output = F.pad(output, (0, rank_padding))
        padded_weights = F.pad(
            weights,
            (0, 0, 0, rank_padding),
        )
        result = original(
            inputs,
            padded_weights,
            padded_output,
            *args,
            **kwargs,
        )
        output.copy_(padded_output[..., :rank])
        return result

    return shrink


def wrap_expand_with_padding(original):
    """Align the FP32 shrink result and LoRA-B rank for AscendC expand."""

    @wraps(original)
    def expand(inputs, weights, output, *args, **kwargs):
        rank = weights.shape[-1]
        padded_rank = pad_lora_rank(rank)
        if padded_rank == rank:
            return original(inputs, weights, output, *args, **kwargs)
        if inputs.shape[-1] < rank:
            raise ValueError("LoRA expand input rank is smaller than LoRA-B rank")

        rank_padding = padded_rank - rank
        padded_inputs = F.pad(inputs[..., :rank], (0, rank_padding))
        padded_weights = F.pad(weights, (0, rank_padding))
        return original(
            padded_inputs,
            padded_weights,
            output,
            *args,
            **kwargs,
        )

    return expand


def wrap_init(original):
    @wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        shrink_module = getattr(self.bgmv_shrink, "__module__", "")
        if shrink_module != _DIRECT_LORA_OPS_MODULE:
            return
        self.bgmv_shrink = wrap_shrink_with_padding(self.bgmv_shrink)
        self.sgmv_shrink = wrap_shrink_with_padding(self.sgmv_shrink)
        self.bgmv_expand = wrap_expand_with_padding(self.bgmv_expand)
        self.bgmv_expand_slice = wrap_expand_with_padding(self.bgmv_expand_slice)
        self.sgmv_expand = wrap_expand_with_padding(self.sgmv_expand)
        self.sgmv_expand_slice = wrap_expand_with_padding(self.sgmv_expand_slice)

    return init


def wrap_update_metadata(original):
    @wraps(original)
    def update_metadata(self, mapping, *args, **kwargs):
        result = original(self, mapping, *args, **kwargs)
        if getattr(self, "_ascend_qwen3_5_lora", False):
            self.no_lora = not any(
                adapter_id > 0 for adapter_id in mapping.index_mapping
            )
        return result

    return update_metadata


def no_lora_guard(original):
    @wraps(original)
    def guarded(self, *args, **kwargs):
        if getattr(self, "_ascend_specialize_lora", False) and self.no_lora:
            return None
        return original(self, *args, **kwargs)

    return guarded
