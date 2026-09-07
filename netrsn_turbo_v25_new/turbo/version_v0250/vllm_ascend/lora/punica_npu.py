# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 dense LoRA behavior for ``PunicaWrapperNPU``."""

from functools import wraps

from vllm.config import CUDAGraphMode


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
