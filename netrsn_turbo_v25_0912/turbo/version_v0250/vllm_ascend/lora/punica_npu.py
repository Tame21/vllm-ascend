# SPDX-License-Identifier: Apache-2.0

"""Dense LoRA graph behavior for Qwen3.5 and Magistral."""

from functools import wraps

from vllm.config import CUDAGraphMode


QWEN3_5 = "qwen3_5"
MAGISTRAL = "magistral"


def model_family(config) -> str | None:
    model_config = config.model_config
    text_type = getattr(model_config.hf_text_config, "model_type", None)
    outer_type = getattr(model_config.hf_config, "model_type", None)
    if text_type == "qwen3_5_text":
        return QWEN3_5
    if outer_type == "mistral3" and text_type == "mistral":
        return MAGISTRAL
    return None


def patch_applies(config) -> bool:
    return config.lora_config is not None and model_family(config) is not None


def needs_attention_metadata_filter(config) -> bool:
    return patch_applies(config) and model_family(config) == QWEN3_5


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
    family = model_family(config)
    model_name = "Qwen3.5" if family == QWEN3_5 else "Magistral"
    if config.use_v2_model_runner:
        raise ValueError(f"The {model_name} LoRA patch supports model runner v1 only")
    if config.lora_config.max_lora_rank > 64:
        raise ValueError(
            f"The {model_name} LoRA patch supports max_lora_rank <= 64 only"
        )
    speculative_config = config.speculative_config
    if speculative_config is not None and (
        family != QWEN3_5 or speculative_config.method != "mtp"
    ):
        raise ValueError(
            "Only Qwen3.5 MTP speculative decoding is supported by this LoRA patch"
        )
    parallel = config.parallel_config
    if (
        parallel.prefill_context_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
    ):
        raise ValueError(f"The {model_name} LoRA patch does not support context parallelism")
    if parallel.enable_dbo or parallel.ubatch_size > 1:
        raise ValueError(f"The {model_name} LoRA patch does not support microbatching")
    if config.model_config.enable_sleep_mode:
        raise ValueError(
            f"The {model_name} LoRA patch has not been validated with sleep mode"
        )


def wrap_update_metadata(original):
    @wraps(original)
    def update_metadata(self, mapping, *args, **kwargs):
        result = original(self, mapping, *args, **kwargs)
        if getattr(self, "_ascend_dense_lora_graph", False):
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
