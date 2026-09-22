# SPDX-License-Identifier: Apache-2.0
"""Shared model matching for the release-lane LoRA graph patches."""

from vllm.config import CUDAGraphMode


def is_qwen3_5(config) -> bool:
    model = config.model_config
    return model is not None and getattr(model.hf_text_config, "model_type", None) == "qwen3_5_text"


def is_mistral3_family(config) -> bool:
    model = config.model_config
    if model is None or getattr(model.hf_text_config, "model_type", None) != "mistral":
        return False
    hf_config = model.hf_config
    architectures = set(getattr(hf_config, "architectures", None) or ())
    return bool(
        "magistral" in str(getattr(model, "model", "")).lower()
        or getattr(hf_config, "model_type", None) == "mistral3"
        or "Mistral3ForConditionalGeneration" in architectures
        or (model.multimodal_config is not None and "MistralForCausalLM" in architectures)
    )


def patch_applies(config) -> bool:
    return config.lora_config is not None and (is_qwen3_5(config) or is_mistral3_family(config))


def specialize_lora(config) -> bool:
    return bool(
        patch_applies(config)
        and not config.model_config.enforce_eager
        and config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and config.compilation_config.cudagraph_specialize_lora
    )


def validate_config(config):
    if not specialize_lora(config):
        return
    if config.use_v2_model_runner:
        raise ValueError("The LoRA graph/AOT patch supports model runner v1 only")
    rank = config.lora_config.max_lora_rank
    if is_qwen3_5(config) and rank not in (8, 16):
        raise ValueError("The Qwen3.5 LoRA graph/AOT patch supports max_lora_rank 8 or 16 only")
    if rank > 64:
        raise ValueError("The LoRA graph/AOT patch supports max_lora_rank <= 64 only")
    if is_mistral3_family(config) and config.lora_config.enable_tower_connector_lora:
        raise ValueError("The Magistral LoRA graph/AOT patch supports language-model adapters only")
    if config.speculative_config is not None and config.speculative_config.method != "mtp":
        raise ValueError("The LoRA graph/AOT patch supports MTP speculative decoding only")
    parallel = config.parallel_config
    if parallel.prefill_context_parallel_size > 1 or parallel.decode_context_parallel_size > 1:
        raise ValueError("The LoRA graph/AOT patch does not support context parallelism")
    if parallel.enable_dbo or parallel.ubatch_size > 1:
        raise ValueError("The LoRA graph/AOT patch does not support microbatching")
    if config.model_config.enable_sleep_mode:
        raise ValueError("The LoRA graph/AOT patch has not been validated with sleep mode")
