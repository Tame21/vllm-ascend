# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 dense LoRA initialization for ``LoRAModelManager``."""

from functools import wraps

from netrsn_turbo.turbo.version_0251.vllm_ascend.lora.punica_npu import (
    patch_applies,
    specialize_lora,
    validate_config,
)


def module_candidates(key, packed_modules_mapping):
    yield key
    prefix, separator, suffix = key.rpartition(".")
    for packed_name, children in packed_modules_mapping.items():
        if suffix in children:
            yield f"{prefix}.{packed_name}" if separator else packed_name


def remap_lora_keys(
    loras,
    module_names,
    packed_modules_mapping,
    language_prefixes,
):
    """Only add a language prefix when it resolves to an existing module."""
    names = set(module_names)
    remapped = {}
    for key, weights in loras.items():
        candidates = tuple(module_candidates(key, packed_modules_mapping))
        target = key
        if not any(candidate in names for candidate in candidates):
            matches = {
                prefix.rstrip(".") + "." + key
                for prefix in language_prefixes
                if prefix
                and any(
                    prefix.rstrip(".") + "." + candidate in names
                    for candidate in candidates
                )
            }
            if len(matches) > 1:
                raise ValueError(f"Ambiguous Qwen3.5 LoRA module prefix: {key}")
            if matches:
                target = matches.pop()
        if target in remapped:
            raise ValueError(
                f"Duplicate Qwen3.5 LoRA module after prefix mapping: {target}"
            )
        remapped[target] = weights
    return remapped


def wrap_init(original):
    @wraps(original)
    def init(
        self,
        model,
        max_num_seqs,
        max_num_batched_tokens,
        vocab_size,
        lora_config,
        device,
        vllm_config,
    ):
        enabled = patch_applies(vllm_config)
        validate_config(vllm_config)
        original(
            self,
            model,
            max_num_seqs,
            max_num_batched_tokens,
            vocab_size,
            lora_config,
            device,
            vllm_config,
        )
        self._ascend_qwen3_5_lora = enabled
        if not enabled:
            return
        prefixes = (
            self.mm_mapping.language_model
            if self.supports_mm
            else tuple(self.punica_wrapper_mapping)
        )
        for prefix in prefixes:
            wrapper = self.punica_wrapper_mapping[prefix]
            wrapper._ascend_qwen3_5_lora = True
            wrapper._ascend_specialize_lora = specialize_lora(vllm_config)

    return init
