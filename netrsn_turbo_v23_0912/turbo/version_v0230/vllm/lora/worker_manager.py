# SPDX-License-Identifier: Apache-2.0

"""Multimodal language-backbone LoRA key remapping."""

from functools import wraps

from netrsn_turbo.turbo.version_v0230.vllm.lora.model_manager import (
    remap_lora_keys,
)


def wrap_load_adapter(original):
    @wraps(original)
    def load_adapter(self, lora_request):
        lora = original(self, lora_request)
        manager = self._adapter_manager
        if getattr(manager, "_ascend_dense_lora_graph", False) and manager.supports_mm:
            lora.loras = remap_lora_keys(
                lora.loras,
                manager.modules,
                manager.packed_modules_mapping,
                manager.mm_mapping.language_model,
            )
        return lora

    return load_adapter

