# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 language-model LoRA adapter module name compatibility."""

from functools import wraps

from vllm.lora.model_manager import LoRAModelManager
from vllm.lora.worker_manager import WorkerLoRAManager

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.patch.lora_graph_config import is_qwen3_5, specialize_lora, validate_config

_ORIGINAL_MANAGER_INIT = LoRAModelManager.__init__
_ORIGINAL_LOAD_ADAPTER = WorkerLoRAManager._load_adapter


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


@wraps(_ORIGINAL_MANAGER_INIT)
def _manager_init(self, model, max_num_seqs, max_num_batched_tokens, vocab_size, lora_config, device, vllm_config):
    enabled = is_qwen3_5(vllm_config) and specialize_lora(vllm_config)
    if enabled:
        validate_config(vllm_config)
    _ORIGINAL_MANAGER_INIT(
        self, model, max_num_seqs, max_num_batched_tokens, vocab_size, lora_config, device, vllm_config
    )
    self._ascend_qwen3_5_lora = enabled


@wraps(_ORIGINAL_LOAD_ADAPTER)
def _load_adapter(self, lora_request):
    lora = _ORIGINAL_LOAD_ADAPTER(self, lora_request)
    manager = self._adapter_manager
    if getattr(manager, "_ascend_qwen3_5_lora", False) and manager.supports_mm:
        lora.loras = _remap_lora_keys(
            lora.loras, manager.modules, manager.packed_modules_mapping, manager.mm_mapping.language_model
        )
    return lora


def _install():
    if getattr(LoRAModelManager, "_ascend_qwen3_5_lora_names_patch_installed", False):
        return
    if getattr(PunicaWrapperNPU, "_external_qwen3_5_dense_lora_patch", False):
        raise RuntimeError("Remove the external v0.23 Qwen3.5 LoRA patch before using this patch")
    LoRAModelManager.__init__ = _manager_init
    WorkerLoRAManager._load_adapter = _load_adapter
    LoRAModelManager._ascend_qwen3_5_lora_names_patch_installed = True


_install()
