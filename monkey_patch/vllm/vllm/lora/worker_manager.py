"""WorkerLoRAManager replacement for wrapped Qwen3.5 adapter names."""

from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.lora.worker_manager import (
    LRUCacheWorkerLoRAManager,
    WorkerLoRAManager,
)

logger = init_logger(__name__)

ORIGINAL_LOAD_ADAPTER = WorkerLoRAManager._load_adapter
ORIGINAL_CREATE_LORA_MANAGER = WorkerLoRAManager.create_lora_manager
ORIGINAL_LRU_CREATE_LORA_MANAGER = (
    LRUCacheWorkerLoRAManager.create_lora_manager
)


def _enable_language_model_expand_slice(manager) -> None:
    if not getattr(manager, "supports_mm", False):
        return

    wrapper_mapping = getattr(manager, "punica_wrapper_mapping", {})
    language_prefixes = getattr(
        getattr(manager, "mm_mapping", None),
        "language_model",
        (),
    )
    for prefix in language_prefixes:
        wrapper = wrapper_mapping.get(prefix)
        enable = getattr(
            wrapper,
            "enable_compatible_lora_bmm_expand_slice",
            None,
        )
        if enable is not None:
            enable()


def create_lora_manager(self, model, vllm_config=None):
    model = ORIGINAL_CREATE_LORA_MANAGER(self, model, vllm_config)
    _enable_language_model_expand_slice(self._adapter_manager)
    return model


def create_lora_manager_lru(self, model, vllm_config=None):
    model = ORIGINAL_LRU_CREATE_LORA_MANAGER(self, model, vllm_config)
    _enable_language_model_expand_slice(self._adapter_manager)
    return model


def _detect_language_model_prefix(
    lora_keys: Iterable[str],
    model_module_names: Iterable[str],
    packed_modules_mapping: dict[str, list[str]] | None = None,
) -> str | None:
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


def load_adapter(self, lora_request):
    lora = ORIGINAL_LOAD_ADAPTER(self, lora_request)
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
