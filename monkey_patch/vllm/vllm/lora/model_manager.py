"""LoRAModelManager replacement for wrapped Qwen3.5 language models."""

from functools import wraps
from typing import Any

from vllm.lora.model_manager import LoRAModelManager


ORIGINAL_INIT = LoRAModelManager.__init__


def _enable_language_model_expand_slice(manager: Any) -> None:
    if not getattr(manager, "supports_mm", False):
        return

    mm_mapping = getattr(manager, "mm_mapping", None)
    language_prefixes = getattr(mm_mapping, "language_model", ())
    wrapper_mapping = getattr(manager, "punica_wrapper_mapping", {})
    seen_wrappers: set[int] = set()
    for prefix in language_prefixes:
        if not prefix:
            continue
        wrapper = wrapper_mapping.get(prefix)
        if wrapper is None or id(wrapper) in seen_wrappers:
            continue
        enable = getattr(
            wrapper,
            "enable_compatible_lora_bmm_expand_slice",
            None,
        )
        if enable is not None:
            enable()
            seen_wrappers.add(id(wrapper))


def init_wrapper(original_init):
    """Wrap LoRAModelManager.__init__ and run post-init setup."""

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        _enable_language_model_expand_slice(self)

    return wrapped_init
