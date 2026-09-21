# SPDX-License-Identifier: Apache-2.0
"""Optional three-variant AOT extension of patch_lora_acl_graph on v0.25.1.

Import after the graph patch and before constructing model wrappers. Ordinary
torch.compile routing and ACL graph fixes remain owned by the graph patch.
"""

import os
from contextlib import nullcontext
from functools import wraps
from types import MethodType

import torch
import vllm.envs as vllm_envs
from vllm.compilation import decorators, monitor
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs
from vllm_ascend.patch.worker import patch_lora_acl_graph as lora_graph

_ORIGINAL_WRAPPER_INIT = TorchCompileWithNoGuardsWrapper.__init__
_ORIGINAL_WRAPPER_AOT_COMPILE = TorchCompileWithNoGuardsWrapper.aot_compile
_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN = decorators._try_load_aot_compiled_fn
_ORIGINAL_COMPILE_BASE_VARIANT = lora_graph._compile_base_variant
_AOT_VARIANTS = (lora_graph._VARIANT_LORA, lora_graph._VARIANT_BASE, lora_graph._VARIANT_BASE_ONE)


@wraps(_ORIGINAL_COMPILE_BASE_VARIANT)
def _compile_base_variant(self, prefix, suffix):
    aot_context = nullcontext()
    if vllm_envs.VLLM_USE_AOT_COMPILE and hasattr(torch._dynamo.config, "enable_aot_compile"):
        aot_context = torch._dynamo.config.patch(enable_aot_compile=True)
    with aot_context:
        return _ORIGINAL_COMPILE_BASE_VARIANT(self, prefix, suffix)


@wraps(_ORIGINAL_WRAPPER_INIT)
def _wrapper_init(self, compile_prefix="", is_encoder=False):
    _ORIGINAL_WRAPPER_INIT(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
    if vllm_envs.VLLM_USE_AOT_COMPILE and getattr(self, "_ascend_specialize_lora", False):
        self.save_aot_compiled_function = MethodType(_save_aot_compiled_function, self)


def _variant_aot_path(aot_compilation_path, variant):
    return f"{aot_compilation_path}.{variant}"


class _AOTVariantDispatcher:
    """Route calls to separately compiled base and LoRA AOT artifacts."""

    def __init__(self):
        self.artifacts = {}
        self.dirty_variants = set()

    def add_loaded(self, variant, artifact):
        self.artifacts[variant] = artifact

    def compile_variant(self, model, variant, args, kwargs):
        if variant != lora_graph._VARIANT_BASE_ONE:
            model._ascend_mark_variant_dynamic_inputs(variant, *args, **kwargs)
        compiled_callable = lora_graph._variant_callable(model, variant)
        if not hasattr(compiled_callable, "aot_compile"):
            raise RuntimeError(f"AOT compile is unavailable for the LoRA {variant} callable")
        self.artifacts[variant] = compiled_callable.aot_compile((args, kwargs))
        self.dirty_variants.add(variant)

    def save_dirty(self, model):
        if vllm_envs.VLLM_DISABLE_COMPILE_CACHE:
            return
        cache_dir = getattr(model, "_aot_cache_dir", None)
        aot_compilation_path = getattr(model, "_aot_compilation_path", None)
        if cache_dir is None or aot_compilation_path is None:
            raise RuntimeError("AOT cache paths were not initialized")
        os.makedirs(cache_dir, exist_ok=True)
        for variant in _AOT_VARIANTS:
            if variant not in self.dirty_variants:
                continue
            path = _variant_aot_path(aot_compilation_path, variant)
            tmp_file = f"{path}.{os.getpid()}.tmp"
            try:
                self.artifacts[variant].save_compiled_function(tmp_file)
                os.replace(tmp_file, path)
                self.dirty_variants.remove(variant)
                compilation_counter.num_aot_artifacts_saved += 1
                logger.info("Saved LoRA %s AOT artifact to %s", variant, path)
            except Exception as error:
                logger.warning("Unable to save LoRA %s AOT artifact to %s: %s", variant, path, error)
                try:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except OSError:
                    logger.warning("Unable to remove temporary AOT artifact %s", tmp_file)

    def __call__(self, model, *args, **kwargs):
        variant = lora_graph._select_variant(model)
        if variant not in self.artifacts:
            with monitor.monitor_torch_compile(model.vllm_config, is_encoder=model._is_encoder):
                self.compile_variant(model, variant, args, kwargs)
                compilation_counter.num_aot_compiles += 1
            self.save_dirty(model)
        return self.artifacts[variant](model, *args, **kwargs)


def _aot_compile(self, *args, **kwargs):
    if not getattr(self, "_ascend_specialize_lora", False):
        return _ORIGINAL_WRAPPER_AOT_COMPILE(self, *args, **kwargs)
    dispatcher = getattr(self, "_ascend_preloaded_aot_dispatcher", None)
    if dispatcher is None:
        dispatcher = _AOTVariantDispatcher()
    else:
        del self._ascend_preloaded_aot_dispatcher
    variant = lora_graph._select_variant(self)
    if variant not in dispatcher.artifacts:
        dispatcher.compile_variant(self, variant, args, kwargs)
    return dispatcher


def _save_aot_compiled_function(self):
    dispatcher = self.aot_compiled_fn
    if not isinstance(dispatcher, _AOTVariantDispatcher):
        raise RuntimeError("Expected the LoRA AOT variant dispatcher")
    dispatcher.save_dirty(self)


def _try_load_aot_compiled_fn(model, aot_compilation_path):
    if not getattr(model, "_ascend_specialize_lora", False):
        return _ORIGINAL_TRY_LOAD_AOT_COMPILED_FN(model, aot_compilation_path)
    dispatcher = _AOTVariantDispatcher()
    for variant in _AOT_VARIANTS:
        artifact = _ORIGINAL_TRY_LOAD_AOT_COMPILED_FN(model, _variant_aot_path(aot_compilation_path, variant))
        if artifact is not None:
            dispatcher.add_loaded(variant, artifact)
    if not dispatcher.artifacts:
        return None
    model._aot_compilation_path = aot_compilation_path
    model._aot_cache_dir = os.path.dirname(aot_compilation_path)
    selected_variant = lora_graph._select_variant(model)
    if selected_variant not in dispatcher.artifacts:
        model._ascend_preloaded_aot_dispatcher = dispatcher
        return None
    return dispatcher


def _install():
    # Also honor the startup switch when the module is imported directly.
    if not ascend_envs.VLLM_ASCEND_ENABLE_LORA_AOT:
        return
    if getattr(TorchCompileWithNoGuardsWrapper, "_ascend_lora_aot_patch_installed", False):
        return
    if not getattr(TorchCompileWithNoGuardsWrapper, "_ascend_lora_graph_patch_installed", False):
        raise RuntimeError("Load patch_lora_acl_graph before patch_lora_aot")
    lora_graph._compile_base_variant = _compile_base_variant
    TorchCompileWithNoGuardsWrapper.__init__ = _wrapper_init
    TorchCompileWithNoGuardsWrapper.aot_compile = _aot_compile
    decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn
    TorchCompileWithNoGuardsWrapper._ascend_lora_aot_patch_installed = True
    logger.info_once("Installed Qwen3.5/Magistral LoRA/base/base_one AOT artifact patch")


_install()
