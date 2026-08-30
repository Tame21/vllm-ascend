# SPDX-License-Identifier: Apache-2.0

"""Base/LoRA callable isolation for the vLLM compile wrapper."""

import inspect
import os
from contextlib import nullcontext
from functools import wraps
from types import FunctionType, MethodType

import torch
import vllm.envs as vllm_envs
from vllm.compilation import backends, monitor
from vllm.compilation import wrapper as wrapper_module
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger

from netrsn_turbo.turbo.version_0251.vllm_ascend.lora.punica_npu import (
    specialize_lora,
    validate_config,
)

logger = init_logger(__name__)

AOT_LORA = "lora"
AOT_BASE = "base"
AOT_BASE_ONE = "base_one"
AOT_VARIANTS = (AOT_LORA, AOT_BASE, AOT_BASE_ONE)


def clone_forward(self, suffix):
    original = self.forward.__func__
    changes = {"co_name": f"{original.__code__.co_name}_{suffix}"}
    if hasattr(original.__code__, "co_qualname"):
        changes["co_qualname"] = f"{original.__code__.co_qualname}_{suffix}"
    code = original.__code__.replace(**changes)
    cloned = FunctionType(
        code,
        original.__globals__,
        code.co_name,
        original.__defaults__,
        original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    cloned.__annotations__ = original.__annotations__
    return MethodType(cloned, self)


def compile_base_variant(self, prefix, suffix):
    config = self.vllm_config
    backend = config.compilation_config.init_backend(
        config,
        prefix=prefix,
        is_encoder=False,
    )
    options = (
        dict(config.compilation_config.inductor_compile_config)
        if backend == "inductor"
        else {}
    )
    if hasattr(torch.compiler, "skip_all_guards_unsafe"):
        options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
    else:
        options["guard_filter_fn"] = lambda entries: [False for _ in entries]
    aot_context = nullcontext()
    if vllm_envs.VLLM_USE_AOT_COMPILE and hasattr(
        torch._dynamo.config,
        "enable_aot_compile",
    ):
        aot_context = torch._dynamo.config.patch(enable_aot_compile=True)
    with aot_context:
        return torch.compile(
            clone_forward(self, suffix),
            fullgraph=True,
            dynamic=False,
            backend=backend,
            options=options,
        )


def select_variant(self):
    if self._ascend_has_lora():
        return AOT_LORA
    descriptor = (
        get_forward_context().batch_descriptor
        if is_forward_context_available()
        else None
    )
    return (
        AOT_BASE_ONE
        if descriptor is not None and descriptor.num_tokens == 1
        else AOT_BASE
    )


def variant_callable(self, variant):
    if variant == AOT_LORA:
        return self._compiled_callable
    if variant == AOT_BASE:
        return self._ascend_base_callable
    if variant == AOT_BASE_ONE:
        return self._ascend_base_one_callable
    raise ValueError(f"Unknown Qwen3.5 LoRA AOT variant: {variant}")


def variant_aot_path(aot_compilation_path, variant):
    return f"{aot_compilation_path}.{variant}"


class AOTVariantDispatcher:
    """Route calls to separately compiled base and LoRA AOT artifacts."""

    def __init__(self):
        self.artifacts = {}
        self.dirty_variants = set()

    def add_loaded(self, variant, artifact):
        self.artifacts[variant] = artifact

    def compile_variant(self, model, variant, args, kwargs):
        if variant != AOT_BASE_ONE:
            model._ascend_mark_variant_dynamic_inputs(variant, *args, **kwargs)
        compiled_callable = variant_callable(model, variant)
        if not hasattr(compiled_callable, "aot_compile"):
            raise RuntimeError(
                "AOT compile is unavailable for the Qwen3.5 LoRA "
                f"{variant} callable"
            )
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
        for variant in AOT_VARIANTS:
            if variant not in self.dirty_variants:
                continue
            path = variant_aot_path(aot_compilation_path, variant)
            tmp_file = f"{path}.{os.getpid()}.tmp"
            try:
                self.artifacts[variant].save_compiled_function(tmp_file)
                os.replace(tmp_file, path)
                self.dirty_variants.remove(variant)
                compilation_counter.num_aot_artifacts_saved += 1
                logger.info(
                    "Saved Qwen3.5 LoRA %s AOT artifact to %s",
                    variant,
                    path,
                )
            except Exception as error:
                logger.warning(
                    "Unable to save Qwen3.5 LoRA %s AOT artifact to %s: %s",
                    variant,
                    path,
                    error,
                )
                try:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except OSError:
                    logger.warning(
                        "Unable to remove temporary AOT artifact %s",
                        tmp_file,
                    )

    def __call__(self, model, *args, **kwargs):
        variant = select_variant(model)
        if variant not in self.artifacts:
            with monitor.monitor_torch_compile(
                model.vllm_config,
                is_encoder=model._is_encoder,
            ):
                self.compile_variant(model, variant, args, kwargs)
                compilation_counter.num_aot_compiles += 1
            self.save_dirty(model)
        return self.artifacts[variant](model, *args, **kwargs)


def wrap_aot_compile(original):
    @wraps(original)
    def aot_compile(self, *args, **kwargs):
        if not getattr(self, "_ascend_specialize_lora", False):
            return original(self, *args, **kwargs)
        dispatcher = getattr(self, "_ascend_preloaded_aot_dispatcher", None)
        if dispatcher is None:
            dispatcher = AOTVariantDispatcher()
        else:
            del self._ascend_preloaded_aot_dispatcher
        variant = select_variant(self)
        if variant not in dispatcher.artifacts:
            dispatcher.compile_variant(self, variant, args, kwargs)
        return dispatcher

    return aot_compile


def save_aot_compiled_function(self):
    dispatcher = self.aot_compiled_fn
    if not isinstance(dispatcher, AOTVariantDispatcher):
        raise RuntimeError("Expected the Qwen3.5 LoRA AOT variant dispatcher")
    dispatcher.save_dirty(self)


def wrap_try_load_aot_compiled_fn(original):
    @wraps(original)
    def try_load_aot_compiled_fn(model, aot_compilation_path):
        if not getattr(model, "_ascend_specialize_lora", False):
            return original(model, aot_compilation_path)
        dispatcher = AOTVariantDispatcher()
        for variant in AOT_VARIANTS:
            artifact = original(
                model,
                variant_aot_path(aot_compilation_path, variant),
            )
            if artifact is not None:
                dispatcher.add_loaded(variant, artifact)
        if not dispatcher.artifacts:
            return None
        model._aot_compilation_path = aot_compilation_path
        model._aot_cache_dir = os.path.dirname(aot_compilation_path)
        selected_variant = select_variant(model)
        if selected_variant not in dispatcher.artifacts:
            model._ascend_preloaded_aot_dispatcher = dispatcher
            return None
        return dispatcher

    return try_load_aot_compiled_fn


def lora_compile_enabled(config, is_encoder):
    return (
        specialize_lora(config)
        and backends.model_tag == "backbone"
        and not is_encoder
    )


def validate_lora_compile_config(config):
    validate_config(config)
    if config.compilation_config.mode != CompilationMode.VLLM_COMPILE:
        raise ValueError(
            "Qwen3.5 LoRA graph isolation requires CompilationMode.VLLM_COMPILE"
        )
    if config.compilation_config.dynamic_shapes_config.evaluate_guards:
        raise ValueError(
            "Qwen3.5 LoRA graph isolation requires "
            "dynamic_shapes_config.evaluate_guards=false"
        )


def call_original_init(original, self, compile_prefix, is_encoder, enabled):
    if not enabled:
        original(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
        return
    previous_bytecode_hook = vllm_envs.VLLM_USE_BYTECODE_HOOK
    vllm_envs.VLLM_USE_BYTECODE_HOOK = False
    try:
        original(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
    finally:
        vllm_envs.VLLM_USE_BYTECODE_HOOK = previous_bytecode_hook


def initialize_lora_variants(self, compile_prefix):
    self._ascend_punica_wrappers = None
    self._ascend_base_dynamic_inputs_marked = False
    self._ascend_aot_dynamic_variants_marked = set()
    self._ascend_base_callable = compile_base_variant(
        self,
        f"{compile_prefix}.base",
        "ascend_base",
    )
    self._ascend_base_one_callable = compile_base_variant(
        self,
        f"{compile_prefix}.base_one",
        "ascend_base_one",
    )
    if vllm_envs.VLLM_USE_AOT_COMPILE:
        self.save_aot_compiled_function = MethodType(
            save_aot_compiled_function,
            self,
        )


def wrap_init(original):
    @wraps(original)
    def init(self, compile_prefix="", is_encoder=False):
        config = get_current_vllm_config()
        enabled = lora_compile_enabled(config, is_encoder)
        if enabled:
            validate_lora_compile_config(config)
        call_original_init(original, self, compile_prefix, is_encoder, enabled)
        self._ascend_specialize_lora = enabled
        if enabled:
            initialize_lora_variants(self, compile_prefix)

    return init


def has_lora(self):
    if self._ascend_punica_wrappers is None:
        self._ascend_punica_wrappers = tuple(
            {
                id(wrapper): wrapper
                for module in self.modules()
                if (wrapper := getattr(module, "punica_wrapper", None)) is not None
            }.values()
        )
    if self._ascend_punica_wrappers:
        return any(not wrapper.no_lora for wrapper in self._ascend_punica_wrappers)
    return True


def mark_variant_dynamic_inputs(self, variant, *args, **kwargs):
    if variant in self._ascend_aot_dynamic_variants_marked:
        return
    dynamic_dims = getattr(self, "_dynamic_arg_dims", {})
    if not dynamic_dims:
        self._ascend_aot_dynamic_variants_marked.add(variant)
        return
    bound = inspect.signature(self.__class__.forward).bind(
        self,
        *args,
        **kwargs,
    )
    bound.apply_defaults()
    for name, dims in dynamic_dims.items():
        arg = bound.arguments.get(name)
        dims = [dims] if isinstance(dims, int) else list(dims)
        tensors = [arg] if isinstance(arg, torch.Tensor) else []
        if hasattr(arg, "tensors"):
            tensors.extend(arg.tensors.values())
        for tensor in tensors:
            torch._dynamo.mark_dynamic(
                tensor,
                [dim + tensor.ndim if dim < 0 else dim for dim in dims],
            )
    self._ascend_aot_dynamic_variants_marked.add(variant)


def wrap_call(original):
    @wraps(original)
    def call(self, *args, **kwargs):
        if not getattr(self, "_ascend_specialize_lora", False):
            return original(self, *args, **kwargs)
        variant = select_variant(self)
        callable_fn = variant_callable(self, variant)
        if variant == AOT_BASE and not self._ascend_base_dynamic_inputs_marked:
            self._ascend_mark_variant_dynamic_inputs(
                variant,
                *args,
                **kwargs,
            )
            self._ascend_base_dynamic_inputs_marked = True
        context = (
            nullcontext()
            if self.first_compile or not self.evaluate_guards
            else torch.compiler.set_stance("fail_on_recompile")
        )
        self.first_compile = False
        with wrapper_module._compilation_context(), context:
            return self._call_with_optional_nvtx_range(
                callable_fn,
                *args,
                **kwargs,
            )

    return call
