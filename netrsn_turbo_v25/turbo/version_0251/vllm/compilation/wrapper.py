# SPDX-License-Identifier: Apache-2.0

"""Base/LoRA callable isolation for the vLLM compile wrapper."""

import inspect
from contextlib import nullcontext
from functools import wraps
from types import FunctionType, MethodType

import torch
import vllm.envs as vllm_envs
from vllm.compilation import backends
from vllm.compilation import wrapper as wrapper_module
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

from netrsn_turbo.turbo.version_0251.vllm_ascend.lora.punica_npu import (
    specialize_lora,
    validate_config,
)


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
    return torch.compile(
        clone_forward(self, suffix),
        fullgraph=True,
        dynamic=False,
        backend=backend,
        options=options,
    )


def wrap_init(original):
    @wraps(original)
    def init(self, compile_prefix="", is_encoder=False):
        config = get_current_vllm_config()
        enabled = (
            specialize_lora(config)
            and backends.model_tag == "backbone"
            and not is_encoder
        )
        if enabled:
            validate_config(config)
            if vllm_envs.VLLM_USE_AOT_COMPILE:
                raise ValueError(
                    "Qwen3.5 LoRA graph isolation does not support "
                    "VLLM_USE_AOT_COMPILE=1"
                )
            if config.compilation_config.mode != CompilationMode.VLLM_COMPILE:
                raise ValueError(
                    "Qwen3.5 LoRA graph isolation requires CompilationMode.VLLM_COMPILE"
                )
            if config.compilation_config.dynamic_shapes_config.evaluate_guards:
                raise ValueError(
                    "Qwen3.5 LoRA graph isolation requires "
                    "dynamic_shapes_config.evaluate_guards=false"
                )
        if enabled:
            previous_bytecode_hook = vllm_envs.VLLM_USE_BYTECODE_HOOK
            vllm_envs.VLLM_USE_BYTECODE_HOOK = False
            try:
                original(
                    self,
                    compile_prefix=compile_prefix,
                    is_encoder=is_encoder,
                )
            finally:
                vllm_envs.VLLM_USE_BYTECODE_HOOK = previous_bytecode_hook
        else:
            original(
                self,
                compile_prefix=compile_prefix,
                is_encoder=is_encoder,
            )
        self._ascend_specialize_lora = enabled
        if not enabled:
            return
        self._ascend_punica_wrappers = None
        self._ascend_base_dynamic_inputs_marked = False
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


def mark_base_dynamic_inputs(self, *args, **kwargs):
    dynamic_dims = getattr(self, "_dynamic_arg_dims", {})
    if not dynamic_dims:
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


def wrap_call(original):
    @wraps(original)
    def call(self, *args, **kwargs):
        if not getattr(self, "_ascend_specialize_lora", False):
            return original(self, *args, **kwargs)
        if self._ascend_has_lora():
            callable_fn = self._compiled_callable
        else:
            descriptor = (
                get_forward_context().batch_descriptor
                if is_forward_context_available()
                else None
            )
            if descriptor is not None and descriptor.num_tokens == 1:
                callable_fn = self._ascend_base_one_callable
            else:
                callable_fn = self._ascend_base_callable
                if not self._ascend_base_dynamic_inputs_marked:
                    self._ascend_mark_base_dynamic_inputs(*args, **kwargs)
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
