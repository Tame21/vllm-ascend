"""Split Base/LoRA compile callables without patching magic methods."""

import inspect
from types import FunctionType, MethodType
from typing import Any

import torch
import vllm.envs as vllm_envs
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode
from vllm.config.compilation import DynamicShapesType
from vllm.forward_context import get_forward_context, is_forward_context_available


ORIGINAL_CALL_WITH_OPTIONAL_NVTX_RANGE = (
    TorchCompileWithNoGuardsWrapper._call_with_optional_nvtx_range
)


def _clone_forward(self: Any, suffix: str) -> MethodType:
    forward_func = self.forward.__func__
    code_name = f"{forward_func.__code__.co_name}_{suffix}"
    cloned_func = FunctionType(
        forward_func.__code__.replace(
            co_name=code_name,
            co_qualname=f"{forward_func.__code__.co_qualname}_{suffix}",
        ),
        forward_func.__globals__,
        name=code_name,
        argdefs=forward_func.__defaults__,
        closure=forward_func.__closure__,
    )
    cloned_func.__kwdefaults__ = forward_func.__kwdefaults__
    cloned_func.__annotations__ = forward_func.__annotations__
    return MethodType(cloned_func, self)


def _variant_compile_options(self: Any, backend: Any) -> dict[str, Any]:
    compilation_config = self.vllm_config.compilation_config
    options: dict[str, Any] = {}
    if isinstance(backend, str) and backend == "inductor":
        options = dict(compilation_config.inductor_compile_config)
    if compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
        return options

    if self.evaluate_guards:
        if compilation_config.dynamic_shapes_config.type == DynamicShapesType.UNBACKED:
            raise AssertionError("UNBACKED dynamic shapes do not add guards")
        options["guard_filter_fn"] = lambda entries: [
            entry.guard_type == "SHAPE_ENV" for entry in entries
        ]
    elif hasattr(torch.compiler, "skip_all_guards_unsafe"):
        options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
    else:
        options["guard_filter_fn"] = lambda entries: [False for _ in entries]
    return options


def _prepare_lora_variants(self: Any) -> None:
    if getattr(self, "_external_lora_variants_ready", False):
        return

    vllm_config = self.vllm_config
    self._external_punica_wrappers = _get_punica_wrappers(self)
    specialize_lora = bool(
        vllm_config.lora_config is not None
        and vllm_config.compilation_config.cudagraph_specialize_lora
        and not self._is_encoder
        and self._external_punica_wrappers
    )
    self._external_specialize_lora = specialize_lora
    self._external_base_dynamic_inputs_marked = False

    if specialize_lora:
        if vllm_envs.VLLM_USE_AOT_COMPILE:
            raise RuntimeError(
                "Ascend Base/LoRA ACL-graph specialization requires "
                "VLLM_USE_AOT_COMPILE=0."
            )
        if vllm_envs.VLLM_USE_BYTECODE_HOOK:
            raise RuntimeError(
                "Ascend Base/LoRA ACL-graph specialization requires "
                "VLLM_USE_BYTECODE_HOOK=0 before model construction."
            )

        compilation_config = vllm_config.compilation_config
        compile_prefix = self._compile_prefix
        variants = (
            ("base", "_external_base_compiled_callable"),
            ("base_one", "_external_base_one_compiled_callable"),
        )
        for suffix, attribute_name in variants:
            prefix = f"{compile_prefix}.{suffix}" if compile_prefix else suffix
            backend = compilation_config.init_backend(
                vllm_config,
                prefix=prefix,
                is_encoder=self._is_encoder,
            )
            setattr(
                self,
                attribute_name,
                torch.compile(
                    _clone_forward(self, suffix),
                    fullgraph=True,
                    dynamic=False,
                    backend=backend,
                    options=_variant_compile_options(self, backend),
                ),
            )

    self._external_lora_variants_ready = True


def _get_punica_wrappers(self: Any) -> list[Any]:
    cached = getattr(self, "_external_punica_wrappers", None)
    if cached is not None:
        return cached
    wrappers: list[Any] = []
    seen: set[int] = set()
    for module in self.modules():
        wrapper = getattr(module, "punica_wrapper", None)
        if wrapper is not None and id(wrapper) not in seen:
            wrappers.append(wrapper)
            seen.add(id(wrapper))
    self._external_punica_wrappers = wrappers
    return wrappers


def _punica_has_lora(self: Any) -> bool | None:
    wrappers = _get_punica_wrappers(self)
    if not wrappers:
        return None
    return any(not wrapper.no_lora for wrapper in wrappers)


def _mark_variant_dynamic_inputs(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    dynamic_arg_dims = getattr(self, "_dynamic_arg_dims", {})
    if not dynamic_arg_dims:
        return
    bound_args = inspect.signature(self.__class__.forward).bind(
        self, *args, **kwargs
    )
    bound_args.apply_defaults()
    for name, dims_value in dynamic_arg_dims.items():
        arg = bound_args.arguments.get(name)
        if arg is None:
            continue
        dims = (
            [dims_value]
            if isinstance(dims_value, int)
            else list(dims_value)
        )
        tensors = [arg] if isinstance(arg, torch.Tensor) else []
        if hasattr(arg, "tensors"):
            tensors.extend(arg.tensors.values())
        for tensor in tensors:
            normalized_dims = [
                tensor.ndim + dim if dim < 0 else dim for dim in dims
            ]
            torch._dynamo.mark_dynamic(tensor, normalized_dims)


def call_with_optional_nvtx_range(
    self: Any,
    callable_fn,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Select the Base/LoRA callable, then preserve the original NVTX path."""
    _prepare_lora_variants(self)

    if self._external_specialize_lora and is_forward_context_available():
        batch_descriptor = get_forward_context().batch_descriptor
        has_lora = (
            batch_descriptor.has_lora
            if batch_descriptor is not None
            else True
        )
        punica_state = _punica_has_lora(self)
        if punica_state is not None:
            has_lora = punica_state

        if not has_lora:
            if batch_descriptor is not None and batch_descriptor.num_tokens == 1:
                callable_fn = self._external_base_one_compiled_callable
            else:
                callable_fn = self._external_base_compiled_callable
                if not self._external_base_dynamic_inputs_marked:
                    _mark_variant_dynamic_inputs(self, *args, **kwargs)
                    self._external_base_dynamic_inputs_marked = True

    return ORIGINAL_CALL_WITH_OPTIONAL_NVTX_RANGE(
        self,
        callable_fn,
        *args,
        **kwargs,
    )
