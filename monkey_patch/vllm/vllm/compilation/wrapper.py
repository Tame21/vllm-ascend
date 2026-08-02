"""Compile-wrapper replacements for independent base and LoRA callables."""

import inspect
from contextlib import contextmanager, nullcontext
from functools import wraps
from types import FunctionType, MethodType
from typing import Any

import torch
import vllm.envs as vllm_envs
from vllm.compilation import backends as backends_module
from vllm.compilation import wrapper as wrapper_module
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.config.compilation import DynamicShapesType
from vllm.forward_context import get_forward_context, is_forward_context_available


ORIGINAL_INIT = TorchCompileWithNoGuardsWrapper.__init__
ORIGINAL_CALL = TorchCompileWithNoGuardsWrapper.__call__


@contextmanager
def _without_bytecode_hook():
    had_override = "VLLM_USE_BYTECODE_HOOK" in vars(vllm_envs)
    previous_override = vars(vllm_envs).get("VLLM_USE_BYTECODE_HOOK")
    vllm_envs.VLLM_USE_BYTECODE_HOOK = False
    try:
        yield
    finally:
        if had_override:
            vllm_envs.VLLM_USE_BYTECODE_HOOK = previous_override
        else:
            delattr(vllm_envs, "VLLM_USE_BYTECODE_HOOK")


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


def _variant_compile_options(
    compilation_config: Any,
    backend: Any,
    evaluate_guards: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if isinstance(backend, str) and backend == "inductor":
        options = dict(compilation_config.inductor_compile_config)
    if compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
        return options

    if evaluate_guards:
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


def _init_impl(
    self: Any,
    original_init,
    compile_prefix: str = "",
    is_encoder: bool = False,
) -> None:
    vllm_config = get_current_vllm_config()
    specialize_lora = bool(
        vllm_config.lora_config is not None
        and vllm_config.compilation_config.cudagraph_specialize_lora
        and backends_module.model_tag == "backbone"
    )
    use_bytecode_hook = vllm_envs.VLLM_USE_BYTECODE_HOOK and not specialize_lora
    if specialize_lora and vllm_envs.VLLM_USE_AOT_COMPILE:
        raise RuntimeError(
            "Ascend base/LoRA ACL-graph specialization requires "
            "VLLM_USE_AOT_COMPILE=0."
        )

    if specialize_lora:
        with _without_bytecode_hook():
            original_init(
                self,
                compile_prefix=compile_prefix,
                is_encoder=is_encoder,
            )
    else:
        original_init(
            self,
            compile_prefix=compile_prefix,
            is_encoder=is_encoder,
        )

    self._external_specialize_lora = specialize_lora
    self._external_use_bytecode_hook = use_bytecode_hook
    self._external_base_dynamic_inputs_marked = False
    self._external_punica_wrappers = None
    if not specialize_lora:
        return

    compilation_config = vllm_config.compilation_config
    base_prefix = f"{compile_prefix}.base" if compile_prefix else "base"
    base_one_prefix = (
        f"{compile_prefix}.base_one" if compile_prefix else "base_one"
    )
    base_backend = compilation_config.init_backend(
        vllm_config,
        prefix=base_prefix,
        is_encoder=is_encoder,
    )
    base_one_backend = compilation_config.init_backend(
        vllm_config,
        prefix=base_one_prefix,
        is_encoder=is_encoder,
    )
    self._external_base_compiled_callable = torch.compile(
        _clone_forward(self, "base"),
        fullgraph=True,
        dynamic=False,
        backend=base_backend,
        options=_variant_compile_options(
            compilation_config,
            base_backend,
            self.evaluate_guards,
        ),
    )
    self._external_base_one_compiled_callable = torch.compile(
        _clone_forward(self, "base_one"),
        fullgraph=True,
        dynamic=False,
        backend=base_one_backend,
        options=_variant_compile_options(
            compilation_config,
            base_one_backend,
            self.evaluate_guards,
        ),
    )


def init_wrapper(original_init):
    """Wrap TorchCompileWithNoGuardsWrapper.__init__."""

    @wraps(original_init)
    def wrapped_init(
        self: Any,
        compile_prefix: str = "",
        is_encoder: bool = False,
    ) -> None:
        _init_impl(
            self,
            original_init,
            compile_prefix=compile_prefix,
            is_encoder=is_encoder,
        )

    return wrapped_init


# These methods do not exist in v0.23 and are attached by direct assignment.
def punica_has_lora(self: Any) -> bool | None:
    if self._external_punica_wrappers is None:
        wrappers: list[Any] = []
        seen: set[int] = set()
        for module in self.modules():
            punica_wrapper = getattr(module, "punica_wrapper", None)
            if punica_wrapper is not None and id(punica_wrapper) not in seen:
                wrappers.append(punica_wrapper)
                seen.add(id(punica_wrapper))
        self._external_punica_wrappers = wrappers
    if not self._external_punica_wrappers:
        return None
    return any(
        not wrapper.no_lora for wrapper in self._external_punica_wrappers
    )


def mark_variant_dynamic_inputs(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    dynamic_arg_dims = getattr(self, "_dynamic_arg_dims", {})
    if not dynamic_arg_dims:
        return
    signature = inspect.signature(self.__class__.forward)
    bound_args = signature.bind(self, *args, **kwargs)
    bound_args.apply_defaults()
    for name, dims_value in dynamic_arg_dims.items():
        arg = bound_args.arguments.get(name)
        if arg is None:
            continue
        if isinstance(dims_value, dict):
            dims = list(dims_value)
        elif isinstance(dims_value, int):
            dims = [dims_value]
        else:
            dims = list(dims_value)
        tensors = [arg] if isinstance(arg, torch.Tensor) else []
        if hasattr(arg, "tensors"):
            tensors.extend(arg.tensors.values())
        for tensor in tensors:
            normalized_dims = [
                tensor.ndim + dim if dim < 0 else dim for dim in dims
            ]
            torch._dynamo.mark_dynamic(tensor, normalized_dims)


def call(self: Any, *args: Any, **kwargs: Any) -> Any:
    compiled_callable = self._compiled_callable
    use_base_callable = False
    if self._external_specialize_lora and is_forward_context_available():
        batch_descriptor = get_forward_context().batch_descriptor
        has_lora = (
            batch_descriptor.has_lora
            if batch_descriptor is not None
            else True
        )
        punica_state = self._external_punica_has_lora()
        if punica_state is not None:
            has_lora = punica_state
        if not has_lora:
            use_base_callable = True
            if (
                batch_descriptor is not None
                and batch_descriptor.num_tokens == 1
            ):
                compiled_callable = self._external_base_one_compiled_callable
            else:
                compiled_callable = self._external_base_compiled_callable

    if (
        use_base_callable
        and compiled_callable is self._external_base_compiled_callable
        and not self._external_base_dynamic_inputs_marked
    ):
        self._external_mark_variant_dynamic_inputs(*args, **kwargs)
        self._external_base_dynamic_inputs_marked = True

    if self._external_use_bytecode_hook:
        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            return compiled_callable(*args, **kwargs)
        if not self._compiled_bytecode:
            torch._dynamo.eval_frame.remove_from_cache(
                self.original_code_object()
            )
            return self._call_with_optional_nvtx_range(
                compiled_callable,
                *args,
                **kwargs,
            )
        with self._dispatch_to_compiled_code():
            return self._call_with_optional_nvtx_range(
                self.forward,
                *args,
                **kwargs,
            )

    context = (
        nullcontext()
        if self.first_compile or not self.evaluate_guards
        else torch.compiler.set_stance("fail_on_recompile")
    )
    self.first_compile = False
    with wrapper_module._compilation_context(), context:
        return self._call_with_optional_nvtx_range(
            compiled_callable,
            *args,
            **kwargs,
        )
