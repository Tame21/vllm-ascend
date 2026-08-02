"""External base/LoRA ACL-graph isolation patch for vLLM v0.23.

Call ``install(register_func)`` before model construction. Existing third-party
functions are replaced only through ``register_func(function)``. Methods that
do not exist in the third-party classes are copied here and attached with
explicit class attribute assignments.
"""

import inspect
import sys
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from types import FunctionType, MethodType
from typing import Any

import torch
import vllm.envs as vllm_envs
from vllm.compilation import backends as backends_module
from vllm.compilation import monitor as compilation_monitor
from vllm.compilation import wrapper as wrapper_module
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CUDAGraphMode, CompilationMode, get_current_vllm_config
from vllm.config.compilation import DynamicShapesType
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.compilation import acl_graph as acl_graph_module
from vllm_ascend.compilation.acl_graph import GraphParams

logger = init_logger(__name__)

RegisterFunc = Callable[[Callable[..., Any]], Any]
_PATCH_MARKER = "_external_lora_acl_graph_v23_patch"


_ORIGINAL_WRAPPER_INIT = TorchCompileWithNoGuardsWrapper.__init__
_ORIGINAL_WRAPPER_CALL = TorchCompileWithNoGuardsWrapper.__call__
_ORIGINAL_SET_GRAPH_PARAMS = acl_graph_module.set_graph_params
_ORIGINAL_SET_DRAFT_GRAPH_PARAMS = acl_graph_module.set_draft_graph_params
_ORIGINAL_SET_DRAFT_PREFILL_PARAMS = (
    acl_graph_module.set_draft_graph_prefill_params
)
_ORIGINAL_WEAK_REF_WORKSPACES = acl_graph_module.weak_ref_workspaces
_ORIGINAL_CAPTURE_MODEL = GPUModelRunner.capture_model
_ORIGINAL_WARMUP_AND_CAPTURE = GPUModelRunner._warmup_and_capture
_ORIGINAL_DUMMY_LORA_CONTEXT = GPUModelRunner.maybe_dummy_run_with_lora


def _register_replacement(
    register_func: RegisterFunc,
    replacement: Callable[..., Any],
    original: Callable[..., Any],
) -> None:
    """Give a replacement the target identity, then register it."""
    replacement.__name__ = original.__name__
    replacement.__qualname__ = original.__qualname__
    replacement.__module__ = original.__module__
    register_func(replacement)


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


class _GraphParamStore(dict):
    """Resolve integer token keys to the active full-graph descriptor."""

    def __init__(self, capture_sizes: list[int], default_factory):
        super().__init__((size, default_factory()) for size in capture_sizes)
        self.default_factory = default_factory

    @staticmethod
    def _resolve_key(key):
        if not isinstance(key, int):
            return key
        try:
            forward_context = get_forward_context()
        except (AssertionError, LookupError, RuntimeError):
            return key
        batch_descriptor = forward_context.batch_descriptor
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and batch_descriptor is not None
            and batch_descriptor.num_tokens == key
        ):
            return batch_descriptor
        return key

    @staticmethod
    def _raise_if_uncaptured(resolved_key) -> None:
        if (
            not isinstance(resolved_key, int)
            and not compilation_monitor.cudagraph_capturing_enabled
        ):
            raise RuntimeError(
                "ACL graph parameters were not captured for runtime batch "
                f"descriptor {resolved_key!r}. Check base/LoRA graph capture "
                "descriptor consistency."
            )

    def __contains__(self, key):
        return dict.__contains__(self, self._resolve_key(key))

    def __getitem__(self, key):
        resolved_key = self._resolve_key(key)
        if not dict.__contains__(self, resolved_key):
            self._raise_if_uncaptured(resolved_key)
            dict.__setitem__(self, resolved_key, self.default_factory())
        return dict.__getitem__(self, resolved_key)

    def __setitem__(self, key, value):
        resolved_key = self._resolve_key(key)
        if not dict.__contains__(self, resolved_key):
            self._raise_if_uncaptured(resolved_key)
        dict.__setitem__(self, resolved_key, value)

    def get(self, key, default=None):
        resolved_key = self._resolve_key(key)
        if not dict.__contains__(self, resolved_key):
            self._raise_if_uncaptured(resolved_key)
        return dict.get(self, resolved_key, default)


def _make_graph_params(capture_sizes: list[int]) -> GraphParams:
    return GraphParams(
        _GraphParamStore(capture_sizes, list),
        _GraphParamStore(capture_sizes, lambda: None),
        _GraphParamStore(capture_sizes, list),
        _GraphParamStore(capture_sizes, list),
    )


def set_graph_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    acl_graph_module._graph_params = _make_graph_params(capture_sizes)


def set_draft_graph_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    acl_graph_module._draft_graph_params = _make_graph_params(capture_sizes)


def set_draft_graph_prefill_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph prefill parameters have already been set!")
    acl_graph_module._draft_graph_prefill_params = _make_graph_params(
        capture_sizes
    )


def weak_ref_workspaces(params) -> None:
    if params is None:
        return
    for graph_key, workspace in list(dict.items(params.workspaces)):
        if workspace is None:
            continue
        dict.__setitem__(
            params.workspaces,
            graph_key,
            acl_graph_module.weak_ref_tensors(workspace),
        )


def _wrapper_init(
    self: Any,
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
            _ORIGINAL_WRAPPER_INIT(
                self,
                compile_prefix=compile_prefix,
                is_encoder=is_encoder,
            )
    else:
        _ORIGINAL_WRAPPER_INIT(
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


# New wrapper methods: attached by explicit assignment in install().
def _external_punica_has_lora(self: Any) -> bool | None:
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


def _external_mark_variant_dynamic_inputs(
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


def _wrapper_call(self: Any, *args: Any, **kwargs: Any) -> Any:
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


@contextmanager
def maybe_dummy_run_with_lora(
    self,
    lora_config,
    num_scheduled_tokens,
    num_sampled_tokens,
    remove_lora=True,
    num_active_loras=0,
    mapping_type=None,
):
    forced_count = getattr(self, "_external_forced_dummy_lora_count", None)
    if forced_count is not None:
        num_active_loras = forced_count
    kwargs = {
        "remove_lora": remove_lora,
        "num_active_loras": num_active_loras,
    }
    if mapping_type is not None:
        kwargs["mapping_type"] = mapping_type
    with _ORIGINAL_DUMMY_LORA_CONTEXT(
        self,
        lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        **kwargs,
    ):
        if forced_count == 0:
            _set_punica_no_lora(self, True)
        yield


def _warmup_and_capture(
    self,
    desc,
    cudagraph_runtime_mode,
    *args,
    **kwargs,
):
    previous = getattr(self, "_external_forced_dummy_lora_count", None)
    self._external_forced_dummy_lora_count = desc.num_active_loras
    try:
        return _ORIGINAL_WARMUP_AND_CAPTURE(
            self,
            desc,
            cudagraph_runtime_mode,
            *args,
            **kwargs,
        )
    finally:
        self._external_forced_dummy_lora_count = previous


def capture_model(self) -> int:
    if (
        self.lora_config is not None
        and self.compilation_config.cudagraph_specialize_lora
    ):
        previous = getattr(self, "_external_forced_dummy_lora_count", None)
        self._external_forced_dummy_lora_count = 0
        try:
            _set_punica_no_lora(self, True)
            self._dummy_run(
                num_tokens=self.max_num_tokens,
                is_profile=True,
                num_active_loras=0,
            )
            _set_punica_no_lora(self, True)
            self._dummy_run(
                num_tokens=1,
                is_profile=True,
                num_active_loras=0,
            )
            torch.npu.synchronize()
        finally:
            self._external_forced_dummy_lora_count = previous
    return _ORIGINAL_CAPTURE_MODEL(self)


def _set_punica_no_lora(model_runner: Any, no_lora: bool) -> None:
    seen: set[int] = set()
    for module in model_runner.model.modules():
        punica_wrapper = getattr(module, "punica_wrapper", None)
        if punica_wrapper is None or id(punica_wrapper) in seen:
            continue
        punica_wrapper.no_lora = no_lora
        seen.add(id(punica_wrapper))


def _refresh_imported_graph_param_aliases() -> None:
    """Refresh aliases imported before this external patch is installed."""
    model_runner_module = sys.modules.get(
        "vllm_ascend.worker.model_runner_v1"
    )
    if model_runner_module is None:
        return
    model_runner_module.set_graph_params = set_graph_params
    model_runner_module.set_draft_graph_params = set_draft_graph_params


def install(register_func: RegisterFunc) -> None:
    """Install all replacements and copied methods."""
    if getattr(TorchCompileWithNoGuardsWrapper, _PATCH_MARKER, False):
        return

    # New methods use explicit assignment, not register_func.
    TorchCompileWithNoGuardsWrapper._external_punica_has_lora = (
        _external_punica_has_lora
    )
    TorchCompileWithNoGuardsWrapper._external_mark_variant_dynamic_inputs = (
        _external_mark_variant_dynamic_inputs
    )

    # Existing functions are replaced exclusively through register_func.
    replacements = (
        (_wrapper_init, _ORIGINAL_WRAPPER_INIT),
        (_wrapper_call, _ORIGINAL_WRAPPER_CALL),
        (set_graph_params, _ORIGINAL_SET_GRAPH_PARAMS),
        (set_draft_graph_params, _ORIGINAL_SET_DRAFT_GRAPH_PARAMS),
        (
            set_draft_graph_prefill_params,
            _ORIGINAL_SET_DRAFT_PREFILL_PARAMS,
        ),
        (weak_ref_workspaces, _ORIGINAL_WEAK_REF_WORKSPACES),
        (maybe_dummy_run_with_lora, _ORIGINAL_DUMMY_LORA_CONTEXT),
        (_warmup_and_capture, _ORIGINAL_WARMUP_AND_CAPTURE),
        (capture_model, _ORIGINAL_CAPTURE_MODEL),
    )
    for replacement, original in replacements:
        _register_replacement(register_func, replacement, original)

    _refresh_imported_graph_param_aliases()
    setattr(TorchCompileWithNoGuardsWrapper, _PATCH_MARKER, True)
    logger.info_once("Installed external v0.23 base/LoRA ACL-graph patch")
