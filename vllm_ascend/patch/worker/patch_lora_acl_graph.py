# SPDX-License-Identifier: Apache-2.0
"""Base/LoRA compilation and ACL graph isolation for Qwen3.5 on v0.25.1."""

import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from types import FunctionType, MethodType

import torch
import vllm.envs as vllm_envs
from vllm.compilation import backends, decorators, monitor
from vllm.compilation import wrapper as wrapper_module
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode, CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.compilation import acl_graph
from vllm_ascend.patch.worker.patch_qwen3_5_dense_lora import specialize_lora, validate_config

_ORIGINAL_GRAPH_PARAMS = acl_graph.GraphParams
_ORIGINAL_WEAK_REF_WORKSPACES = acl_graph.weak_ref_workspaces
_ORIGINAL_WRAPPER_INIT = TorchCompileWithNoGuardsWrapper.__init__
_ORIGINAL_WRAPPER_CALL = TorchCompileWithNoGuardsWrapper.__call__
_ORIGINAL_WRAPPER_AOT_COMPILE = TorchCompileWithNoGuardsWrapper.aot_compile
_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN = decorators._try_load_aot_compiled_fn
_ORIGINAL_DUMMY_LORA_CONTEXT = GPUModelRunner.maybe_dummy_run_with_lora
_ORIGINAL_WARMUP_AND_CAPTURE = GPUModelRunner._warmup_and_capture
_ORIGINAL_CAPTURE_MODEL = GPUModelRunner.capture_model

logger = init_logger(__name__)

_AOT_LORA = "lora"
_AOT_BASE = "base"
_AOT_BASE_ONE = "base_one"
_AOT_VARIANTS = (_AOT_LORA, _AOT_BASE, _AOT_BASE_ONE)


class _GraphParamStore(dict):
    """Resolve token-only accesses to the currently active FULL graph key."""

    def __init__(self, values, default_factory):
        super().__init__(values)
        self.default_factory = default_factory

    @staticmethod
    def _resolve_key(key):
        if not isinstance(key, int) or not is_forward_context_available():
            return key
        context = get_forward_context()
        descriptor = context.batch_descriptor
        if (
            context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and descriptor is not None
            and descriptor.num_tokens == key
        ):
            return descriptor
        return key

    @staticmethod
    def _check_missing(key):
        if not isinstance(key, int) and not monitor.cudagraph_capturing_enabled:
            raise RuntimeError(f"ACL graph parameters were not captured for {key!r}")

    def __contains__(self, key):
        return dict.__contains__(self, self._resolve_key(key))

    def __getitem__(self, key):
        key = self._resolve_key(key)
        if not dict.__contains__(self, key):
            self._check_missing(key)
            dict.__setitem__(self, key, self.default_factory())
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        key = self._resolve_key(key)
        if not dict.__contains__(self, key):
            self._check_missing(key)
        dict.__setitem__(self, key, value)

    def get(self, key, default=None):
        key = self._resolve_key(key)
        if not dict.__contains__(self, key):
            self._check_missing(key)
        return dict.get(self, key, default)


class _LoRAGraphParams(_ORIGINAL_GRAPH_PARAMS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            config = get_current_vllm_config()
        except AssertionError:
            return
        if specialize_lora(config):
            self.events = _GraphParamStore(self.events, list)
            self.workspaces = _GraphParamStore(self.workspaces, lambda: None)
            self.handles = _GraphParamStore(self.handles, list)
            self.attn_params = _GraphParamStore(self.attn_params, list)


def weak_ref_workspaces(params):
    if params is None or not isinstance(params.workspaces, _GraphParamStore):
        return _ORIGINAL_WEAK_REF_WORKSPACES(params)
    # Iterate physical keys: an integer key must not resolve to the active
    # descriptor while we weaken workspaces belonging to all captured graphs.
    for key, workspace in list(dict.items(params.workspaces)):
        if workspace is not None:
            dict.__setitem__(params.workspaces, key, acl_graph.weak_ref_tensors(workspace))


def _clone_forward(self, suffix):
    original = self.forward.__func__
    changes = {"co_name": f"{original.__code__.co_name}_{suffix}"}
    if hasattr(original.__code__, "co_qualname"):
        changes["co_qualname"] = f"{original.__code__.co_qualname}_{suffix}"
    code = original.__code__.replace(**changes)
    cloned = FunctionType(code, original.__globals__, code.co_name, original.__defaults__, original.__closure__)
    cloned.__kwdefaults__ = original.__kwdefaults__
    cloned.__annotations__ = original.__annotations__
    return MethodType(cloned, self)


def _compile_base_variant(self, prefix, suffix):
    config = self.vllm_config
    backend = config.compilation_config.init_backend(config, prefix=prefix, is_encoder=False)
    options = dict(config.compilation_config.inductor_compile_config) if backend == "inductor" else {}
    if hasattr(torch.compiler, "skip_all_guards_unsafe"):
        options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
    else:
        options["guard_filter_fn"] = lambda entries: [False for _ in entries]
    aot_context = nullcontext()
    if vllm_envs.VLLM_USE_AOT_COMPILE and hasattr(torch._dynamo.config, "enable_aot_compile"):
        aot_context = torch._dynamo.config.patch(enable_aot_compile=True)
    with aot_context:
        return torch.compile(
            _clone_forward(self, suffix),
            fullgraph=True,
            dynamic=False,
            backend=backend,
            options=options,
        )


def _select_variant(self):
    if self._ascend_has_lora():
        return _AOT_LORA
    descriptor = get_forward_context().batch_descriptor if is_forward_context_available() else None
    return _AOT_BASE_ONE if descriptor is not None and descriptor.num_tokens == 1 else _AOT_BASE


def _variant_callable(self, variant):
    if variant == _AOT_LORA:
        return self._compiled_callable
    if variant == _AOT_BASE:
        return self._ascend_base_callable
    if variant == _AOT_BASE_ONE:
        return self._ascend_base_one_callable
    raise ValueError(f"Unknown Qwen3.5 LoRA AOT variant: {variant}")


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
        if variant != _AOT_BASE_ONE:
            model._ascend_mark_variant_dynamic_inputs(variant, *args, **kwargs)
        compiled_callable = _variant_callable(model, variant)
        if not hasattr(compiled_callable, "aot_compile"):
            raise RuntimeError(f"AOT compile is unavailable for the Qwen3.5 LoRA {variant} callable")
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
                logger.info("Saved Qwen3.5 LoRA %s AOT artifact to %s", variant, path)
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
                    logger.warning("Unable to remove temporary AOT artifact %s", tmp_file)

    def __call__(self, model, *args, **kwargs):
        variant = _select_variant(model)
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
    variant = _select_variant(self)
    if variant not in dispatcher.artifacts:
        dispatcher.compile_variant(self, variant, args, kwargs)
    return dispatcher


def _save_aot_compiled_function(self):
    dispatcher = self.aot_compiled_fn
    if not isinstance(dispatcher, _AOTVariantDispatcher):
        raise RuntimeError("Expected the Qwen3.5 LoRA AOT variant dispatcher")
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
    selected_variant = _select_variant(model)
    if selected_variant not in dispatcher.artifacts:
        model._ascend_preloaded_aot_dispatcher = dispatcher
        return None
    return dispatcher


@wraps(_ORIGINAL_WRAPPER_INIT)
def _wrapper_init(self, compile_prefix="", is_encoder=False):
    config = get_current_vllm_config()
    enabled = specialize_lora(config) and backends.model_tag == "backbone" and not is_encoder
    if enabled:
        validate_config(config)
        if config.compilation_config.mode != CompilationMode.VLLM_COMPILE:
            raise ValueError("Qwen3.5 LoRA graph isolation requires CompilationMode.VLLM_COMPILE")
        if config.compilation_config.dynamic_shapes_config.evaluate_guards:
            raise ValueError("Qwen3.5 LoRA graph isolation requires dynamic_shapes_config.evaluate_guards=false")
    if enabled:
        previous_bytecode_hook = vllm_envs.VLLM_USE_BYTECODE_HOOK
        vllm_envs.VLLM_USE_BYTECODE_HOOK = False
        try:
            _ORIGINAL_WRAPPER_INIT(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
        finally:
            vllm_envs.VLLM_USE_BYTECODE_HOOK = previous_bytecode_hook
    else:
        _ORIGINAL_WRAPPER_INIT(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
    self._ascend_specialize_lora = enabled
    if not enabled:
        return
    self._ascend_punica_wrappers = None
    self._ascend_base_dynamic_inputs_marked = False
    self._ascend_aot_dynamic_variants_marked = set()
    self._ascend_base_callable = _compile_base_variant(self, f"{compile_prefix}.base", "ascend_base")
    self._ascend_base_one_callable = _compile_base_variant(self, f"{compile_prefix}.base_one", "ascend_base_one")
    if vllm_envs.VLLM_USE_AOT_COMPILE:
        self.save_aot_compiled_function = MethodType(_save_aot_compiled_function, self)


def _has_lora(self):
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


def _mark_variant_dynamic_inputs(self, variant, *args, **kwargs):
    if variant in self._ascend_aot_dynamic_variants_marked:
        return
    dynamic_dims = getattr(self, "_dynamic_arg_dims", {})
    if not dynamic_dims:
        self._ascend_aot_dynamic_variants_marked.add(variant)
        return
    bound = inspect.signature(self.__class__.forward).bind(self, *args, **kwargs)
    bound.apply_defaults()
    for name, dims in dynamic_dims.items():
        arg = bound.arguments.get(name)
        dims = [dims] if isinstance(dims, int) else list(dims)
        tensors = [arg] if isinstance(arg, torch.Tensor) else []
        if hasattr(arg, "tensors"):
            tensors.extend(arg.tensors.values())
        for tensor in tensors:
            torch._dynamo.mark_dynamic(tensor, [dim + tensor.ndim if dim < 0 else dim for dim in dims])
    self._ascend_aot_dynamic_variants_marked.add(variant)


@wraps(_ORIGINAL_WRAPPER_CALL)
def _wrapper_call(self, *args, **kwargs):
    if not getattr(self, "_ascend_specialize_lora", False):
        return _ORIGINAL_WRAPPER_CALL(self, *args, **kwargs)
    variant = _select_variant(self)
    callable_fn = _variant_callable(self, variant)
    if variant == _AOT_BASE and not self._ascend_base_dynamic_inputs_marked:
        self._ascend_mark_variant_dynamic_inputs(variant, *args, **kwargs)
        self._ascend_base_dynamic_inputs_marked = True
    context = (
        nullcontext()
        if self.first_compile or not self.evaluate_guards
        else torch.compiler.set_stance("fail_on_recompile")
    )
    self.first_compile = False
    with wrapper_module._compilation_context(), context:
        return self._call_with_optional_nvtx_range(callable_fn, *args, **kwargs)


@contextmanager
def _force_lora_count(self, count):
    previous = getattr(self, "_ascend_forced_dummy_lora_count", None)
    self._ascend_forced_dummy_lora_count = count
    try:
        yield
    finally:
        self._ascend_forced_dummy_lora_count = previous


@contextmanager
def maybe_dummy_run_with_lora(
    self, lora_config, num_scheduled_tokens, num_sampled_tokens, remove_lora=True, num_active_loras=0, mapping_type=None
):
    count = getattr(self, "_ascend_forced_dummy_lora_count", None)
    if count is not None:
        num_active_loras = count
    kwargs = {"remove_lora": remove_lora, "num_active_loras": num_active_loras}
    if mapping_type is not None:
        kwargs["mapping_type"] = mapping_type
    with _ORIGINAL_DUMMY_LORA_CONTEXT(self, lora_config, num_scheduled_tokens, num_sampled_tokens, **kwargs):
        yield


@wraps(_ORIGINAL_WARMUP_AND_CAPTURE)
def _warmup_and_capture(self, desc, cudagraph_runtime_mode, *args, **kwargs):
    context = _force_lora_count(self, desc.num_active_loras) if specialize_lora(self.vllm_config) else nullcontext()
    with context:
        return _ORIGINAL_WARMUP_AND_CAPTURE(self, desc, cudagraph_runtime_mode, *args, **kwargs)


@wraps(_ORIGINAL_CAPTURE_MODEL)
def capture_model(self):
    if specialize_lora(self.vllm_config):
        # Compile base variants before the shared graph capture/workspace lock.
        with _force_lora_count(self, 0):
            self._dummy_run(num_tokens=self.max_num_tokens, is_profile=True, num_active_loras=0)
            self._dummy_run(num_tokens=1, is_profile=True, num_active_loras=0)
            torch.npu.synchronize()
    return _ORIGINAL_CAPTURE_MODEL(self)


def _install():
    if getattr(TorchCompileWithNoGuardsWrapper, "_ascend_lora_graph_patch_installed", False):
        return
    if getattr(TorchCompileWithNoGuardsWrapper, "_external_lora_acl_graph_v23_patch", False):
        raise RuntimeError("Remove the external v0.23 LoRA graph patch before using this patch")
    # Existing set_* functions resolve GraphParams in their module globals.
    # This also covers aliases imported by model_runner_v1 before worker init.
    acl_graph.GraphParams = _LoRAGraphParams
    acl_graph.weak_ref_workspaces = weak_ref_workspaces
    TorchCompileWithNoGuardsWrapper.__init__ = _wrapper_init
    TorchCompileWithNoGuardsWrapper.__call__ = _wrapper_call
    TorchCompileWithNoGuardsWrapper.aot_compile = _aot_compile
    TorchCompileWithNoGuardsWrapper._ascend_has_lora = _has_lora
    TorchCompileWithNoGuardsWrapper._ascend_mark_variant_dynamic_inputs = _mark_variant_dynamic_inputs
    decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn
    GPUModelRunner.maybe_dummy_run_with_lora = maybe_dummy_run_with_lora
    GPUModelRunner._warmup_and_capture = _warmup_and_capture
    GPUModelRunner.capture_model = capture_model
    TorchCompileWithNoGuardsWrapper._ascend_lora_graph_patch_installed = True


_install()
