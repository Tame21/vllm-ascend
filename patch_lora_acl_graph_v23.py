#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep base and LoRA ACL graphs independent on the vLLM v0.23 baseline."""

import hashlib
import inspect
from contextlib import contextmanager, nullcontext
from types import FunctionType, MethodType
from typing import Any

import torch
import vllm.envs as vllm_envs
from vllm.compilation import backends as backends_module
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

_SUPPORTED_WRAPPER_SHA256 = (
    "361ddf8ce90e2d2bb3dfa0e563b15cc04bf7d5d2215137bd8abe9e1966537af7"
)
_PATCH_MARKER = "_vllm_ascend_lora_acl_graph_v23_patch"
_ORIGINAL_INIT_ATTR = "_vllm_ascend_lora_acl_graph_v23_original_init"
_BASELINE_VERIFIED = False


def _verify_vllm_wrapper_baseline() -> None:
    global _BASELINE_VERIFIED
    if _BASELINE_VERIFIED:
        return

    wrapper_path = inspect.getsourcefile(wrapper_module)
    if wrapper_path is None:
        raise RuntimeError("Cannot locate the vLLM compilation wrapper source")
    with open(wrapper_path, "rb") as wrapper_file:
        wrapper_source = wrapper_file.read().replace(b"\r\n", b"\n")
        actual_sha256 = hashlib.sha256(wrapper_source).hexdigest()
    if actual_sha256 != _SUPPORTED_WRAPPER_SHA256:
        raise RuntimeError(
            "The Ascend v0.23 LoRA ACL-graph patch requires the verified "
            "vLLM v0.23.0 compilation wrapper. Expected SHA256 "
            f"{_SUPPORTED_WRAPPER_SHA256}, got {actual_sha256} from "
            f"{wrapper_path}."
        )
    _BASELINE_VERIFIED = True


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
    """Resolve token-count access to the active full-graph descriptor."""

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

    def __contains__(self, key):
        return dict.__contains__(self, self._resolve_key(key))

    def __getitem__(self, key):
        resolved_key = self._resolve_key(key)
        if not dict.__contains__(self, resolved_key):
            dict.__setitem__(self, resolved_key, self.default_factory())
        return dict.__getitem__(self, resolved_key)

    def __setitem__(self, key, value):
        dict.__setitem__(self, self._resolve_key(key), value)

    def get(self, key, default=None):
        return dict.get(self, self._resolve_key(key), default)


def _make_graph_params(capture_sizes: list[int]) -> GraphParams:
    return GraphParams(
        _GraphParamStore(capture_sizes, list),
        _GraphParamStore(capture_sizes, lambda: None),
        _GraphParamStore(capture_sizes, list),
        _GraphParamStore(capture_sizes, list),
    )


def _set_graph_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    acl_graph_module._graph_params = _make_graph_params(capture_sizes)


def _set_draft_graph_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    acl_graph_module._draft_graph_params = _make_graph_params(capture_sizes)


def _set_draft_graph_prefill_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph prefill parameters have already been set!")
    acl_graph_module._draft_graph_prefill_params = _make_graph_params(capture_sizes)


def _weak_ref_workspaces(params) -> None:
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


def _patch_compile_wrapper() -> None:
    original_init = getattr(
        TorchCompileWithNoGuardsWrapper,
        _ORIGINAL_INIT_ATTR,
        TorchCompileWithNoGuardsWrapper.__init__,
    )

    def patched_init(
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
            _verify_vllm_wrapper_baseline()
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

        self._ascend_specialize_lora = specialize_lora
        self._ascend_use_bytecode_hook = use_bytecode_hook
        self._ascend_base_dynamic_inputs_marked = False
        self._ascend_punica_wrappers = None
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
        self._ascend_base_compiled_callable = torch.compile(
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
        self._ascend_base_one_compiled_callable = torch.compile(
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

    def punica_has_lora(self: Any) -> bool | None:
        if self._ascend_punica_wrappers is None:
            wrappers: list[Any] = []
            seen: set[int] = set()
            for module in self.modules():
                punica_wrapper = getattr(module, "punica_wrapper", None)
                if punica_wrapper is not None and id(punica_wrapper) not in seen:
                    wrappers.append(punica_wrapper)
                    seen.add(id(punica_wrapper))
            self._ascend_punica_wrappers = wrappers
        if not self._ascend_punica_wrappers:
            return None
        return any(
            not wrapper.no_lora for wrapper in self._ascend_punica_wrappers
        )

    def mark_variant_dynamic_inputs(self: Any, *args: Any, **kwargs: Any) -> None:
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

    def patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        compiled_callable = self._compiled_callable
        use_base_callable = False
        if self._ascend_specialize_lora and is_forward_context_available():
            batch_descriptor = get_forward_context().batch_descriptor
            has_lora = (
                batch_descriptor.has_lora
                if batch_descriptor is not None
                else True
            )
            punica_state = self._ascend_punica_has_lora()
            if punica_state is not None:
                has_lora = punica_state
            if not has_lora:
                use_base_callable = True
                if (
                    batch_descriptor is not None
                    and batch_descriptor.num_tokens == 1
                ):
                    compiled_callable = self._ascend_base_one_compiled_callable
                else:
                    compiled_callable = self._ascend_base_compiled_callable

        if (
            use_base_callable
            and compiled_callable is self._ascend_base_compiled_callable
            and not self._ascend_base_dynamic_inputs_marked
        ):
            self._ascend_mark_variant_dynamic_inputs(*args, **kwargs)
            self._ascend_base_dynamic_inputs_marked = True

        if self._ascend_use_bytecode_hook:
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

    setattr(TorchCompileWithNoGuardsWrapper, _ORIGINAL_INIT_ATTR, original_init)
    TorchCompileWithNoGuardsWrapper.__init__ = patched_init
    TorchCompileWithNoGuardsWrapper.__call__ = patched_call
    TorchCompileWithNoGuardsWrapper._ascend_punica_has_lora = punica_has_lora
    TorchCompileWithNoGuardsWrapper._ascend_mark_variant_dynamic_inputs = (
        mark_variant_dynamic_inputs
    )


def _patch_graph_capture_warmup() -> None:
    original_capture_model = GPUModelRunner.capture_model
    original_warmup_and_capture = GPUModelRunner._warmup_and_capture
    original_dummy_lora_context = GPUModelRunner.maybe_dummy_run_with_lora

    @contextmanager
    def patched_dummy_lora_context(
        self,
        lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        remove_lora=True,
        num_active_loras=0,
        mapping_type=None,
    ):
        forced_count = getattr(
            self,
            "_ascend_forced_dummy_lora_count",
            None,
        )
        if forced_count is not None:
            num_active_loras = forced_count
        kwargs = {
            "remove_lora": remove_lora,
            "num_active_loras": num_active_loras,
        }
        if mapping_type is not None:
            kwargs["mapping_type"] = mapping_type
        with original_dummy_lora_context(
            self,
            lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            **kwargs,
        ):
            if forced_count == 0:
                _set_punica_no_lora(self, True)
            yield

    def patched_warmup_and_capture(
        self,
        desc,
        cudagraph_runtime_mode,
        *args,
        **kwargs,
    ):
        previous = getattr(self, "_ascend_forced_dummy_lora_count", None)
        self._ascend_forced_dummy_lora_count = desc.num_active_loras
        try:
            return original_warmup_and_capture(
                self,
                desc,
                cudagraph_runtime_mode,
                *args,
                **kwargs,
            )
        finally:
            self._ascend_forced_dummy_lora_count = previous

    def patched_capture_model(self) -> int:
        if (
            self.lora_config is not None
            and self.compilation_config.cudagraph_specialize_lora
        ):
            previous = getattr(self, "_ascend_forced_dummy_lora_count", None)
            self._ascend_forced_dummy_lora_count = 0
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
                self._ascend_forced_dummy_lora_count = previous
        return original_capture_model(self)

    GPUModelRunner.maybe_dummy_run_with_lora = patched_dummy_lora_context
    GPUModelRunner._warmup_and_capture = patched_warmup_and_capture
    GPUModelRunner.capture_model = patched_capture_model


def _set_punica_no_lora(model_runner: Any, no_lora: bool) -> None:
    seen: set[int] = set()
    for module in model_runner.model.modules():
        punica_wrapper = getattr(module, "punica_wrapper", None)
        if punica_wrapper is None or id(punica_wrapper) in seen:
            continue
        punica_wrapper.no_lora = no_lora
        seen.add(id(punica_wrapper))


def apply_patch() -> None:
    if getattr(TorchCompileWithNoGuardsWrapper, _PATCH_MARKER, False):
        return

    _patch_compile_wrapper()
    acl_graph_module.set_graph_params = _set_graph_params
    acl_graph_module.set_draft_graph_params = _set_draft_graph_params
    acl_graph_module.set_draft_graph_prefill_params = (
        _set_draft_graph_prefill_params
    )
    acl_graph_module.weak_ref_workspaces = _weak_ref_workspaces
    _patch_graph_capture_warmup()
    setattr(TorchCompileWithNoGuardsWrapper, _PATCH_MARKER, True)
    logger.info_once("Applied vLLM v0.23 base/LoRA ACL-graph isolation patch")


apply_patch()
