# SPDX-License-Identifier: Apache-2.0
"""Run LoRA patch unit tests with external vLLM/NPU dependencies stubbed.

The production patch modules are imported intact, including installation.
This verifies Python routing/cache behavior, not actual AOT or NPU execution.
Run from the repository root with ``python -m tools.verify_lora_graph_cpu``.
"""

import importlib
import sys
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch  # noqa: F401

REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]


def module(name, **attrs):
    if name in sys.modules:
        target = sys.modules[name]
    else:
        target = ModuleType(name)
        target.__path__ = []
        sys.modules[name] = target
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(module(parent), child, target)
    target.__dict__.update(attrs)
    return target


def no_op(*args, **kwargs):
    pass


class Wrapper:
    __init__ = no_op
    __call__ = no_op
    aot_compile = no_op


class Runner:
    maybe_dummy_run_with_lora = staticmethod(lambda *args, **kwargs: nullcontext())
    _warmup_and_capture = no_op
    capture_model = no_op


class Manager:
    __init__ = no_op


class Punica:
    _shrink_prefill = no_op
    _shrink_decode = no_op
    _expand_slice_prefill = no_op
    _expand_slice_decode = no_op
    update_metadata = no_op
    add_shrink = no_op
    add_expand = no_op
    add_lora_embedding = no_op
    add_lora_linear = no_op
    add_lora_logits = no_op


class WorkerManager:
    _load_adapter = no_op


class Platform:
    check_and_update_config = classmethod(no_op)


class Attention:
    update_graph_params = staticmethod(no_op)


class GraphParams:
    def __init__(self):
        self.events = {}
        self.workspaces = {}
        self.handles = {}
        self.attn_params = {}


class GraphMode(Enum):
    NONE = 0
    FULL = 1


class CompileMode(Enum):
    VLLM_COMPILE = 1


module("vllm.envs", VLLM_USE_AOT_COMPILE=True, VLLM_USE_BYTECODE_HOOK=True, VLLM_DISABLE_COMPILE_CACHE=False)
module("vllm.compilation.backends", model_tag="backbone")
module(
    "vllm.compilation.monitor", monitor_torch_compile=lambda *a, **k: nullcontext(), cudagraph_capturing_enabled=False
)
module("vllm.compilation.decorators", _try_load_aot_compiled_fn=MagicMock(return_value=None))
module("vllm.compilation.counter", compilation_counter=SimpleNamespace(num_aot_compiles=0, num_aot_artifacts_saved=0))
module("vllm.compilation.wrapper", TorchCompileWithNoGuardsWrapper=Wrapper, _compilation_context=nullcontext)
module("vllm.config", CompilationMode=CompileMode, CUDAGraphMode=GraphMode, get_current_vllm_config=MagicMock())
module("vllm.forward_context", get_forward_context=MagicMock(), is_forward_context_available=lambda: False)
module("vllm.logger", logger=MagicMock())
module("vllm.lora.model_manager", LoRAModelManager=Manager)
module("vllm.lora.worker_manager", WorkerLoRAManager=WorkerManager)
module("vllm.v1.worker.gpu_model_runner", GPUModelRunner=Runner)
module("vllm_ascend.ascend_forward_context", _EXTRA_CTX=SimpleNamespace(is_draft_model=False))
module("vllm_ascend.attention.attention_v1", AscendAttentionBackendImpl=Attention)
module("vllm_ascend.attention.utils", using_paged_attention=lambda *a: False)
module(
    "vllm_ascend.compilation.acl_graph",
    GraphParams=GraphParams,
    weak_ref_workspaces=no_op,
    weak_ref_tensors=lambda x: x,
)
module("vllm_ascend.lora.punica_npu", PunicaWrapperNPU=Punica)
module("vllm_ascend.platform", NPUPlatform=Platform)
module("vllm_ascend.utils", is_310p=lambda: False, vllm_version_is=lambda version: True)
module("vllm_ascend", __path__=[str(REPO / "vllm_ascend")])
module("vllm_ascend.patch", __path__=[str(REPO / "vllm_ascend/patch")])
module("vllm_ascend.patch.platform", __path__=[str(REPO / "vllm_ascend/patch/platform")])
module("vllm_ascend.patch.worker", __path__=[str(REPO / "vllm_ascend/patch/worker")])

# Import the real modules, including their installation hooks.
importlib.import_module("vllm_ascend.patch.worker.patch_qwen3_5_dense_lora")
graph = importlib.import_module("vllm_ascend.patch.worker.patch_lora_acl_graph")
for method in ("_shrink_prefill", "_shrink_decode", "_expand_slice_prefill", "_expand_slice_decode"):
    assert getattr(Punica, method) is no_op, f"Unexpected LoRA kernel replacement: {method}"
assert Wrapper.__call__ is graph._wrapper_call
assert Wrapper.aot_compile is graph._aot_compile
print("Real patches loaded with CPU dependency stubs; no NPU execution is performed.")
raise SystemExit(
    pytest.main(
        [
            "--noconftest",
            "-q",
            str(REPO / "tests/ut/patch/worker/test_patch_lora_acl_graph.py"),
            str(REPO / "tests/ut/patch/worker/test_patch_lora_aot.py"),
            str(REPO / "tests/ut/patch/worker/test_patch_qwen3_5_dense_lora.py"),
            str(REPO / "tests/ut/patch/platform/test_patch_lora_graph_config.py"),
        ]
    )
)
