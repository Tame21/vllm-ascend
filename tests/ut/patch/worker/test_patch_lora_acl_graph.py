# SPDX-License-Identifier: Apache-2.0

import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.patch.worker import patch_lora_acl_graph as lora_graph


@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("is_310p", [False, True])
def test_worker_loader_is_release_scoped(release, is_310p):
    loader = Path(lora_graph.__file__).with_name("__init__.py")
    tree = ast.parse(loader.read_text(encoding="utf-8"))
    graph_module = "vllm_ascend.patch.worker.patch_lora_acl_graph"
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Import) and any(alias.name == graph_module for alias in child.names)
            for child in ast.walk(node)
        )
    )
    imported = []

    def record_import(name, *args, **kwargs):
        imported.append(name)
        return SimpleNamespace()

    namespace = {
        "__builtins__": {"__import__": record_import},
        "vllm_version_is": lambda version: release,
        "is_310p": lambda: is_310p,
    }
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(loader), "exec"), namespace)
    expected = ["vllm_ascend.patch.worker.patch_qwen3_5_dense_lora", graph_module] if release and not is_310p else []
    assert imported == expected


def test_base_variants_compile_before_capture_and_restore_count(monkeypatch):
    runner = SimpleNamespace(vllm_config="config", max_num_tokens=32)
    events = []

    def dummy(**kwargs):
        assert runner._ascend_forced_dummy_lora_count == 0
        assert kwargs["num_active_loras"] == 0
        events.append(kwargs["num_tokens"])

    runner._dummy_run = dummy
    monkeypatch.setattr(lora_graph, "specialize_lora", lambda c: True)
    monkeypatch.setattr(
        lora_graph.torch, "npu", SimpleNamespace(synchronize=lambda: events.append("sync")), raising=False
    )
    monkeypatch.setattr(lora_graph, "_ORIGINAL_CAPTURE_MODEL", lambda r: events.append("capture"))
    lora_graph.capture_model(runner)
    assert events == [32, 1, "sync", "capture"]
    assert runner._ascend_forced_dummy_lora_count is None


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int = 4
    num_active_loras: int = 0
    uniform: bool = False


def test_graph_parameters_are_isolated_for_same_token_count(monkeypatch):
    context = SimpleNamespace(cudagraph_runtime_mode=lora_graph.CUDAGraphMode.FULL)
    monkeypatch.setattr(lora_graph, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(lora_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(lora_graph.monitor, "cudagraph_capturing_enabled", True)
    store = lora_graph._GraphParamStore({4: ["legacy"]}, list)
    descriptors = [
        Descriptor(),
        Descriptor(num_active_loras=1),
        Descriptor(num_active_loras=2),
        Descriptor(uniform=True),
    ]
    for i, descriptor in enumerate(descriptors):
        context.batch_descriptor = descriptor
        store[4].append(i)
    monkeypatch.setattr(lora_graph.monitor, "cudagraph_capturing_enabled", False)
    for i, descriptor in enumerate(descriptors):
        context.batch_descriptor = descriptor
        assert store[4] == [i]
        assert store.get(4) == [i]
        assert 4 in store
    assert dict.__getitem__(store, 4) == ["legacy"]


@pytest.mark.parametrize("operation", [lambda s: s[4], lambda s: s.get(4), lambda s: s.__setitem__(4, [])])
def test_missing_descriptor_cannot_create_replay_parameters(monkeypatch, operation):
    context = SimpleNamespace(cudagraph_runtime_mode=lora_graph.CUDAGraphMode.FULL, batch_descriptor=Descriptor())
    monkeypatch.setattr(lora_graph, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(lora_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(lora_graph.monitor, "cudagraph_capturing_enabled", False)
    store = lora_graph._GraphParamStore({4: []}, list)
    with pytest.raises(RuntimeError, match="not captured"):
        operation(store)
    assert len(store) == 1


def test_workspace_weakening_uses_physical_keys(monkeypatch):
    descriptor = Descriptor()
    context = SimpleNamespace(cudagraph_runtime_mode=lora_graph.CUDAGraphMode.FULL, batch_descriptor=descriptor)
    monkeypatch.setattr(lora_graph, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(lora_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(lora_graph.acl_graph, "weak_ref_tensors", lambda x: f"weak:{x}")
    params = SimpleNamespace(workspaces=lora_graph._GraphParamStore({4: "legacy", descriptor: "base"}, lambda: None))
    lora_graph.weak_ref_workspaces(params)
    assert dict.__getitem__(params.workspaces, 4) == "weak:legacy"
    assert params.workspaces[descriptor] == "weak:base"


@pytest.mark.parametrize("is_qwen", [True, False])
def test_fia_metadata_filter_is_qwen_only_and_does_not_mutate_context(is_qwen):
    fia = SimpleNamespace(seq_lens_list=[4], actual_seq_lengths_q=[4])
    gdn = SimpleNamespace(num_prefills=1)
    context = SimpleNamespace(attn_metadata={"gdn": gdn, "fia": fia})
    with (
        patch.object(lora_graph, "is_qwen3_5", return_value=is_qwen),
        patch.object(lora_graph, "specialize_lora", return_value=True),
        patch.object(lora_graph, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
        patch.object(lora_graph, "using_paged_attention", return_value=False),
        patch.object(lora_graph, "_ORIGINAL_UPDATE_GRAPH_PARAMS") as original,
    ):
        lora_graph.update_graph_params("stream", context, 4, "config")
    assert context.attn_metadata == {"gdn": gdn, "fia": fia}
    passed = original.call_args.args[1]
    if is_qwen:
        assert passed is not context
        assert passed.attn_metadata == {"fia": fia}
    else:
        assert passed is context


def test_dummy_lora_count_matches_capture_and_is_restored_on_error():
    runner = SimpleNamespace(vllm_config="config")

    def capture(*args):
        with lora_graph.maybe_dummy_run_with_lora(runner, "lora", "scheduled", "sampled", num_active_loras=8):
            raise RuntimeError("capture failed")

    with (
        patch.object(lora_graph, "specialize_lora", return_value=True),
        patch.object(lora_graph, "_ORIGINAL_WARMUP_AND_CAPTURE", side_effect=capture),
        patch.object(lora_graph, "_ORIGINAL_DUMMY_LORA_CONTEXT", return_value=nullcontext()) as dummy,
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        lora_graph._warmup_and_capture(runner, Descriptor(), lora_graph.CUDAGraphMode.FULL)
    assert dummy.call_args.kwargs["num_active_loras"] == 0
    assert runner._ascend_forced_dummy_lora_count is None


@pytest.mark.parametrize("aot_enabled", [False, True])
@pytest.mark.parametrize("split_aot", [False, True])
@pytest.mark.parametrize("specialize", [False, True])
def test_wrapper_init_scopes_bytecode_setting_and_aot_saver(monkeypatch, aot_enabled, split_aot, specialize):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=lora_graph.CompilationMode.VLLM_COMPILE,
            dynamic_shapes_config=SimpleNamespace(evaluate_guards=False),
        )
    )
    saver = MagicMock()
    model = SimpleNamespace(save_aot_compiled_function=saver)
    monkeypatch.setattr(lora_graph.vllm_envs, "VLLM_USE_AOT_COMPILE", aot_enabled)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_LORA_AOT", str(int(split_aot)))
    monkeypatch.setattr(lora_graph.vllm_envs, "VLLM_USE_BYTECODE_HOOK", True)
    monkeypatch.setattr(lora_graph.backends, "model_tag", "backbone")
    with (
        patch.object(lora_graph, "get_current_vllm_config", return_value=config),
        patch.object(lora_graph, "specialize_lora", return_value=specialize),
        patch.object(lora_graph, "validate_config"),
        patch.object(lora_graph, "_ORIGINAL_WRAPPER_INIT"),
        patch.object(lora_graph, "_compile_base_variant") as compile_base,
        patch.object(lora_graph.logger, "warning_once") as warning,
    ):
        lora_graph._wrapper_init(model)
    assert lora_graph.vllm_envs.VLLM_USE_BYTECODE_HOOK is True
    assert compile_base.call_count == (2 if specialize else 0)
    assert model._ascend_specialize_lora is specialize
    assert model._ascend_lora_aot_enabled is (aot_enabled and split_aot and specialize)
    assert warning.call_count == int(aot_enabled and specialize and not split_aot)
    if aot_enabled and split_aot and specialize:
        assert model.save_aot_compiled_function.__func__ is lora_graph._save_aot_compiled_function
    else:
        assert model.save_aot_compiled_function is saver


def test_lora_manager_marks_each_punica_wrapper_for_graph_routing():
    shared_wrapper = SimpleNamespace()
    language_wrapper = SimpleNamespace()
    manager = SimpleNamespace()

    def initialize_manager(*args, **kwargs):
        manager.punica_wrapper_mapping = {
            "language_model": language_wrapper,
            "vision_tower": shared_wrapper,
            "multi_modal_projector": shared_wrapper,
        }

    with (
        patch.object(lora_graph, "validate_config") as validate,
        patch.object(lora_graph, "specialize_lora", return_value=True),
        patch.object(
            lora_graph,
            "_ORIGINAL_MANAGER_INIT",
            side_effect=initialize_manager,
        ) as original,
    ):
        lora_graph._lora_manager_init(
            manager,
            "model",
            8,
            64,
            32000,
            "lora_config",
            "npu",
            "vllm_config",
        )

    validate.assert_called_once_with("vllm_config")
    original.assert_called_once()
    assert language_wrapper._ascend_graph_specialize_lora is True
    assert shared_wrapper._ascend_graph_specialize_lora is True


@pytest.mark.parametrize(
    ("has_lora", "num_tokens", "expected"),
    [
        (True, 1, lora_graph._VARIANT_LORA),
        (False, 1, lora_graph._VARIANT_BASE_ONE),
        (False, 2, lora_graph._VARIANT_BASE),
    ],
)
def test_select_variant(has_lora, num_tokens, expected):
    model = SimpleNamespace(_ascend_has_lora=lambda: has_lora)
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=num_tokens))
    with (
        patch.object(lora_graph, "is_forward_context_available", return_value=True),
        patch.object(lora_graph, "get_forward_context", return_value=context),
    ):
        assert lora_graph._select_variant(model) == expected


@pytest.mark.parametrize(
    ("mapping", "expected_no_lora"),
    [([0, 0], True), ([0, 2], False)],
)
def test_update_lora_metadata_tracks_decode_lora_state(mapping, expected_no_lora):
    wrapper = SimpleNamespace(
        _ascend_graph_specialize_lora=True,
        no_lora=not expected_no_lora,
    )
    lora_mapping = SimpleNamespace(index_mapping=mapping)
    with patch.object(lora_graph, "_ORIGINAL_UPDATE_LORA_METADATA") as original:
        lora_graph._update_lora_metadata(wrapper, lora_mapping)

    original.assert_called_once_with(wrapper, lora_mapping)
    assert wrapper.no_lora is expected_no_lora


def test_no_lora_graph_guard_skips_lora_operator():
    original = MagicMock(return_value="applied")
    guarded = lora_graph._no_lora_graph_guard(original)
    wrapper = SimpleNamespace(_ascend_graph_specialize_lora=True, no_lora=True)

    assert guarded(wrapper, "input") is None
    original.assert_not_called()


@pytest.mark.parametrize(
    ("has_lora", "num_tokens", "expected"),
    [(True, 2, "lora"), (False, 2, "base"), (False, 1, "base_one")],
)
@pytest.mark.parametrize("split_aot", [False, True])
def test_ordinary_compile_routes_variants_without_aot_dispatcher(
    monkeypatch, has_lora, num_tokens, expected, split_aot
):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_LORA_AOT", str(int(split_aot)))
    model = SimpleNamespace(
        _ascend_specialize_lora=True,
        _ascend_has_lora=lambda: has_lora,
        _compiled_callable=MagicMock(return_value="lora"),
        _ascend_base_callable=MagicMock(return_value="base"),
        _ascend_base_one_callable=MagicMock(return_value="base_one"),
        _ascend_base_dynamic_inputs_marked=False,
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
        first_compile=True,
        evaluate_guards=False,
        _call_with_optional_nvtx_range=lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=num_tokens))
    with (
        patch.object(lora_graph, "is_forward_context_available", return_value=True),
        patch.object(lora_graph, "get_forward_context", return_value=context),
        patch.object(lora_graph.wrapper_module, "_compilation_context", side_effect=nullcontext),
    ):
        assert lora_graph._wrapper_call(model, "input") == expected
        assert lora_graph._wrapper_call(model, "input") == expected
    assert model.first_compile is False
    assert model._ascend_mark_variant_dynamic_inputs.call_count == (1 if expected == "base" else 0)
