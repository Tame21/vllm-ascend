# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend.patch.worker import patch_lora_acl_graph as lora_graph
from vllm_ascend.patch.worker import patch_lora_aot as lora_aot


def test_aot_dispatcher_routes_loaded_variants():
    model = object()
    dispatcher = lora_aot._AOTVariantDispatcher()
    artifacts = {variant: MagicMock(return_value=variant) for variant in lora_aot._AOT_VARIANTS}
    for variant, artifact in artifacts.items():
        dispatcher.add_loaded(variant, artifact)

    for variant in lora_aot._AOT_VARIANTS:
        with patch.object(lora_graph, "_select_variant", return_value=variant):
            assert dispatcher(model, "input") == variant
        artifacts[variant].assert_called_once_with(model, "input")


def test_aot_dispatcher_lazily_compiles_missing_variant(monkeypatch):
    artifact = MagicMock(return_value="output")
    base_callable = MagicMock()
    base_callable.aot_compile.return_value = artifact
    model = SimpleNamespace(
        _compiled_callable=MagicMock(),
        _ascend_base_callable=base_callable,
        _ascend_base_one_callable=MagicMock(),
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
        vllm_config=MagicMock(),
        _is_encoder=False,
    )
    dispatcher = lora_aot._AOTVariantDispatcher()
    monkeypatch.setattr(lora_aot.vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", True)
    monkeypatch.setattr(lora_aot.compilation_counter, "num_aot_compiles", 0)

    with (
        patch.object(lora_graph, "_select_variant", return_value=lora_graph._VARIANT_BASE),
        patch.object(
            lora_aot.monitor,
            "monitor_torch_compile",
            return_value=nullcontext(),
        ),
    ):
        assert dispatcher(model, "input") == "output"

    model._ascend_mark_variant_dynamic_inputs.assert_called_once_with(lora_graph._VARIANT_BASE, "input")
    base_callable.aot_compile.assert_called_once_with((("input",), {}))
    artifact.assert_called_once_with(model, "input")


def test_dispatcher_compiles_each_variant_once_for_alternating_requests(monkeypatch):
    state = SimpleNamespace(has_lora=True)
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=4))
    callables = {variant: MagicMock() for variant in lora_aot._AOT_VARIANTS}
    for variant, compiled in callables.items():
        compiled.aot_compile.return_value = MagicMock(return_value=variant)
    model = SimpleNamespace(
        _ascend_specialize_lora=True,
        _ascend_has_lora=lambda: state.has_lora,
        _compiled_callable=callables["lora"],
        _ascend_base_callable=callables["base"],
        _ascend_base_one_callable=callables["base_one"],
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
        vllm_config=MagicMock(),
        _is_encoder=False,
    )
    monkeypatch.setattr(lora_aot.vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", True)
    monkeypatch.setattr(lora_aot.compilation_counter, "num_aot_compiles", 0)
    with (
        patch.object(lora_graph, "is_forward_context_available", return_value=True),
        patch.object(lora_graph, "get_forward_context", return_value=context),
        patch.object(lora_aot.monitor, "monitor_torch_compile", side_effect=lambda *a, **k: nullcontext()),
    ):
        model.aot_compiled_fn = lora_aot._aot_compile(model, "input")
        for has_lora, tokens, expected in (
            (True, 4, "lora"),
            (False, 4, "base"),
            (True, 4, "lora"),
            (False, 1, "base_one"),
            (False, 4, "base"),
            (False, 1, "base_one"),
        ):
            state.has_lora = has_lora
            context.batch_descriptor.num_tokens = tokens
            # Match decorators.py's direct AOT fast path, not _wrapper_call.
            assert model.aot_compiled_fn(model, "input") == expected
    for compiled in callables.values():
        compiled.aot_compile.assert_called_once_with((("input",), {}))
        assert compiled.aot_compile.return_value.call_count == 2
    assert lora_aot.compilation_counter.num_aot_compiles == 2  # Initial compile is counted by vLLM.


def test_try_loads_each_aot_variant(tmp_path):
    model = SimpleNamespace(_ascend_specialize_lora=True)
    artifacts = {variant: MagicMock(name=variant) for variant in lora_aot._AOT_VARIANTS}
    aot_path = str(tmp_path / "model")

    def load_artifact(_model, path):
        return artifacts[path.rsplit(".", 1)[-1]]

    with (
        patch.object(
            lora_aot,
            "_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN",
            side_effect=load_artifact,
        ) as load,
        patch.object(lora_graph, "_select_variant", return_value=lora_graph._VARIANT_LORA),
    ):
        dispatcher = lora_aot._try_load_aot_compiled_fn(model, aot_path)

    assert isinstance(dispatcher, lora_aot._AOTVariantDispatcher)
    assert dispatcher.artifacts == artifacts
    assert load.call_args_list == [call(model, f"{aot_path}.{variant}") for variant in lora_aot._AOT_VARIANTS]


def test_partial_aot_cache_is_reused_when_selected_variant_is_missing(tmp_path):
    base_artifact = MagicMock(name="base")
    lora_artifact = MagicMock(name="lora")
    lora_callable = MagicMock()
    lora_callable.aot_compile.return_value = lora_artifact
    model = SimpleNamespace(
        _ascend_specialize_lora=True,
        _compiled_callable=lora_callable,
        _ascend_base_callable=MagicMock(),
        _ascend_base_one_callable=MagicMock(),
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
    )
    aot_path = str(tmp_path / "model")

    def load_artifact(_model, path):
        return base_artifact if path.endswith(f".{lora_graph._VARIANT_BASE}") else None

    with (
        patch.object(
            lora_aot,
            "_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN",
            side_effect=load_artifact,
        ),
        patch.object(lora_graph, "_select_variant", return_value=lora_graph._VARIANT_LORA),
    ):
        assert lora_aot._try_load_aot_compiled_fn(model, aot_path) is None
        dispatcher = lora_aot._aot_compile(model, "input")

    assert dispatcher.artifacts == {
        lora_graph._VARIANT_BASE: base_artifact,
        lora_graph._VARIANT_LORA: lora_artifact,
    }
    assert not hasattr(model, "_ascend_preloaded_aot_dispatcher")


def test_saves_each_aot_variant_to_an_independent_path(tmp_path, monkeypatch):
    class Artifact:
        def __init__(self, value):
            self.value = value

        def save_compiled_function(self, path):
            Path(path).write_text(self.value, encoding="utf-8")

    dispatcher = lora_aot._AOTVariantDispatcher()
    dispatcher.artifacts = {variant: Artifact(variant) for variant in lora_aot._AOT_VARIANTS}
    dispatcher.dirty_variants = set(lora_aot._AOT_VARIANTS)
    model = SimpleNamespace(
        _aot_cache_dir=str(tmp_path),
        _aot_compilation_path=str(tmp_path / "model"),
    )
    monkeypatch.setattr(lora_aot.vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", False)
    monkeypatch.setattr(lora_aot.compilation_counter, "num_aot_artifacts_saved", 0)

    dispatcher.save_dirty(model)

    assert not dispatcher.dirty_variants
    for variant in lora_aot._AOT_VARIANTS:
        path = tmp_path / f"model.{variant}"
        assert path.read_text(encoding="utf-8") == variant


@pytest.mark.parametrize("aot_enabled", [False, True])
@pytest.mark.parametrize("specialize", [False, True])
def test_wrapper_init_only_replaces_saver_for_specialized_aot(monkeypatch, aot_enabled, specialize):
    original_saver = MagicMock()
    model = SimpleNamespace(save_aot_compiled_function=original_saver)

    def initialize(model, **kwargs):
        model._ascend_specialize_lora = specialize

    monkeypatch.setattr(lora_aot.vllm_envs, "VLLM_USE_AOT_COMPILE", aot_enabled)
    with patch.object(lora_aot, "_ORIGINAL_WRAPPER_INIT", side_effect=initialize) as original:
        lora_aot._wrapper_init(model, "prefix", is_encoder=False)
    original.assert_called_once_with(model, compile_prefix="prefix", is_encoder=False)
    if aot_enabled and specialize:
        assert model.save_aot_compiled_function.__self__ is model
        assert model.save_aot_compiled_function.__func__ is lora_aot._save_aot_compiled_function
    else:
        assert model.save_aot_compiled_function is original_saver


@pytest.mark.parametrize("aot_enabled", [False, True])
def test_base_callable_aot_context_is_scoped(monkeypatch, aot_enabled):
    active = []
    context = MagicMock()
    context.__enter__.side_effect = lambda: active.append(True)
    context.__exit__.side_effect = lambda *args: active.clear()
    dynamo_config = SimpleNamespace(enable_aot_compile=False, patch=MagicMock(return_value=context))
    model = object()

    def compile_base(*args):
        assert bool(active) is aot_enabled
        return "compiled"

    monkeypatch.setattr(lora_aot.vllm_envs, "VLLM_USE_AOT_COMPILE", aot_enabled)
    with (
        patch.object(lora_aot.torch._dynamo, "config", dynamo_config),
        patch.object(lora_aot, "_ORIGINAL_COMPILE_BASE_VARIANT", side_effect=compile_base) as original,
    ):
        assert lora_aot._compile_base_variant(model, "prefix", "suffix") == "compiled"
    original.assert_called_once_with(model, "prefix", "suffix")
    assert not active
    assert dynamo_config.patch.call_count == int(aot_enabled)


def test_unspecialized_model_uses_upstream_aot_compile_and_load():
    model = SimpleNamespace(_ascend_specialize_lora=False)
    with (
        patch.object(lora_aot, "_ORIGINAL_WRAPPER_AOT_COMPILE", return_value="compiled") as compile_aot,
        patch.object(lora_aot, "_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN", return_value="loaded") as load,
    ):
        assert lora_aot._aot_compile(model, "input", flag=True) == "compiled"
        assert lora_aot._try_load_aot_compiled_fn(model, "model") == "loaded"
    compile_aot.assert_called_once_with(model, "input", flag=True)
    load.assert_called_once_with(model, "model")


def test_empty_aot_cache_falls_back_to_first_compile():
    model = SimpleNamespace(_ascend_specialize_lora=True)
    with patch.object(lora_aot, "_ORIGINAL_TRY_LOAD_AOT_COMPILED_FN", return_value=None) as load:
        assert lora_aot._try_load_aot_compiled_fn(model, "model") is None
    assert load.call_count == 3
    assert not hasattr(model, "_ascend_preloaded_aot_dispatcher")


def test_single_token_base_does_not_mark_dynamic_inputs():
    model = SimpleNamespace(
        _ascend_base_one_callable=MagicMock(),
        _ascend_mark_variant_dynamic_inputs=MagicMock(),
    )
    dispatcher = lora_aot._AOTVariantDispatcher()
    dispatcher.compile_variant(model, lora_graph._VARIANT_BASE_ONE, ("input",), {})
    model._ascend_mark_variant_dynamic_inputs.assert_not_called()
    model._ascend_base_one_callable.aot_compile.assert_called_once_with((("input",), {}))


def test_failed_save_keeps_variant_dirty_and_removes_temporary_file(tmp_path, monkeypatch):
    class BrokenArtifact:
        def save_compiled_function(self, path):
            Path(path).write_text("partial", encoding="utf-8")
            raise OSError("disk full")

    dispatcher = lora_aot._AOTVariantDispatcher()
    dispatcher.artifacts["lora"] = BrokenArtifact()
    dispatcher.dirty_variants.add("lora")
    model = SimpleNamespace(_aot_cache_dir=str(tmp_path), _aot_compilation_path=str(tmp_path / "model"))
    existing_artifact = tmp_path / "model.lora"
    existing_artifact.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(lora_aot.vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", False)
    monkeypatch.setattr(lora_aot.compilation_counter, "num_aot_artifacts_saved", 0)
    dispatcher.save_dirty(model)
    assert dispatcher.dirty_variants == {"lora"}
    assert lora_aot.compilation_counter.num_aot_artifacts_saved == 0
    assert existing_artifact.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_install_is_idempotent_and_does_not_replace_ordinary_call():
    wrapper = lora_aot.TorchCompileWithNoGuardsWrapper
    original_init = wrapper.__init__
    original_compile = wrapper.aot_compile
    original_load = lora_aot.decorators._try_load_aot_compiled_fn
    original_call = wrapper.__call__
    lora_aot._install()
    assert wrapper.__init__ is original_init
    assert wrapper.aot_compile is original_compile
    assert lora_aot.decorators._try_load_aot_compiled_fn is original_load
    assert wrapper.__call__ is original_call
    assert original_call is lora_graph._wrapper_call
