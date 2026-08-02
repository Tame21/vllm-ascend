"""Install the LoRA patch after NPUModelRunner finishes importing."""

import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from threading import RLock
from types import ModuleType
from typing import Any


_TARGET_MODULE = "vllm_ascend.worker.model_runner_v1"
_LOCK = RLock()
_FINDER: "_ModelRunnerPatchFinder | None" = None
_APPLIED = False


def _disable_bytecode_hook() -> None:
    # Set the process environment before vllm.envs is imported.
    os.environ["VLLM_USE_BYTECODE_HOOK"] = "0"

    # vllm.envs may already be initialized by another startup thread.
    loaded_envs = sys.modules.get("vllm.envs")
    if loaded_envs is not None:
        loaded_envs.VLLM_USE_BYTECODE_HOOK = False


def _is_initializing(module: ModuleType) -> bool:
    spec = getattr(module, "__spec__", None)
    return bool(getattr(spec, "_initializing", False))


def _apply_after_model_runner_import() -> None:
    global _APPLIED
    with _LOCK:
        if _APPLIED:
            return

        from new_23.apply_patches import apply_lora_patch

        apply_lora_patch()
        _APPLIED = True


class _AfterExecLoader(Loader):
    def __init__(self, loader: Loader) -> None:
        self._loader = loader

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        _apply_after_model_runner_import()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _ModelRunnerPatchFinder(MetaPathFinder):
    def __init__(self) -> None:
        self._used = False

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        del target
        if fullname != _TARGET_MODULE:
            return None

        with _LOCK:
            if self._used:
                return None
            self._used = True
            if self in sys.meta_path:
                sys.meta_path.remove(self)

        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find loader for {fullname}")
        if not hasattr(spec.loader, "exec_module"):
            raise ImportError(f"Loader for {fullname} has no exec_module")

        spec.loader = _AfterExecLoader(spec.loader)
        return spec


def install_lora_patch_hook() -> None:
    """Apply now if model_runner is ready, otherwise hook its first import."""
    global _FINDER
    _disable_bytecode_hook()

    with _LOCK:
        if _APPLIED or _FINDER is not None:
            return

        model_runner = sys.modules.get(_TARGET_MODULE)
        if model_runner is not None:
            if _is_initializing(model_runner):
                raise RuntimeError(
                    "The LoRA patch hook must be installed before "
                    f"{_TARGET_MODULE} starts importing."
                )
            _apply_after_model_runner_import()
            return

        _FINDER = _ModelRunnerPatchFinder()
        sys.meta_path.insert(0, _FINDER)


install_lora_patch_hook()
