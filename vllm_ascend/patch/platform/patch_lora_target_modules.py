# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from argparse import Namespace
from collections.abc import Iterable
from functools import wraps
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import get_adapter_absolute_path

logger = init_logger(__name__)

_ORIGINAL_FROM_CLI_ARGS_ATTRIBUTE = "_ascend_original_from_cli_args"


def _get_static_lora_paths(args: Namespace) -> Iterable[str]:
    for lora_module in getattr(args, "lora_modules", None) or ():
        yield lora_module.path

    default_mm_loras = getattr(args, "default_mm_loras", None) or {}
    yield from default_mm_loras.values()


def _load_target_modules(lora_path: str) -> list[str]:
    resolved_path = get_adapter_absolute_path(lora_path)
    peft_helper = PEFTHelper.from_local_dir(
        resolved_path,
        max_position_embeddings=None,
    )
    target_modules = peft_helper.target_modules
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError(
            "Automatic LoRA target-module inference requires "
            f"{resolved_path}/adapter_config.json to contain a non-empty "
            "'target_modules' list. Pass --lora-target-modules explicitly "
            "when the PEFT config uses a regular expression."
        )
    if not all(isinstance(module_name, str) and module_name for module_name in target_modules):
        raise ValueError(
            "Automatic LoRA target-module inference requires every entry in "
            f"{resolved_path}/adapter_config.json 'target_modules' to be a non-empty string."
        )
    return target_modules


def _infer_lora_target_modules(args: Namespace) -> None:
    if not getattr(args, "enable_lora", False):
        return
    if getattr(args, "lora_target_modules", None) is not None:
        return

    lora_paths = tuple(dict.fromkeys(_get_static_lora_paths(args)))
    if not lora_paths:
        return

    target_modules: set[str] = set()
    for lora_path in lora_paths:
        try:
            target_modules.update(_load_target_modules(lora_path))
        except Exception as error:
            raise ValueError(
                "Failed to infer LoRA target modules from the static adapter "
                f"at {lora_path!r}."
            ) from error

    args.lora_target_modules = sorted(target_modules)
    logger.info(
        "Inferred LoRA target modules from %d static adapter(s): %s",
        len(lora_paths),
        args.lora_target_modules,
    )


def _patch_async_engine_args_from_cli_args() -> None:
    if hasattr(AsyncEngineArgs, _ORIGINAL_FROM_CLI_ARGS_ATTRIBUTE):
        return

    original_from_cli_args = AsyncEngineArgs.from_cli_args.__func__
    setattr(
        AsyncEngineArgs,
        _ORIGINAL_FROM_CLI_ARGS_ATTRIBUTE,
        original_from_cli_args,
    )

    @classmethod
    @wraps(original_from_cli_args)
    def from_cli_args(cls: type[AsyncEngineArgs], args: Namespace) -> Any:
        _infer_lora_target_modules(args)
        return original_from_cli_args(cls, args)

    AsyncEngineArgs.from_cli_args = from_cli_args


_patch_async_engine_args_from_cli_args()
