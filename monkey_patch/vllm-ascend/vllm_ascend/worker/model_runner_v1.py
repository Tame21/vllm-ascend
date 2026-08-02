"""NPUModelRunner graph-capture replacements and alias assignments."""

import sys
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.worker import model_runner_v1 as model_runner_module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


ORIGINAL_CAPTURE_MODEL = NPUModelRunner.capture_model
ORIGINAL_MAYBE_DUMMY_RUN_WITH_LORA = (
    NPUModelRunner.maybe_dummy_run_with_lora
)
ORIGINAL_WARMUP_AND_CAPTURE = NPUModelRunner._warmup_and_capture


def _set_punica_no_lora(model_runner: Any, no_lora: bool) -> None:
    seen: set[int] = set()
    for module in model_runner.model.modules():
        punica_wrapper = getattr(module, "punica_wrapper", None)
        if punica_wrapper is None or id(punica_wrapper) in seen:
            continue
        punica_wrapper.no_lora = no_lora
        seen.add(id(punica_wrapper))


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
    with ORIGINAL_MAYBE_DUMMY_RUN_WITH_LORA(
        self,
        lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        **kwargs,
    ):
        if forced_count == 0:
            _set_punica_no_lora(self, True)
        yield


def warmup_and_capture(
    self,
    desc,
    cudagraph_runtime_mode,
    *args,
    **kwargs,
):
    previous = getattr(self, "_external_forced_dummy_lora_count", None)
    self._external_forced_dummy_lora_count = desc.num_active_loras
    try:
        return ORIGINAL_WARMUP_AND_CAPTURE(
            self,
            desc,
            cudagraph_runtime_mode,
            *args,
            **kwargs,
        )
    finally:
        self._external_forced_dummy_lora_count = previous


def capture_model(self) -> int:
    """Precompile base variants, then run the original NPU graph capture."""
    parent_module_name = model_runner_module._get_gpu_model_runner_module_name(
        self
    )
    with (
        model_runner_module._torch_cuda_wrapper(),
        model_runner_module._replace_gpu_model_runner_function_wrapper(
            parent_module_name
        ),
    ):
        if (
            self.lora_config is not None
            and self.compilation_config.cudagraph_specialize_lora
        ):
            previous = getattr(
                self,
                "_external_forced_dummy_lora_count",
                None,
            )
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
        cuda_graph_size = GPUModelRunner.capture_model(self)

    manager = self.encoder_cudagraph_manager
    if manager is not None and hasattr(self, "update_stream"):
        manager.update_stream = self.update_stream
    return cuda_graph_size


def assign_graph_param_functions(
    set_graph_params: Callable[..., Any],
    set_draft_graph_params: Callable[..., Any],
) -> None:
    """Refresh aliases that were imported before register_func ran."""
    imported_model_runner = sys.modules.get(
        "vllm_ascend.worker.model_runner_v1"
    )
    if imported_model_runner is None:
        return
    imported_model_runner.set_graph_params = set_graph_params
    imported_model_runner.set_draft_graph_params = set_draft_graph_params
