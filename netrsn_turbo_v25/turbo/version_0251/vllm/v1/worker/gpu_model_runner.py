# SPDX-License-Identifier: Apache-2.0

"""LoRA-aware warmup and ACL graph capture for ``GPUModelRunner``."""

from contextlib import contextmanager, nullcontext
from functools import wraps

import torch

from netrsn_turbo.turbo.version_0251.vllm_ascend.lora.punica_npu import (
    specialize_lora,
)


@contextmanager
def force_lora_count(self, count):
    previous = getattr(self, "_ascend_forced_dummy_lora_count", None)
    self._ascend_forced_dummy_lora_count = count
    try:
        yield
    finally:
        self._ascend_forced_dummy_lora_count = previous


def wrap_maybe_dummy_run_with_lora(original):
    @contextmanager
    @wraps(original)
    def maybe_dummy_run_with_lora(
        self,
        lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        remove_lora=True,
        num_active_loras=0,
        mapping_type=None,
    ):
        count = getattr(self, "_ascend_forced_dummy_lora_count", None)
        if count is not None:
            num_active_loras = count
        kwargs = {
            "remove_lora": remove_lora,
            "num_active_loras": num_active_loras,
        }
        if mapping_type is not None:
            kwargs["mapping_type"] = mapping_type
        with original(
            self,
            lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            **kwargs,
        ):
            yield

    return maybe_dummy_run_with_lora


def wrap_warmup_and_capture(original):
    @wraps(original)
    def warmup_and_capture(
        self,
        desc,
        cudagraph_runtime_mode,
        *args,
        **kwargs,
    ):
        context = (
            force_lora_count(self, desc.num_active_loras)
            if specialize_lora(self.vllm_config)
            else nullcontext()
        )
        with context:
            return original(
                self,
                desc,
                cudagraph_runtime_mode,
                *args,
                **kwargs,
            )

    return warmup_and_capture


def wrap_capture_model(original):
    @wraps(original)
    def capture_model(self):
        if specialize_lora(self.vllm_config):
            with force_lora_count(self, 0):
                self._dummy_run(
                    num_tokens=self.max_num_tokens,
                    is_profile=True,
                    num_active_loras=0,
                )
                self._dummy_run(
                    num_tokens=1,
                    is_profile=True,
                    num_active_loras=0,
                )
                torch.npu.synchronize()
        return original(self)

    return capture_model
