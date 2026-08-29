# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 dense LoRA behavior for ``PunicaWrapperNPU``."""

from functools import wraps

import torch
from vllm.config import CUDAGraphMode


def patch_applies(config) -> bool:
    return bool(
        config.lora_config is not None
        and getattr(
            config.model_config.hf_text_config,
            "model_type",
            None,
        )
        == "qwen3_5_text"
    )


def specialize_lora(config) -> bool:
    return bool(
        patch_applies(config)
        and not config.model_config.enforce_eager
        and config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and config.compilation_config.cudagraph_specialize_lora
    )


def validate_config(config) -> None:
    if not patch_applies(config):
        return
    if config.use_v2_model_runner:
        raise ValueError("The Qwen3.5 LoRA patch supports model runner v1 only")
    if config.lora_config.max_lora_rank > 64:
        raise ValueError(
            "The Qwen3.5 LoRA patch currently supports max_lora_rank <= 64 only"
        )
    speculative_config = config.speculative_config
    if speculative_config is not None and speculative_config.method != "mtp":
        raise ValueError(
            "The Qwen3.5 LoRA patch supports MTP speculative decoding only"
        )
    parallel = config.parallel_config
    if (
        parallel.prefill_context_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
    ):
        raise ValueError("The Qwen3.5 LoRA patch does not support context parallelism")
    if parallel.enable_dbo or parallel.ubatch_size > 1:
        raise ValueError("The Qwen3.5 LoRA patch does not support microbatching")
    if config.model_config.enable_sleep_mode:
        raise ValueError(
            "The Qwen3.5 LoRA patch has not been validated with sleep mode"
        )


@torch.library.custom_op(
    "vllm_ascend::qwen3_5_lora_expand_slice",
    mutates_args={"y"},
)
def _lora_expand_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    offset: int,
    width: int,
    add_inputs: bool,
) -> None:
    rows_per_chunk = 128
    rank = weights.shape[-1]
    if rank > x.shape[-1] or weights.shape[-2] != width:
        raise ValueError("Incompatible Qwen3.5 LoRA-B slice shape")
    for start in range(0, x.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, x.shape[0])
        token_indices = indices[start:end]
        selected = weights[token_indices.clamp(min=0).long(), 0].to(y.dtype)
        inputs = x[start:end, :rank].to(y.dtype)
        delta = (inputs.unsqueeze(1) * selected).sum(dim=-1)
        delta = torch.where(
            (token_indices >= 0).unsqueeze(-1),
            delta,
            torch.zeros_like(delta),
        )
        target = y[start:end, offset : offset + width]
        if add_inputs:
            target.add_(delta)
        else:
            target.copy_(delta)


@_lora_expand_slice.register_fake
def _lora_expand_slice_fake(
    y,
    x,
    weights,
    indices,
    offset,
    width,
    add_inputs,
) -> None:
    return None


def wrap_expand_slice(original):
    @wraps(original)
    def expand(
        self,
        y,
        x,
        w_t_all,
        y_offset,
        y_slice_size,
        add_inputs,
    ):
        if not getattr(self, "_ascend_qwen3_5_lora", False):
            return original(
                self,
                y,
                x,
                w_t_all,
                y_offset,
                y_slice_size,
                add_inputs,
            )
        if getattr(self, "_ascend_specialize_lora", False) and self.no_lora:
            return None
        _lora_expand_slice(
            y,
            x,
            w_t_all,
            self._get_token_lora_indices(x),
            y_offset,
            y_slice_size,
            add_inputs,
        )

    return expand


def wrap_update_metadata(original):
    @wraps(original)
    def update_metadata(self, mapping, *args, **kwargs):
        result = original(self, mapping, *args, **kwargs)
        if getattr(self, "_ascend_qwen3_5_lora", False):
            self.no_lora = not any(
                adapter_id > 0 for adapter_id in mapping.index_mapping
            )
        return result

    return update_metadata


def no_lora_guard(original):
    @wraps(original)
    def guarded(self, *args, **kwargs):
        if getattr(self, "_ascend_specialize_lora", False) and self.no_lora:
            return None
        return original(self, *args, **kwargs)

    return guarded
