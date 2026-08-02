"""Patched and new Punica functions for vLLM-Ascend v0.23."""

from collections.abc import Callable
from typing import Any

import torch

from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


@torch.library.custom_op(
    "external_patch::lora_bmm_expand_slice",
    mutates_args={"y"},
)
def lora_bmm_expand_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    indices: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    """Apply a packed LoRA-B slice without the fused rank/width constraint."""
    rows = x.shape[0]
    if indices.shape[0] != rows:
        indices = indices[:rows]

    safe_indices = indices.clamp(min=0).to(torch.long)
    gathered = lora_b_stacked[safe_indices, 0].to(y.dtype)
    active_rank = gathered.shape[-1]
    x_active = x[..., :active_rank].to(y.dtype).contiguous()
    if x_active.shape[0] == 0 or gathered.shape[1] == 0:
        return

    delta = (x_active.unsqueeze(1) * gathered).sum(dim=-1)
    delta = torch.where(
        (indices >= 0).unsqueeze(-1),
        delta,
        torch.zeros_like(delta),
    )
    y_slice = y.narrow(1, y_offset, y_slice_size)
    if y_slice.shape[0] != delta.shape[0]:
        y_slice = y_slice[: delta.shape[0]]
    if add_inputs:
        y_slice.add_(delta)
    else:
        y_slice.copy_(delta)


@lora_bmm_expand_slice.register_fake
def lora_bmm_expand_slice_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    indices: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    return None


# These methods do not exist in v0.23 and are attached by direct assignment.
def enable_compatible_lora_bmm_expand_slice(self) -> None:
    self._force_compatible_lora_expand_slice = True


def requires_compatible_lora_expand_slice(
    self,
    x: torch.Tensor,
    y_slice_size: int,
) -> bool:
    return (
        getattr(self, "_force_compatible_lora_expand_slice", False)
        or x.shape[-1] > y_slice_size
    )


def compatible_lora_bmm_expand_slice(
    self,
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    lora_bmm_expand_slice(
        y,
        x,
        lora_b_stacked,
        self._get_token_lora_indices(x),
        y_offset,
        y_slice_size,
        add_inputs,
    )


ORIGINAL_EXPAND_SLICE_PREFILL = PunicaWrapperNPU._expand_slice_prefill
ORIGINAL_EXPAND_SLICE_DECODE = PunicaWrapperNPU._expand_slice_decode
ORIGINAL_ADD_SHRINK = PunicaWrapperNPU.add_shrink
ORIGINAL_ADD_EXPAND = PunicaWrapperNPU.add_expand
ORIGINAL_ADD_LORA_EMBEDDING = PunicaWrapperNPU.add_lora_embedding
ORIGINAL_ADD_LORA_LINEAR = PunicaWrapperNPU.add_lora_linear
ORIGINAL_ADD_LORA_LOGITS = PunicaWrapperNPU.add_lora_logits


def expand_slice_prefill(
    self,
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    if self.no_lora:
        return
    if self._requires_compatible_lora_expand_slice(x, y_slice_size):
        self._compatible_lora_bmm_expand_slice(
            y,
            x,
            lora_b_stacked,
            y_offset,
            y_slice_size,
            add_inputs,
        )
        return
    ORIGINAL_EXPAND_SLICE_PREFILL(
        self,
        y,
        x,
        lora_b_stacked,
        y_offset,
        y_slice_size,
        add_inputs,
    )


def expand_slice_decode(
    self,
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
) -> None:
    if self.no_lora:
        return
    if self._requires_compatible_lora_expand_slice(x, y_slice_size):
        self._compatible_lora_bmm_expand_slice(
            y,
            x,
            lora_b_stacked,
            y_offset,
            y_slice_size,
            add_inputs,
        )
        return
    ORIGINAL_EXPAND_SLICE_DECODE(
        self,
        y,
        x,
        lora_b_stacked,
        y_offset,
        y_slice_size,
        add_inputs,
    )


def _run_unless_no_lora(
    self: Any,
    original: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if self.no_lora:
        return None
    return original(self, *args, **kwargs)


def add_shrink(self, *args, **kwargs):
    return _run_unless_no_lora(self, ORIGINAL_ADD_SHRINK, *args, **kwargs)


def add_expand(self, *args, **kwargs):
    return _run_unless_no_lora(self, ORIGINAL_ADD_EXPAND, *args, **kwargs)


def add_lora_embedding(self, *args, **kwargs):
    return _run_unless_no_lora(
        self,
        ORIGINAL_ADD_LORA_EMBEDDING,
        *args,
        **kwargs,
    )


def add_lora_linear(self, *args, **kwargs):
    return _run_unless_no_lora(
        self,
        ORIGINAL_ADD_LORA_LINEAR,
        *args,
        **kwargs,
    )


def add_lora_logits(self, *args, **kwargs):
    return _run_unless_no_lora(
        self,
        ORIGINAL_ADD_LORA_LOGITS,
        *args,
        **kwargs,
    )
