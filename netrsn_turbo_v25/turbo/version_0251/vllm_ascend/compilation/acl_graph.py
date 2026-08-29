# SPDX-License-Identifier: Apache-2.0

"""Base/LoRA ACL graph storage keyed by the full batch descriptor."""

from functools import wraps

from vllm.compilation import monitor
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_ascend.compilation import acl_graph

from netrsn_turbo.turbo.version_0251.vllm_ascend.lora.punica_npu import (
    specialize_lora,
)


class GraphParamStore(dict):
    """Resolve token-only accesses to the active FULL graph descriptor."""

    def __init__(self, values, default_factory):
        super().__init__(values)
        self.default_factory = default_factory

    @staticmethod
    def _resolve_key(key):
        if not isinstance(key, int) or not is_forward_context_available():
            return key
        context = get_forward_context()
        descriptor = context.batch_descriptor
        if (
            context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and descriptor is not None
            and descriptor.num_tokens == key
        ):
            return descriptor
        return key

    @staticmethod
    def _check_missing(key):
        if not isinstance(key, int) and not monitor.cudagraph_capturing_enabled:
            raise RuntimeError(f"ACL graph parameters were not captured for {key!r}")

    def __contains__(self, key):
        return dict.__contains__(self, self._resolve_key(key))

    def __getitem__(self, key):
        key = self._resolve_key(key)
        if not dict.__contains__(self, key):
            self._check_missing(key)
            dict.__setitem__(self, key, self.default_factory())
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        key = self._resolve_key(key)
        if not dict.__contains__(self, key):
            self._check_missing(key)
        dict.__setitem__(self, key, value)

    def get(self, key, default=None):
        key = self._resolve_key(key)
        if not dict.__contains__(self, key):
            self._check_missing(key)
        return dict.get(self, key, default)


class LoRAGraphParams(acl_graph.GraphParams):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            config = get_current_vllm_config()
        except AssertionError:
            return
        if specialize_lora(config):
            self.events = GraphParamStore(self.events, list)
            self.workspaces = GraphParamStore(
                self.workspaces,
                lambda: None,
            )
            self.handles = GraphParamStore(self.handles, list)
            self.attn_params = GraphParamStore(self.attn_params, list)


def wrap_weak_ref_workspaces(original):
    @wraps(original)
    def weak_ref_workspaces(params):
        if params is None or not isinstance(
            params.workspaces,
            GraphParamStore,
        ):
            return original(params)
        for key, workspace in list(dict.items(params.workspaces)):
            if workspace is not None:
                dict.__setitem__(
                    params.workspaces,
                    key,
                    acl_graph.weak_ref_tensors(workspace),
                )

    return weak_ref_workspaces
