"""ACL graph parameter-store replacements for base/LoRA isolation."""

from vllm.compilation import monitor as compilation_monitor
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context

from vllm_ascend.compilation import acl_graph as acl_graph_module
from vllm_ascend.compilation.acl_graph import GraphParams


ORIGINAL_SET_GRAPH_PARAMS = acl_graph_module.set_graph_params
ORIGINAL_SET_DRAFT_GRAPH_PARAMS = acl_graph_module.set_draft_graph_params
ORIGINAL_SET_DRAFT_PREFILL_PARAMS = (
    acl_graph_module.set_draft_graph_prefill_params
)
ORIGINAL_WEAK_REF_WORKSPACES = acl_graph_module.weak_ref_workspaces


class GraphParamStore(dict):
    """Resolve integer token keys to the active full-graph descriptor."""

    def __init__(self, capture_sizes: list[int], default_factory):
        super().__init__((size, default_factory()) for size in capture_sizes)
        self.default_factory = default_factory

    @staticmethod
    def _resolve_key(key):
        if not isinstance(key, int):
            return key
        try:
            forward_context = get_forward_context()
        except (AssertionError, LookupError, RuntimeError):
            return key
        batch_descriptor = forward_context.batch_descriptor
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and batch_descriptor is not None
            and batch_descriptor.num_tokens == key
        ):
            return batch_descriptor
        return key

    @staticmethod
    def _raise_if_uncaptured(resolved_key) -> None:
        if (
            not isinstance(resolved_key, int)
            and not compilation_monitor.cudagraph_capturing_enabled
        ):
            raise RuntimeError(
                "ACL graph parameters were not captured for runtime batch "
                f"descriptor {resolved_key!r}. Check base/LoRA graph capture "
                "descriptor consistency."
            )

    def __contains__(self, key):
        return dict.__contains__(self, self._resolve_key(key))

    def __getitem__(self, key):
        resolved_key = self._resolve_key(key)
        if not dict.__contains__(self, resolved_key):
            self._raise_if_uncaptured(resolved_key)
            dict.__setitem__(self, resolved_key, self.default_factory())
        return dict.__getitem__(self, resolved_key)

    def __setitem__(self, key, value):
        resolved_key = self._resolve_key(key)
        if not dict.__contains__(self, resolved_key):
            self._raise_if_uncaptured(resolved_key)
        dict.__setitem__(self, resolved_key, value)

    def get(self, key, default=None):
        resolved_key = self._resolve_key(key)
        if not dict.__contains__(self, resolved_key):
            self._raise_if_uncaptured(resolved_key)
        return dict.get(self, resolved_key, default)


def make_graph_params(capture_sizes: list[int]) -> GraphParams:
    return GraphParams(
        GraphParamStore(capture_sizes, list),
        GraphParamStore(capture_sizes, lambda: None),
        GraphParamStore(capture_sizes, list),
        GraphParamStore(capture_sizes, list),
    )


def set_graph_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    acl_graph_module._graph_params = make_graph_params(capture_sizes)


def set_draft_graph_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    acl_graph_module._draft_graph_params = make_graph_params(capture_sizes)


def set_draft_graph_prefill_params(capture_sizes: list[int]) -> None:
    if acl_graph_module._draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph prefill parameters have already been set!")
    acl_graph_module._draft_graph_prefill_params = make_graph_params(
        capture_sizes
    )


def weak_ref_workspaces(params) -> None:
    if params is None:
        return
    for graph_key, workspace in list(dict.items(params.workspaces)):
        if workspace is None:
            continue
        dict.__setitem__(
            params.workspaces,
            graph_key,
            acl_graph_module.weak_ref_tensors(workspace),
        )
