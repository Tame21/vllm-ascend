from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.__post_init__ = _dspark_post_init


def _max_num_new_slots_for_drafting(self) -> int:
    """Return the maximum extra query slots inserted by the drafter.

    The scheduler budget already contains one query slot for every decoding
    request. DFlash cannot reuse that slot for its bonus query: it inserts the
    bonus query followed by ``num_speculative_tokens`` mask queries. Therefore
    it needs K additional slots, rather than the K - 1 used by parallel EAGLE.

    This is a compatibility backport of vLLM #51256. Remove it after the
    minimum supported vLLM version includes that fix.
    """
    num_draft_tokens = self.num_speculative_tokens

    if self.use_dflash():
        return num_draft_tokens

    if self.parallel_drafting:
        if self.uses_draft_model():
            return num_draft_tokens
        return num_draft_tokens - 1

    if self.uses_draft_model():
        return 1

    return 0


SpeculativeConfig.max_num_new_slots_for_drafting = property(_max_num_new_slots_for_drafting)
