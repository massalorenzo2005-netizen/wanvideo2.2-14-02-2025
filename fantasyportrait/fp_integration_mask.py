from __future__ import annotations
import logging
from .fp_mask_ctx import fp_mask_session, enable_probe_logging

log = logging.getLogger("wanvw.fp")

def run_with_mask(callable_fn, ids_resolver, *args, **kwargs):
    """
    Wrap ANY existing sampling call and enable masked attention
    (plus probe logs for visibility).
    """
    enable_probe_logging(logging.INFO)
    log.info("[FP-MASK] starting masked session (probe=True, mask=True, strict)")
    with fp_mask_session(
        probe=True, mask=True, mode="strict", soft_bias=4.0,
        allow_global=True, debug=2, ids_resolver=ids_resolver
    ):
        return callable_fn(*args, **kwargs)
