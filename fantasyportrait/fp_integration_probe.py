from __future__ import annotations
import logging
from .fp_mask_ctx import fp_mask_session, enable_probe_logging

log = logging.getLogger("wanvw.fp")

def run_with_probe(callable_fn, *args, **kwargs):
    """
    Wrap ANY existing sampling/inference call.
    Probe-only (no behavior change): prints [PROBE:sdpa] lines.
    """
    enable_probe_logging(logging.INFO)
    log.info("[FP-PROBE] starting probe session (mask=False, debug=1)")
    with fp_mask_session(probe=True, mask=False, debug=1):
        return callable_fn(*args, **kwargs)
