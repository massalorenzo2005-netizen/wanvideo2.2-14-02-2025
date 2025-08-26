# File: fantasyportrait/auto_probe.py
# Auto-enable global probe when WANVW_FP_PROBE=1 is set.
import os
if os.getenv("WANVW_FP_PROBE", "0") == "1":
    from .fp_mask_ctx import enable_probe_globally
    enable_probe_globally(True)
