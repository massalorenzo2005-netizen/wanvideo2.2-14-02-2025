# Purpose: Merge multiple FantasyPortrait driver streams before WanVideoAddFantasyPortrait.

from __future__ import annotations
import logging
import torch

log = logging.getLogger("wanvw.fp")

class FantasyPortraitMerge2:
    """
    Merge two PORTRAIT_EMBEDS tensors by concatenating along the token-length axis.
    Expected input shape per tensor: [B, F, Tk, C]
      - B: batch (usually 1)
      - F: frames window the model is currently sampling
      - Tk: driver token length (e.g., 512 for FP)
      - C: channel dim (must match across inputs)

    If the two references have different frame counts (F), you can crop to the
    shortest or zero-pad the shorter to match.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "portrait1": ("PORTRAIT_EMBEDS",),
                "portrait2": ("PORTRAIT_EMBEDS",),
            },
            "optional": {
                "align_frames": ("BOOLEAN", {"default": True, "tooltip": "Align frame dimension (F) across inputs"}),
                "pad_shorter": ("BOOLEAN", {"default": False, "tooltip": "If True, zero-pad the shorter F; else crop to min(F1,F2)"}),
            },
        }

    RETURN_TYPES = ("PORTRAIT_EMBEDS",)
    RETURN_NAMES = ("portrait_embeds",)
    FUNCTION = "merge"
    CATEGORY = "WanVideoWrapper"

    def merge(self, portrait1, portrait2, align_frames=True, pad_shorter=False):
        t1: torch.Tensor = portrait1
        t2: torch.Tensor = portrait2

        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
            raise ValueError("Inputs must be tensors produced by FantasyPortraitFaceDetector.")

        if t1.dim() != 4 or t2.dim() != 4:
            raise ValueError(f"Expected 4D tensors [B,F,Tk,C]; got {t1.shape=} and {t2.shape=}")

        B1, F1, L1, C1 = t1.shape
        B2, F2, L2, C2 = t2.shape

        if B1 != B2:
            log.warning(f"[FP-MERGE] Batch sizes differ (B1={B1}, B2={B2}); using min.")
        B = min(B1, B2)

        if C1 != C2:
            raise ValueError(f"Channel dims must match: C1={C1}, C2={C2}")

        # Align frame dimension if requested
        if align_frames:
            if pad_shorter and F1 != F2:
                F = max(F1, F2)
                def _pad_to_F(x, F_target):
                    Bx, Fx, Lx, Cx = x.shape
                    if Fx == F_target:
                        return x
                    # zero pad at end on frame axis
                    pad_frames = F_target - Fx
                    pad_tensor = torch.zeros(Bx, pad_frames, Lx, Cx, dtype=x.dtype, device=x.device)
                    return torch.cat([x, pad_tensor], dim=1)

                t1 = _pad_to_F(t1, F)
                t2 = _pad_to_F(t2, F)
            else:
                F = min(F1, F2)
                if F1 != F or F2 != F:
                    log.info(f"[FP-MERGE] Cropping frames to F={F} (F1={F1}, F2={F2})")
                    t1 = t1[:, :F]
                    t2 = t2[:, :F]
        else:
            # no alignment: require equal F
            if F1 != F2:
                raise ValueError(f"align_frames=False but F mismatch (F1={F1}, F2={F2})")
            F = F1

        # Ensure same dtype/device for safe concat
        if t1.dtype != t2.dtype:
            log.info(f"[FP-MERGE] Casting dtypes to match: {t1.dtype=} {t2.dtype=}")
            common_dtype = torch.promote_types(t1.dtype, t2.dtype)
            t1 = t1.to(common_dtype)
            t2 = t2.to(common_dtype)
        if t1.device != t2.device:
            log.info(f"[FP-MERGE] Moving to common device: {t1.device=} {t2.device=}")
            if str(t1.device) != "cpu":
                t2 = t2.to(t1.device)
            else:
                t1 = t1.to(t2.device)

        merged = torch.cat([t1[:B], t2[:B]], dim=2)  # concat on Tk dimension

        return (merged,)

NODE_CLASS_MAPPINGS = {
    "FantasyPortraitMerge2": FantasyPortraitMerge2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FantasyPortraitMerge2": "FantasyPortrait Merge (2)"
}
