from __future__ import annotations

# Comfy typing helpers
class Sockets:
    pass

# WanVideo FP masking utilities
from .fp_mask_ctx import (
    enable_probe_globally,
    enable_mask_globally,
    set_ids_resolver_factory,
)
from .fp_ids_resolver import (
    make_wan720_ids_resolver,
    make_full_frame_masks,
)

# ---------- Node: FPMaskAutoConfig ----------
class FPMaskAutoConfig:
    """
    Configure FantasyPortrait masking/probe WITHOUT env vars.
    Place this node early and pass its output (portrait_model passthrough)
    into WanVideoAddFantasyPortrait so it executes before sampling.
    """

    CATEGORY = "FantasyPortrait"
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        # NOTE: portrait_model is a passthrough to enforce execution order
        return {
            "required": {
                "portrait_model": ("FANTASYPORTRAITMODEL",),
                "enable_probe": (["disabled", "enabled"], {"default": "enabled"}),
                "enable_mask": (["disabled", "enabled"], {"default": "enabled"}),
                "mask_mode": (["strict", "soft"], {"default": "strict"}),
                "mask_layout": (["leftright", "topbottom"], {"default": "leftright"}),
                "num_people": ("INT", {"default": 2, "min": 1, "max": 8}),
                "driver_per_person": ("INT", {"default": 97, "min": 1, "max": 1024}),
                "image_w": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 1}),
                "image_h": ("INT", {"default": 720,  "min": 64, "max": 8192, "step": 1}),
                "latent_down": ("INT", {"default": 16, "min": 4, "max": 64, "step": 1}),
                "allow_global": (["false", "true"], {"default": "true"}),
                "soft_bias": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "debug": ("INT", {"default": 2, "min": 0, "max": 3}),
            },
        }

    RETURN_TYPES = ("FANTASYPORTRAITMODEL",)
    RETURN_NAMES = ("portrait_model",)
    OUTPUT_NODE = False

    def apply(
        self,
        portrait_model,
        enable_probe,
        enable_mask,
        mask_mode,
        mask_layout,
        num_people,
        driver_per_person,
        image_w,
        image_h,
        latent_down,
        allow_global,
        soft_bias,
        debug,
    ):
        # 1) Enable/disable probe + mask
        probe_on = (enable_probe == "enabled")
        mask_on  = (enable_mask == "enabled")
        allow_g  = (allow_global == "true")

        enable_probe_globally(probe_on, log_level="INFO")
        if mask_on:
            enable_mask_globally(True, mode=mask_mode, soft_bias=float(soft_bias), allow_global=allow_g, debug=int(debug))
        else:
            enable_mask_globally(False, mode=mask_mode, soft_bias=float(soft_bias), allow_global=allow_g, debug=int(debug))

        # 2) Build simple rectangular masks (batch will be replicated per-chunk later)
        masks = make_full_frame_masks(mask_layout, B=1, H=image_h, W=image_w)

        # 3) Driver lengths for N subjects
        driver_lengths = [int(driver_per_person)] * int(num_people)

        # 4) Install resolver factory that uses these settings (dynamic-B safe)
        def _factory():
            return make_wan720_ids_resolver(
                masks,
                driver_lengths,
                latent_down=int(latent_down),
                image_h=int(image_h),
                image_w=int(image_w),
            )
        set_ids_resolver_factory(_factory)
        
        # Return passthrough portrait model so the graph can wire it into WanVideoAddFantasyPortrait
        return (portrait_model,)


# ---- Node registration
NODE_CLASS_MAPPINGS = {
    "FPMaskAutoConfig": FPMaskAutoConfig,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FPMaskAutoConfig": "FantasyPortrait Mask AutoConfig",
}
