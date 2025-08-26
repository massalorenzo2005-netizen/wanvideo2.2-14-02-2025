import os, logging, torch
from .fp_mask_ctx import enable_mask_globally
from .fp_ids_resolver import (
    make_wan_ids_resolver_dynamic,
    make_full_frame_masks,
)

log = logging.getLogger("wanvw.fp")

def _left_right_masks(batch:int, H:int, W:int):
    left  = torch.zeros((batch,1,H,W)); left[:,:,:, :W//2]  = 1
    right = torch.zeros((batch,1,H,W)); right[:,:,:, W//2:] = 1
    return [left, right]

if os.getenv("WANVW_FP_MASK","0") == "1":
    try:
        B = int(os.getenv("WANVW_BATCH","1"))
        H = int(os.getenv("WANVW_IMAGE_H","720"))
        W = int(os.getenv("WANVW_IMAGE_W","720"))
        N = int(os.getenv("WANVW_NUM_PEOPLE","1"))
        driver_pp = int(os.getenv("WANVW_DRIVER_PER_PERSON","512"))
        mode = os.getenv("WANVW_MASK_MODE","strict")  # "strict" | "soft"
        layout = os.getenv("WANVW_MASK_LAYOUT","full") # "full" | "leftright"
        latent_down = int(os.getenv("WANVW_LATENT_DOWN","16"))  # typical = 16

        # Build masks (quick smoke test or full-frame per-person)
        if layout == "leftright" and N == 2:
            masks = _left_right_masks(B, H, W)
        else:
            masks = make_full_frame_masks(B, H, W, num_people=N)

        driver_lengths = [driver_pp] * N

        # Resolver: prefer dynamic rectangular mapping using H×W
        ids_resolver = make_wan_ids_resolver_dynamic(
            masks, driver_lengths,
            image_h=H, image_w=W, latent_down=latent_down
        )

        enable_mask_globally(
            ids_resolver, mode=mode,
            soft_bias=float(os.getenv("WANVW_SOFT_BIAS","4.0")),
            allow_global=True, debug=int(os.getenv("WANVW_MASK_DEBUG","2"))
        )
        log.info(f"[AUTO_MASK] Enabled: N={N}, driver_per_person={driver_pp}, layout={layout}, mode={mode}, out={W}x{H}, down={latent_down}")
    except Exception as e:
        log.exception(f"[AUTO_MASK] failed to enable global masking: {e}")
