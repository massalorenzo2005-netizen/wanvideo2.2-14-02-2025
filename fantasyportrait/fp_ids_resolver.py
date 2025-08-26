from __future__ import annotations
import os
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F

from .fp_mask_utils import (
    downsample_masks_to_tokens,
    driver_token_ids_from_lengths,
)

try:
    import torch._dynamo as _dynamo
    _dynamo_disable = _dynamo.disable
except Exception:
    def _dynamo_disable(fn): return fn  # no-op fallback


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


@_dynamo_disable
def make_full_frame_masks(
    layout: str,
    B: int,
    H: int,
    W: int,
    *,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Return list of [B,1,H,W] masks for quick tests."""
    device = device or torch.device("cpu")
    ly = layout.lower()
    if ly in ("leftright", "left-right", "lr"):
        left = torch.zeros((B, 1, H, W), device=device)
        right = torch.zeros((B, 1, H, W), device=device)
        left[:, :, :, : W // 2] = 1.0
        right[:, :, :, W // 2 :] = 1.0
        return [left, right]
    if ly in ("topbottom", "top-bottom", "tb"):
        top = torch.zeros((B, 1, H, W), device=device)
        bot = torch.zeros((B, 1, H, W), device=device)
        top[:, :, : H // 2, :] = 1.0
        bot[:, :, H // 2 :, :] = 1.0
        return [top, bot]
    raise ValueError(f"Unknown layout '{layout}'")


@_dynamo_disable
def make_wan720_ids_resolver(
    masks_bchw: List[torch.Tensor],     # each [B or 1, 1, H, W]
    driver_lengths: List[int],          # e.g., [97, 97]
    *,
    global_prefix: int = 0,
    global_suffix: int = 0,
    latent_down: Optional[int] = None,
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
):
    """
    Returns ids_resolver(qshape, kshape, info) -> (ids_q[B,Tq], ids_k[B,Tk]).
    Handles dynamic batch B (Wan’s frame chunking) and caches per-(B, Tq/Tk).
    Only activates when Tq == (H/down)*(W/down) and Tk == prefix+sum(driver_lengths)+suffix.
    """
    H = image_h or _env_int("WANVW_IMAGE_H", 720)
    W = image_w or _env_int("WANVW_IMAGE_W", 1280)
    down = latent_down or _env_int("WANVW_LATENT_DOWN", 16)
    Ht, Wt = H // down, W // down
    Tpf = Ht * Wt  # tokens per frame

    ids_q_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    ids_k_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    base_masks = [m.clone() for m in masks_bchw]

    def _replicate_masks_for_B(B: int, device: torch.device) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for m in base_masks:
            t = m
            if t.dim() == 3:
                t = t.unsqueeze(1)  # [B,1,H,W]
            if t.shape[0] != B:
                if t.shape[0] == 1:
                    t = t.repeat(B, 1, 1, 1)  # no shared storage
                else:
                    t = t[:B]
            out.append(t.to(device=device, dtype=torch.float32))
        return out

    @_dynamo_disable
    def _resolve(qshape: Tuple[int, int, int, int], kshape: Tuple[int, int, int, int], info: dict):
        B, _, Tq, _ = qshape
        Tk = kshape[2]

        expected_Tk = int(global_prefix) + sum(int(x) for x in driver_lengths) + int(global_suffix)
        if Tq != Tpf or Tk != expected_Tk:
            return (None, None)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        key_q = (int(B), int(Tq))
        ids_q = ids_q_cache.get(key_q)
        if ids_q is None:
            ms = _replicate_masks_for_B(int(B), device)
            ids_q = downsample_masks_to_tokens(ms, Ht, Wt)  # [B, Tpf]
            ids_q_cache[key_q] = ids_q
            print(f"[IDS] ids_q built: B={B} Ht={Ht} Wt={Wt} -> {tuple(ids_q.shape)}")

        key_k = (int(B), int(Tk))
        ids_k = ids_k_cache.get(key_k)
        if ids_k is None:
            ids_k = driver_token_ids_from_lengths(
                driver_lengths,
                global_prefix=global_prefix,
                global_suffix=global_suffix,
                batch=int(B),
                device=device,
            )
            ids_k_cache[key_k] = ids_k
            print(f"[IDS] ids_k built: B={B} Tk={Tk} -> {tuple(ids_k.shape)}")

        return (ids_q, ids_k)

    return _resolve


# --- Back-compat alias expected by some FantasyPortrait nodes ---
@_dynamo_disable
def make_wan_ids_resolver_dynamic(
    masks_bchw: List[torch.Tensor],
    driver_lengths: List[int],
    **kwargs,
):
    """
    Backward-compatible alias. Calls make_wan720_ids_resolver(...) internally.
    """
    return make_wan720_ids_resolver(masks_bchw, driver_lengths, **kwargs)


@_dynamo_disable
def make_env_rect_ids_resolver():
    H = _env_int("WANVW_IMAGE_H", 720)
    W = _env_int("WANVW_IMAGE_W", 1280)
    down = _env_int("WANVW_LATENT_DOWN", 16)
    layout = os.environ.get("WANVW_MASK_LAYOUT", "leftright")
    n_people = _env_int("WANVW_NUM_PEOPLE", 2)
    driver_per = _env_int("WANVW_DRIVER_PER_PERSON", 97)
    # dummy B for mask creation; real B comes from qshape during resolve
    masks = make_full_frame_masks(layout, 1, H, W)
    driver_lengths = [driver_per] * n_people
    return make_wan720_ids_resolver(
        masks, driver_lengths, latent_down=down, image_h=H, image_w=W
    )


__all__ = [
    "make_full_frame_masks",
    "make_wan720_ids_resolver",
    "make_wan_ids_resolver_dynamic",  # <— the alias FantasyPortrait nodes import
    "make_env_rect_ids_resolver",
]
