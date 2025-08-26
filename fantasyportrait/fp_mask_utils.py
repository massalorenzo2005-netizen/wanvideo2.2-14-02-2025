from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Optional

# Some call sites allow a "global" class that is never blocked.
# We use a high sentinel that won't collide with small #people IDs.
GLOBAL_ID: int = 255

# TorchDynamo can try to trace our utilities when called from inside SDPA.
# Keep these helpers out of graph capture to avoid aliasing/inplace issues.
try:
    import torch._dynamo as _dynamo
    _dynamo_disable = _dynamo.disable
except Exception:
    def _dynamo_disable(fn):  # no-op fallback for older PyTorch
        return fn


@_dynamo_disable
def downsample_masks_to_tokens(
    masks_bchw: List[torch.Tensor],   # each [B,1,H,W] (or [1,1,H,W] which we will repeat to B)
    token_h: int,
    token_w: int,
    *,
    global_id: int = GLOBAL_ID,
) -> torch.Tensor:
    """
    Produce query identity IDs per latent token.
    - Input: list of Bx1xHxW binary-ish masks, one per person (values in [0,1])
    - Output: LongTensor [B, token_h*token_w] with values in {0..N-1, global_id}
      Picks the argmax across persons at each token; if all ~0, emits GLOBAL_ID.
    Notes:
      * Uses out-of-place ops only (no .clamp_ or expand-inplace) to keep Dynamo happy.
      * Works for arbitrary rectangular token grids (token_h x token_w).
    """
    if not masks_bchw:
        raise ValueError("masks_bchw must contain at least one [B,1,H,W] tensor")

    # Normalize reference shape
    ref = masks_bchw[0]
    if ref.dim() == 3:
        ref = ref.unsqueeze(1)  # [B,1,H,W]
    if ref.dim() != 4 or ref.shape[1] != 1:
        raise ValueError(f"mask must be [B,1,H,W], got {tuple(ref.shape)}")
    B, _, H, W = ref.shape

    proc: List[torch.Tensor] = []
    for m in masks_bchw:
        if m.dim() == 3:
            m = m.unsqueeze(1)  # [B,1,H,W]
        if m.dim() != 4 or m.shape[1] != 1:
            raise ValueError(f"mask must be [B,1,H,W], got {tuple(m.shape)}")
        b = int(m.shape[0])
        if b != B:
            if b == 1:
                # repeat makes a real tensor; avoid expand (no shared storage)
                m = m.repeat(B, 1, 1, 1)
            else:
                m = m[:B]
        # ensure float32 and clamp without in-place
        m = torch.clamp(m.to(dtype=torch.float32), 0.0, 1.0)
        proc.append(m)

    # Stack to [B, N, 1, H, W] -> [B*N, 1, H, W] for one interpolate call
    M = torch.stack(proc, dim=1)
    BN = M.shape[0] * M.shape[1]
    M = M.view(BN, 1, H, W)
    ds = F.interpolate(M, size=(int(token_h), int(token_w)), mode="area")
    ds = torch.clamp(ds, 0.0, 1.0)
    ds = ds.view(B, -1, int(token_h), int(token_w))  # [B, N, Th, Tw]

    # Argmax over persons → ids; mark empty as GLOBAL_ID
    vals, idx = ds.max(dim=1)              # [B, Th, Tw]
    ids = idx.view(B, -1).to(torch.long)   # [B, Th*Tw]
    empty = (vals.view(B, -1) <= 1e-6)
    if empty.any():
        ids = ids.masked_fill(empty, int(global_id))
    return ids


@_dynamo_disable
def driver_token_ids_from_lengths(
    driver_lengths: List[int],          # e.g., [97, 97] for two FP drivers
    *,
    global_prefix: int = 0,
    global_suffix: int = 0,
    batch: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Produce key identity IDs for the driver/context sequence.
    Layout: [global_prefix][driver_0][driver_1]...[driver_N-1][global_suffix]
    - Each driver_i span is filled with ID=i
    - prefix/suffix filled with GLOBAL_ID
    Returns: LongTensor [B, total_len]
    """
    if any(l < 0 for l in driver_lengths):
        raise ValueError(f"driver_lengths must be >= 0: {driver_lengths}")

    total_drivers = sum(driver_lengths)
    total_len = int(global_prefix) + int(total_drivers) + int(global_suffix)

    # Build a single row then repeat to batch
    row = torch.full((total_len,), int(GLOBAL_ID), dtype=torch.long, device=device)

    cursor = int(global_prefix)
    for i, L in enumerate(driver_lengths):
        if L > 0:
            row[cursor:cursor + L] = int(i)
            cursor += int(L)
    # suffix is already GLOBAL_ID from initialization

    if batch > 1:
        row = row.unsqueeze(0).repeat(batch, 1)
    else:
        row = row.unsqueeze(0)  # [1, total_len]
    return row


__all__ = [
    "GLOBAL_ID",
    "downsample_masks_to_tokens",
    "driver_token_ids_from_lengths",
]