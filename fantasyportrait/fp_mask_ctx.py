from __future__ import annotations
import contextlib, contextvars, math
from typing import Callable, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F

# --- globals (lazily defined to avoid NameError on reloads) ---
try:
    _GLOBAL_IDS_RESOLVER
except NameError:
    _GLOBAL_IDS_RESOLVER = None

try:
    _GLOBAL_ALLOW_GLOBAL
except NameError:
    _GLOBAL_ALLOW_GLOBAL = True  # default

try:
    _GLOBAL_MODE
except NameError:
    _GLOBAL_MODE = "strict"

try:
    _GLOBAL_SOFT_BIAS
except NameError:
    _GLOBAL_SOFT_BIAS = 4.0

try:
    _GLOBAL_DEBUG
except NameError:
    _GLOBAL_DEBUG = 1

# factory holder
try:
    _IDS_RESOLVER_FACTORY
except NameError:
    _IDS_RESOLVER_FACTORY = None

# --- probe/mask runtime toggles ---
_ORIG_SDPA = None
_PATCHED = False
_PROBE_ALWAYS = False
_GLOBAL_SEEN: set = set()   # retained to preserve behavior (shape-dedup), but no logging uses it now
_MASK_ALWAYS = False

# --- lightweight session context ---
class _Ctx:
    __slots__ = ("probe","mask","mode","soft_bias","allow_global","debug",
                 "static_ids_q","static_ids_k","ids_resolver","_seen_shapes","_max_logs_per_kind")
    def __init__(self, probe, mask, mode, soft_bias, allow_global, debug,
                 static_ids_q, static_ids_k, ids_resolver, max_logs_per_kind=6):
        self.probe = bool(probe)
        self.mask = bool(mask)
        self.mode = str(mode)
        self.soft_bias = float(soft_bias)
        self.allow_global = bool(allow_global)
        self.debug = int(debug)
        self.static_ids_q = static_ids_q
        self.static_ids_k = static_ids_k
        self.ids_resolver = ids_resolver
        self._seen_shapes = set()
        self._max_logs_per_kind = int(max_logs_per_kind)

_CTX_VAR: contextvars.ContextVar[Optional[_Ctx]] = contextvars.ContextVar("fp_mask_ctx", default=None)

# --- resolver registration helpers (API preserved) ---
def set_ids_resolver(ids_resolver_fn):
    """
    Set a concrete ids_resolver:
      ids_resolver_fn(qshape, kshape, info) -> (ids_q[B,Tq], ids_k[B,Tk]) or (None, None)
    """
    global _GLOBAL_IDS_RESOLVER
    _GLOBAL_IDS_RESOLVER = ids_resolver_fn

def set_ids_resolver_factory(factory):
    """
    Register a factory that returns an ids_resolver callable.
    Instantiated immediately for back-compat.
    """
    global _IDS_RESOLVER_FACTORY, _GLOBAL_IDS_RESOLVER
    _IDS_RESOLVER_FACTORY = factory
    try:
        _GLOBAL_IDS_RESOLVER = factory()
    except Exception:
        _GLOBAL_IDS_RESOLVER = None

def get_ids_resolver_factory():
    return _IDS_RESOLVER_FACTORY

try:
    __all__
except NameError:
    __all__ = []
for _n in ("set_ids_resolver", "set_ids_resolver_factory", "get_ids_resolver_factory"):
    if _n not in __all__:
        __all__.append(_n)

# --- probe helpers (now no-ops, to keep behavior without printing) ---
def _log_probe(*args, **kwargs):
    return None

def _maybe_log_shape(tag: str, q, k, v, attn_mask, is_causal, ctx: Optional[_Ctx]):
    # Keep dedup logic to avoid unbounded growth of _GLOBAL_SEEN/_seen_shapes,
    # but don't emit logs.
    key = (q.shape[-2], k.shape[-2], q.shape[1], q.dtype, k.dtype, is_causal, bool(attn_mask is not None))
    if ctx is None:
        if not _PROBE_ALWAYS:
            return
        if key in _GLOBAL_SEEN or len(_GLOBAL_SEEN) >= 10:
            return
        _GLOBAL_SEEN.add(key)
        _log_probe(tag, q, k, v, attn_mask, is_causal)
        return
    if not ctx.probe:
        return
    if key in ctx._seen_shapes or len(ctx._seen_shapes) >= ctx._max_logs_per_kind:
        return
    ctx._seen_shapes.add(key)
    _log_probe(tag, q, k, v, attn_mask, is_causal)

# --- masking core ---
def _build_same_char_mask(ids_q: torch.Tensor, ids_k: torch.Tensor, B: int, H: int, Tq: int, Tk: int, allow_global: bool):
    iq = ids_q.view(B, 1, Tq, 1)
    ik = ids_k.view(B, 1, 1, Tk)
    same = (iq == ik)
    if allow_global:
        GLOBAL = torch.tensor(255, device=ids_q.device, dtype=ids_q.dtype)
        same = same | (iq == GLOBAL) | (ik == GLOBAL)
    return same.expand(B, H, Tq, Tk)

def _sdpa_wrapper(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    ctx = _CTX_VAR.get()
    tag = "sdpa"

    # If nothing to do, fall back immediately
    if ctx is None and not _PROBE_ALWAYS and not _MASK_ALWAYS:
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    # Optional probe (no-op)
    _maybe_log_shape(tag, q, k, v, attn_mask, is_causal, ctx)

    # Session masking path
    if ctx is not None and ctx.mask:
        ids_q = ctx.static_ids_q
        ids_k = ctx.static_ids_k
        if (ids_q is None or ids_k is None) and ctx.ids_resolver is not None:
            try:
                ids_q, ids_k = ctx.ids_resolver(
                    q.shape, k.shape,
                    {"is_causal": is_causal, "has_attn_mask": attn_mask is not None,
                     "dtype_q": str(q.dtype), "dtype_k": str(k.dtype)}
                )
            except Exception:
                ids_q, ids_k = None, None

        if ids_q is None or ids_k is None:
            return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        allow = _build_same_char_mask(
            ids_q.to(q.device), ids_k.to(q.device),
            q.shape[0], q.shape[1], q.shape[2], k.shape[-2],
            allow_global=ctx.allow_global
        )
        bias_val = -abs(float(ctx.soft_bias)) if ctx.mode == "soft" else max(torch.finfo(logits.dtype).min, -1e4)
        logits = logits.masked_fill(~allow, bias_val)
        if attn_mask is not None:
            logits = logits + attn_mask
        attn = torch.softmax(logits.float(), dim=-1).to(q.dtype)
        if dropout_p and (q.requires_grad or k.requires_grad or v.requires_grad):
            attn = torch.nn.functional.dropout(attn, p=dropout_p, training=True)
        return torch.matmul(attn, v)

    # Global masking path (outside a session)
    if ctx is None and _MASK_ALWAYS and _GLOBAL_IDS_RESOLVER is not None:
        try:
            ids_q, ids_k = _GLOBAL_IDS_RESOLVER(
                q.shape, k.shape,
                {"is_causal": is_causal, "has_attn_mask": attn_mask is not None,
                 "dtype_q": str(q.dtype), "dtype_k": str(k.dtype)}
            )
        except Exception:
            ids_q, ids_k = None, None

        if ids_q is not None and ids_k is not None:
            if scale is None:
                scale = 1.0 / math.sqrt(q.shape[-1])
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            allow = _build_same_char_mask(
                ids_q.to(q.device), ids_k.to(q.device),
                q.shape[0], q.shape[1], q.shape[2], k.shape[-2],
                allow_global=_GLOBAL_ALLOW_GLOBAL
            )
            bias_val = -abs(float(_GLOBAL_SOFT_BIAS)) if _GLOBAL_MODE == "soft" else max(torch.finfo(logits.dtype).min, -1e4)
            logits = logits.masked_fill(~allow, bias_val)
            if attn_mask is not None:
                logits = logits + attn_mask
            attn = torch.softmax(logits.float(), dim=-1).to(q.dtype)
            if dropout_p and (q.requires_grad or k.requires_grad or v.requires_grad):
                attn = torch.nn.functional.dropout(attn, p=dropout_p, training=True)
            return torch.matmul(attn, v)

    # Fallback: call original SDPA
    return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

def _ensure_patched():
    global _ORIG_SDPA, _PATCHED
    if _PATCHED:
        return
    _ORIG_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa_wrapper
    _PATCHED = True

@contextlib.contextmanager
def fp_mask_session(probe=True, mask=False, *, mode="strict", soft_bias=4.0, allow_global=True, debug=1,
                    static_ids_q=None, static_ids_k=None, ids_resolver=None, max_logs_per_kind=6):
    _ensure_patched()
    ctx = _Ctx(probe, mask, mode, soft_bias, allow_global, debug,
               static_ids_q, static_ids_k, ids_resolver, max_logs_per_kind)
    token = _CTX_VAR.set(ctx)
    try:
        yield
    finally:
        _CTX_VAR.reset(token)

# Kept for API compatibility; now a no-op
def enable_probe_logging(level: int = 20):
    return None

def enable_probe_globally(enable: bool = True, log_level: int = 20):
    global _PROBE_ALWAYS
    _ensure_patched()
    _PROBE_ALWAYS = bool(enable)

def enable_mask_globally(ids_resolver: Callable, *, mode: str = "strict", soft_bias: float = 4.0,
                         allow_global: bool = True, debug: int = 1, log_level: int = 20):
    """Enable masked attention everywhere without editing sampler code."""
    global _MASK_ALWAYS, _GLOBAL_IDS_RESOLVER, _GLOBAL_MODE, _GLOBAL_SOFT_BIAS, _GLOBAL_ALLOW_GLOBAL, _GLOBAL_DEBUG
    _ensure_patched()
    _MASK_ALWAYS = True
    _GLOBAL_IDS_RESOLVER = ids_resolver
    _GLOBAL_MODE = mode
    _GLOBAL_SOFT_BIAS = float(soft_bias)
    _GLOBAL_ALLOW_GLOBAL = bool(allow_global)
    _GLOBAL_DEBUG = int(debug)

def disable_mask_globally():
    global _MASK_ALWAYS, _GLOBAL_IDS_RESOLVER
    _MASK_ALWAYS = False
    _GLOBAL_IDS_RESOLVER = None