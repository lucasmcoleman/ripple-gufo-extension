#!/usr/bin/env python3
"""CPU reference for the Ripple-65B DFlash2 drafter (stage 0 of the drafter design).

Built the way upstream gufo's tools/qwen27b/dflash_reference.py does it: the
pinned, unmodified upstream model definition (z-lab/dflash dflash/model.py at
07ebd93, sha256 below) is loaded and given the drafter GGUF's weights,
dequantized to float32. The embedding rows of the block and the output head
come from the target GGUF. Nothing here is a second implementation of the
drafter: the forward pass, the dynamic convolutions and the candidate selector
are the upstream classes. What this file adds is the weight binding, a
per-layer context cache (equivalent to the upstream persistent draft cache,
checked below), the variants, and the acceptance bookkeeping.

Variants (docs/DRAFTER-STAGE0.md):
  R   the recipe: no rotary on the layers listed in `rope_pattern` as False
      (layer 4), block of 16 rows [anchor, MASK x 15], raw embedding rows.
  Rs  R with the embedding rows multiplied by the target's embedding scale.
  L   llama.cpp-like: rotary on every layer, block of d+1 rows for d drafts.

SAFETY. The upstream file is downloaded code. It was read before first use
(imports: time, types, typing, torch, transformers; no network, file system,
subprocess, eval/exec or environment access), its sha256 is checked against the
pin before it is executed, and this script refuses to run it unless it was
started as

  env -i PATH=/usr/bin:/bin HOME=<scratch> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      <venv>/bin/python -I tools/drafter_oracle.py ...

(isolated interpreter, offline, no tokens in the environment, CPU torch only,
script kept outside the directory of the downloaded file).

Commands:
  selfcheck  --draft D.gguf --target T.gguf [--card]    the checks of stage 0 step 2
  run        --draft D.gguf --target T.gguf (--dump DIR | --sets FILE --dumps DIR) --out DIR
             [--variants R,Rs,L] [--l-d 2,3,..] [--batch 32] [--stride 1]
             reads ricursive_dump_taps directories, writes per-anchor drafts and accepted
             counts for every draft length (L: the lengths in --l-d)
  timing     --draft D.gguf --target T.gguf [--ctx N] [--anchors M]
             one-anchor and batched timing on random inputs, for the extrapolation
  stages     --draft D.gguf --target T.gguf --dump DIR --out FILE.bin
             (--anchors a,b,c | --first A --last B [--stride S])
             variant R, one anchor per forward (the upstream mask path), with the named intermediate
             stages of every anchor written to one binary file for the C++ drafter test (docs/DRAFTER-S1.md)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM_SHA256 = "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e"
# The same pin as /server/programming/gufo-ripple/gufo/tools/qwen27b/dflash_reference.py:21-22.
UPSTREAM_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"
DEFAULT_UPSTREAM = "/server/staging/ripple-gufo/ref/dflash2/zlab-07ebd93-dflash_model.py"
CARD_PARAMS = 4_037_752_064  # model card
CARD_TENSORS = 81  # GGUF tensor count (15 per layer x 5 + fc, enc norm, final norm, 3 selector tensors)
MAX_D = 15


# --------------------------------------------------------------------------------------
# dequantization (own implementation, checked against gguf-py)
# --------------------------------------------------------------------------------------
QK_K = 256
# gguf type id -> (name, elements per block, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144),
         13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2)}
IQ4_KVALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16).astype(np.float32)


def _scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 12 packed bytes of a K-quant super-block -> 8 scales and 8 mins (ggml get_scale_min_k4)."""
    s = scales.reshape(-1, 3, 4)
    d, m, md = s[:, 0], s[:, 1], s[:, 2]
    sc = np.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc, mn


def _dequant_blocks(kind: str, blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    if kind == "Q8_0":
        d = _f16(blocks[:, :2])
        return np.ascontiguousarray(blocks[:, 2:]).view(np.int8).astype(np.float32) * d
    if kind in ("Q4_K", "Q5_K"):
        d = _f16(blocks[:, 0:2])
        dmin = _f16(blocks[:, 2:4])
        sc, mn = _scale_min_k4(np.ascontiguousarray(blocks[:, 4:16]))
        d = (d * sc.astype(np.float32)).reshape(n, 8, 1)
        dm = (dmin * mn.astype(np.float32)).reshape(n, 8, 1)
        if kind == "Q4_K":
            qs = blocks[:, 16:144].reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
            q = (qs & np.uint8(0x0F)).reshape(n, 8, 32).astype(np.float32)
        else:
            qh = blocks[:, 16:48].reshape(n, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)
            ql = blocks[:, 48:176].reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
            ql = (ql & np.uint8(0x0F)).reshape(n, 8, 32)
            qh = (qh & np.uint8(0x01)).reshape(n, 8, 32)
            q = (ql | (qh << np.uint8(4))).astype(np.float32)
        return (d * q - dm).reshape(n, QK_K)
    if kind == "Q6_K":
        ql = blocks[:, 0:128]
        qh = blocks[:, 128:192]
        scales = np.ascontiguousarray(blocks[:, 192:208]).view(np.int8).astype(np.float32)
        d = (_f16(blocks[:, 208:210]) * scales).reshape(n, 16, 1)
        ql = ql.reshape(n, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
        ql = (ql & np.uint8(0x0F)).reshape(n, 8, 32)
        qh = qh.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)
        qh = (qh & np.uint8(0x03)).reshape(n, 8, 32)
        q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
        q = q.reshape(n, 16, 16).astype(np.float32)
        return (d * q).reshape(n, QK_K)
    if kind == "IQ4_XS":
        d = _f16(blocks[:, 0:2])
        sh = np.ascontiguousarray(blocks[:, 2:4]).view(np.uint16)
        sl = blocks[:, 4:8]
        lo = np.stack([sl & np.uint8(0x0F), sl >> np.uint8(4)], axis=-1).reshape(n, 8)
        hi = ((sh >> (2 * np.arange(8, dtype=np.uint16))) & np.uint16(3)).astype(np.uint8)
        scales = (lo | (hi << np.uint8(4))).astype(np.int8) - np.int8(32)
        dl = (d * scales.astype(np.float32)).reshape(n, 8, 1)
        qs = blocks[:, 8:136].reshape(n, 8, 1, 16) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
        qs = (qs & np.uint8(0x0F)).reshape(n, 8, 32)
        return (dl * IQ4_KVALUES[qs].astype(np.float32)).reshape(n, QK_K)
    raise NotImplementedError(kind)


def dequantize(gtype: int, raw: np.ndarray, n_elements: int) -> np.ndarray:
    """raw: contiguous uint8 bytes of whole blocks -> float32 [n_elements]."""
    if gtype not in TYPES:
        raise NotImplementedError(f"ggml type {gtype}")
    kind, per, size = TYPES[gtype]
    raw = np.ascontiguousarray(raw).reshape(-1)
    if raw.dtype != np.uint8:
        raw = raw.view(np.uint8)
    if raw.size != n_elements // per * size or n_elements % per:
        raise ValueError(f"{kind}: {raw.size} bytes do not hold {n_elements} elements")
    if kind == "F32":
        return raw.view(np.float32).copy()
    if kind == "F16":
        return raw.view(np.float16).astype(np.float32)
    if kind == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    out = np.empty(n_elements, dtype=np.float32)
    blocks = raw.reshape(-1, size)
    step = 1 << 20  # blocks per chunk: bounds temporary memory
    for lo in range(0, blocks.shape[0], step):
        part = _dequant_blocks(kind, blocks[lo:lo + step])
        out[lo * per:(lo + part.shape[0]) * per] = part.reshape(-1)
    return out


class Gguf:
    """Read-only view of a GGUF file: metadata and float32 tensors (numpy order shapes)."""

    def __init__(self, path: Path):
        import gguf  # PyPI package in the oracle venv
        self.path = Path(path)
        self.reader = gguf.GGUFReader(str(path))
        self.tensors = {t.name: t for t in self.reader.tensors}

    def meta(self, key: str, default=None):
        f = self.reader.fields.get(key)
        if f is None:
            return default
        v = f.contents()
        return v.item() if isinstance(v, np.generic) else v

    @staticmethod
    def shape(t) -> tuple[int, ...]:
        return tuple(int(x) for x in reversed(t.shape))  # ggml ne order -> numpy order

    def raw(self, name: str) -> np.ndarray:
        return np.ascontiguousarray(self.tensors[name].data).reshape(-1).view(np.uint8)

    def read(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        shape = self.shape(t)
        return dequantize(int(t.tensor_type), self.raw(name), int(np.prod(shape))).reshape(shape)

    def rows(self, name: str, indices: list[int]) -> np.ndarray:
        t = self.tensors[name]
        shape = self.shape(t)
        if len(shape) != 2:
            raise ValueError(f"{name} is not a matrix")
        kind, per, size = TYPES[int(t.tensor_type)]
        row_bytes = shape[1] // per * size
        raw = self.raw(name)
        out = np.empty((len(indices), shape[1]), dtype=np.float32)
        for i, r in enumerate(indices):
            if not 0 <= r < shape[0]:
                raise ValueError(f"row {r} outside {name} ({shape[0]} rows)")
            out[i] = dequantize(int(t.tensor_type), raw[r * row_bytes:(r + 1) * row_bytes], shape[1])
        return out


# --------------------------------------------------------------------------------------
# environment guard and upstream loading
# --------------------------------------------------------------------------------------
def require_safe_environment() -> None:
    problems = []
    if not sys.flags.isolated:
        problems.append("python was not started with -I")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        problems.append("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    risky = [k for k in os.environ if any(w in k.upper() for w in ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL"))]
    if risky:
        problems.append("the environment holds credential-like variables: " + ", ".join(sorted(risky)))
    if problems:
        sys.exit("drafter_oracle: refusing to run downloaded code: " + "; ".join(problems) +
                 "\n  use: env -i PATH=/usr/bin:/bin HOME=<scratch dir> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                 "<venv>/bin/python -I tools/drafter_oracle.py ...")


def load_upstream(path: Path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        sys.exit(f"drafter_oracle: {path} has sha256 {digest}, not the pinned {UPSTREAM_SHA256}")
    require_safe_environment()
    spec = importlib.util.spec_from_file_location("dflash_upstream", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# the drafter
# --------------------------------------------------------------------------------------
def weight_name(name: str) -> str:
    """Upstream parameter name -> GGUF tensor name (the mapping gufo's dflash_reference.py uses)."""
    shared = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "enc.output_norm.weight",
        "norm.weight": "output_norm.weight",
        "candidate_selector.predecessor_codebook.weight": "selector_predecessor.weight",
        "candidate_selector.successor_codebook.weight": "selector_successor.weight",
        "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    }
    if name in shared:
        return shared[name]
    _, index, suffix = name.split(".", 2)
    suffixes = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "attention_conv.base_kernel": "attn_conv_base",
        "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
        "mlp_conv.base_kernel": "ffn_conv_base",
        "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
    }
    return f"blk.{index}.{suffixes[suffix]}"


def config_dict(g: Gguf, target: Gguf) -> dict:
    """The upstream Qwen3Config fields, from the drafter GGUF metadata (cross-checked with the card's
    config.json for the real file: see selfcheck)."""
    n = int(g.meta("dflash.block_count"))
    sliding = [bool(x) for x in g.meta("dflash.attention.sliding_window_pattern")]
    rope = [bool(x) for x in g.meta("dflash.attention.rope_pattern")]
    # float32 round trip of 1e-6 in the GGUF; the card says 1e-06.
    eps = float(f"{float(g.meta('dflash.attention.layer_norm_rms_epsilon')):.6g}")
    vocab = int(g.tensors["selector_predecessor.weight"].shape[1])
    theta = float(g.meta("dflash.rope.freq_base"))
    return {
        "vocab_size": vocab,
        "hidden_size": int(g.meta("dflash.embedding_length")),
        "intermediate_size": int(g.meta("dflash.feed_forward_length")),
        "num_hidden_layers": n,
        "num_attention_heads": int(g.meta("dflash.attention.head_count")),
        "num_key_value_heads": int(g.meta("dflash.attention.head_count_kv")),
        "head_dim": int(g.meta("dflash.attention.key_length")),
        "rms_norm_eps": eps,
        "max_position_embeddings": int(g.meta("dflash.context_length")),
        "sliding_window": int(g.meta("dflash.attention.sliding_window")),
        "use_sliding_window": True,
        "layer_types": ["sliding_attention" if s else "full_attention" for s in sliding],
        "no_rope_layers": [i for i, r in enumerate(rope) if not r],
        "is_causal": not (g.meta("dflash.attention.causal") is False),
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": theta, "rope_type": "default"},
        "rope_theta": theta,
        "num_target_layers": int(target.meta("ricursive.block_count")),
        "dflash_config": {
            "block_size": int(g.meta("dflash.block_size")),
            "conv_kernel_size": int(g.meta("dflash.conv_kernel_size")),
            "conv_group_size": int(g.meta("dflash.conv_group_size")),
            "mask_token_id": int(g.meta("tokenizer.ggml.mask_token_id")),
            "selector_rank": int(g.meta("dflash.selector_rank")),
            "selector_top_k": int(g.meta("dflash.selector_top_k")),
            # GGUF stores layer INPUT numbering (HF list + 1); the tap is the output of block i.
            "target_layer_ids": [int(x) - 1 for x in g.meta("dflash.target_layers")],
        },
    }


class CtxCache:
    """Per-layer context K/V of every position (rotated for its layer), indexed by absolute position.

    Stands in for the upstream persistent draft cache (past_key_values): `update` returns the context rows
    [lo, hi) followed by the block's own rows, and never stores the block rows, which upstream crops away
    after every step. `selfcheck` compares it with the uncached upstream forward.
    """

    def __init__(self, k: list, v: list):
        self.k, self.v = k, v
        self.lo = self.hi = 0

    def set_range(self, lo: int, hi: int) -> None:
        self.lo, self.hi = lo, hi

    def update(self, key, value, layer_idx, cache_kwargs=None):
        import torch
        kc = self.k[layer_idx][:, :, self.lo:self.hi].expand(key.shape[0], -1, -1, -1)
        vc = self.v[layer_idx][:, :, self.lo:self.hi].expand(key.shape[0], -1, -1, -1)
        return torch.cat([kc, key], dim=-2), torch.cat([vc, value], dim=-2)


VARIANTS = {
    "R": {"nope": True, "scale": False, "block": "fixed"},
    "Rs": {"nope": True, "scale": True, "block": "fixed"},
    "L": {"nope": False, "scale": False, "block": "d+1"},
}


class Drafter:
    def __init__(self, upstream, draft: Gguf, target: Gguf, threads: int | None = None):
        import torch
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("highest")
        self.up, self.draft, self.target = upstream, draft, target
        cfgd = config_dict(draft, target)
        self.cfgd = cfgd
        cfg = upstream.Qwen3Config(**cfgd)
        cfg._attn_implementation = "sdpa"
        self.cfg = cfg
        self.hidden = cfg.hidden_size
        self.tap_ids = list(cfg.dflash_config["target_layer_ids"])
        self.block = int(cfg.dflash_config["block_size"])
        self.conv_kernel = int(cfg.dflash_config["conv_kernel_size"])
        self.mask_id = int(cfg.dflash_config["mask_token_id"])
        self.window = int(cfg.sliding_window)
        self.context_window = int(draft.meta("dflash.attention.context_window", 0) or 0)
        self.no_rope = set(cfg.no_rope_layers)
        if self.hidden != int(target.meta("ricursive.embedding_length")):
            raise ValueError("drafter and target hidden sizes differ")
        if cfg.vocab_size != int(target.meta("ricursive.vocab_size")):
            raise ValueError("drafter and target vocabularies differ")
        with torch.device("meta"):
            model = upstream.DFlash2DraftModel(cfg)
        state, mapped = {}, set()
        self.param_count = 0
        for name, p in model.state_dict().items():
            gname = weight_name(name)
            value = torch.from_numpy(draft.read(gname)).reshape(p.shape)
            state[name] = value
            mapped.add(gname)
            self.param_count += value.numel()
        extra = set(draft.tensors) - mapped
        if extra:
            raise ValueError(f"GGUF tensors the model does not use: {sorted(extra)}")
        self.tensor_count = len(state)
        model.load_state_dict(state, strict=True, assign=True)
        model.rotary_emb = upstream.Qwen3RotaryEmbedding(cfg)
        model.eval()
        self.model = model
        head = torch.nn.Linear(self.hidden, cfg.vocab_size, bias=False, device="meta")
        head.weight = torch.nn.Parameter(torch.from_numpy(target.read("output.weight")), requires_grad=False)
        self.head = head
        self.embed_scale = float(target.meta("ricursive.embedding_scale"))
        self._rows: dict[int, np.ndarray] = {}
        # rope control: upstream applies rotary on every layer; the hook below skips the layers in `nope`.
        self.active = {"layer": None, "nope": set()}
        original = upstream.apply_rotary_pos_emb
        active = self.active

        def rope(q, k, cos, sin, unsqueeze_dim=1):
            if active["layer"] in active["nope"]:
                return q, k
            return original(q, k, cos, sin, unsqueeze_dim)

        upstream.apply_rotary_pos_emb = rope
        self.layer_masks = None  # set by draft_blocks: per-layer boolean masks for a batch of anchors

        def pre(module, args, kwargs, index):
            active["layer"] = index
            if self.layer_masks is not None:
                kwargs["attention_mask"] = self.layer_masks[index]
                return args, kwargs
            return None

        for index, layer in enumerate(model.layers):
            layer.register_forward_pre_hook(lambda m, a, kw, index=index: pre(m, a, kw, index), with_kwargs=True)

    # ---- inputs -------------------------------------------------------------------
    def embed(self, ids: list[int]) -> "torch.Tensor":
        need = [i for i in set(ids) if i not in self._rows]
        if need:
            for i, row in zip(need, self.target.rows("token_embd.weight", need)):
                self._rows[i] = row
        return self.torch.from_numpy(np.stack([self._rows[i] for i in ids]))[None]

    def set_variant_rope(self, variant: str) -> None:
        self.active["nope"] = set(self.no_rope) if VARIANTS[variant]["nope"] else set()

    # ---- context cache --------------------------------------------------------------
    def build_ctx(self, feats: np.ndarray, variant: str, chunk: int = 256) -> CtxCache:
        """feats: float32 [N][len(tap_ids)*hidden], position-major as the dump tool writes them."""
        torch, m = self.torch, self.model
        n = feats.shape[0]
        self.set_variant_rope(variant)
        ks = [[] for _ in m.layers]
        vs = [[] for _ in m.layers]
        with torch.inference_mode():
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                f = torch.from_numpy(np.array(feats[lo:hi]))[None]
                ctx = m.hidden_norm(m.fc(f))
                pos = torch.arange(lo, hi)[None]
                cos, sin = m.rotary_emb(ctx, pos)
                for li, layer in enumerate(m.layers):
                    a = layer.self_attn
                    k = a.k_norm(a.k_proj(ctx).view(1, hi - lo, -1, a.head_dim)).transpose(1, 2)
                    v = a.v_proj(ctx).view(1, hi - lo, -1, a.head_dim).transpose(1, 2)
                    self.active["layer"] = li
                    _, k = self.up.apply_rotary_pos_emb(k, k, cos, sin)
                    ks[li].append(k)
                    vs[li].append(v)
        self.active["layer"] = None
        return CtxCache([torch.cat(x, dim=2) for x in ks], [torch.cat(x, dim=2) for x in vs])

    # ---- one anchor -------------------------------------------------------------------
    def draft_block(self, cache: CtxCache, anchor: int, s: int, rows: int, scale: bool,
                    windowed: bool) -> list[int]:
        """rows-1 draft tokens for the anchor token at absolute position s, whose context is positions < s."""
        torch, m = self.torch, self.model
        ids = [anchor] + [self.mask_id] * (rows - 1)
        emb = self.embed(ids)
        if scale:
            emb = emb * self.embed_scale
        lo = max(0, s - self.context_window) if (windowed and self.context_window) else 0
        cache.set_range(lo, s)
        empty = torch.zeros(1, 0, len(self.tap_ids) * self.hidden)
        with torch.inference_mode():
            h = m(position_ids=torch.arange(s, s + rows)[None], noise_embedding=emb, target_hidden=empty,
                  past_key_values=cache, use_cache=True)
            tokens, _, _ = m.propose(h[:, 1:], torch.tensor([anchor]), self.head, 0.0)
        self.active["layer"] = None
        return [int(x) for x in tokens[0]]

    # ---- named stages (stage 1: the C++ drafter test) ---------------------------------------
    def stage_ctx(self, feats: np.ndarray, rec) -> None:
        """Context stages for positions [0, n): ctx_h (hidden_norm(fc(features))) and, per layer, the context
        K (k_norm, rotated unless the layer is NoPE) and V rows, [n][kv_heads*head_dim]."""
        torch, m = self.torch, self.model
        n = feats.shape[0]
        cache = self.build_ctx(feats, "R")
        with torch.inference_mode():
            ctx = m.hidden_norm(m.fc(torch.from_numpy(np.array(feats))[None]))
        rec("ctx_h", ctx[0].numpy())
        for li in range(len(m.layers)):
            rec(f"ctx_k.{li}", cache.k[li][0].transpose(0, 1).reshape(n, -1).numpy())
            rec(f"ctx_v.{li}", cache.v[li][0].transpose(0, 1).reshape(n, -1).numpy())

    def stage_block(self, cache: CtxCache, anchor: int, s: int, rec) -> list[int]:
        """Variant R for one anchor at position s (as draft_block), recording the intermediate stages of the
        upstream forward through forward hooks on the upstream modules. The candidate selector's walk is
        repeated here with the upstream modules to expose candidates and scores; its tokens are asserted equal
        to the upstream `propose` result."""
        torch, m = self.torch, self.model
        self.set_variant_rope("R")
        rows = self.block
        ids = [anchor] + [self.mask_id] * (rows - 1)
        emb = self.embed(ids)
        rec("emb", emb[0].numpy())
        lo = max(0, s - self.context_window) if self.context_window else 0
        cache.set_range(lo, s)
        empty = torch.zeros(1, 0, len(self.tap_ids) * self.hidden)
        handles = []

        def flat(t):
            return t.detach().float().reshape(-1, t.shape[-1]).numpy()

        for li, layer in enumerate(m.layers):
            handles.append(layer.input_layernorm.register_forward_hook(
                lambda mod, a, out, li=li: rec(f"attn_norm.{li}", flat(out))))
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda mod, a, kw, li=li: rec(f"attn_conv_in.{li}", flat(kw["hidden_states"])), with_kwargs=True))
            handles.append(layer.self_attn.register_forward_hook(
                lambda mod, a, kw, out, li=li: rec(f"attn_o.{li}", flat(out[0])), with_kwargs=True))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda mod, a, li=li: rec(f"h_attn.{li}", flat(a[0]))))
            handles.append(layer.post_attention_layernorm.register_forward_hook(
                lambda mod, a, out, li=li: rec(f"ffn_norm.{li}", flat(out))))
            handles.append(layer.mlp.register_forward_pre_hook(
                lambda mod, a, li=li: rec(f"mlp_conv_in.{li}", flat(a[0]))))
            handles.append(layer.mlp.register_forward_hook(
                lambda mod, a, out, li=li: rec(f"mlp_o.{li}", flat(out))))
            handles.append(layer.register_forward_hook(
                lambda mod, a, kw, out, li=li: rec(f"h.{li}", flat(out)), with_kwargs=True))
        handles.append(m.norm.register_forward_hook(lambda mod, a, out: rec("final", flat(out))))
        try:
            with torch.inference_mode():
                h = m(position_ids=torch.arange(s, s + rows)[None], noise_embedding=emb, target_hidden=empty,
                      past_key_values=cache, use_cache=True)
                hid = h[:, 1:]
                logits = m.compute_logits(hid, self.head)
                tokens, _, _ = m.propose(hid, torch.tensor([anchor]), self.head, 0.0)
                sel = m.candidate_selector
                unary, cands = torch.topk(logits, sel.top_k, dim=-1, sorted=False)
                proj = sel.hidden_projection(hid)
                prev = torch.tensor([anchor])
                out_c, out_s, out_t, conf = [], [], [], []
                for p in range(rows - 1):
                    sc = unary[:, p] + torch.einsum("br,bkr->bk", sel.predecessor_codebook(prev) * proj[:, p],
                                                    sel.successor_codebook(cands[:, p]))
                    idx = torch.argmax(sc, dim=-1)
                    prev = cands[:, p].gather(-1, idx[:, None])[:, 0]
                    order = torch.argsort(cands[0, p])
                    out_c.append(cands[0, p][order].numpy().astype(np.int32))
                    out_s.append(sc[0][order].numpy().astype(np.float32))
                    out_t.append(int(prev[0]))
                    e = torch.exp(sc[0].double() - sc[0].double().max())
                    conf.append(float(e[idx[0]] / e.sum()))
        finally:
            for hd in handles:
                hd.remove()
            self.active["layer"] = None
        assert out_t == [int(x) for x in tokens[0]], "selector walk differs from the upstream propose()"
        rec("hproj", proj[0].numpy())
        rec("logits", logits[0].numpy())
        rec("cand", np.stack(out_c))
        rec("score", np.stack(out_s))
        rec("conf", np.array(conf, dtype=np.float32)[:, None])
        rec("tokens", np.array(out_t, dtype=np.int32)[:, None])
        return out_t

    def draft_blocks(self, cache: CtxCache, anchor_pos: list[int], ids: np.ndarray, rows: int, scale: bool,
                     windowed: bool) -> list[list[int]]:
        """draft_block for several anchors in one forward (the weights are streamed once for all of them: the
        float32 matmuls at 16 rows are ~6x slower per row than at 256). Every anchor attends to its own context
        rows [max(0, s - context_window), s) and its own block; the masks are built per layer with the upstream
        rule (non-causal, |q - k| < sliding_window on the sliding layers). Rounding differs from the one-anchor
        forward at the 1e-6 level, so a near-tie can in principle pick another token; selfcheck compares both."""
        torch, m = self.torch, self.model
        b = len(anchor_pos)
        s = torch.tensor(anchor_pos)
        cw = self.context_window if (windowed and self.context_window) else None
        lo_each = torch.clamp(s - cw, min=0) if cw else torch.zeros_like(s)
        lo, hi = int(lo_each.min()), int(s.max())
        j = torch.arange(lo, hi)
        vis = (j[None, :] < s[:, None]) & (j[None, :] >= lo_each[:, None])  # [b, hi-lo]
        qpos = s[:, None] + torch.arange(rows)[None, :]  # [b, rows]
        kpos = torch.cat([j[None, :].expand(b, -1), qpos], dim=1)  # [b, hi-lo+rows]
        pad = torch.cat([vis[:, None, :].expand(b, rows, -1), torch.ones(b, rows, rows, dtype=torch.bool)], dim=2)
        near = (qpos[:, :, None] - kpos[:, None, :]).abs() < self.window
        full = pad[:, None]
        sliding = (pad & near)[:, None]
        types = self.cfg.layer_types
        self.layer_masks = [sliding if t == "sliding_attention" else full for t in types]
        cache.set_range(lo, hi)
        ids_b = torch.tensor([[int(ids[a])] + [self.mask_id] * (rows - 1) for a in anchor_pos])
        emb = torch.from_numpy(np.stack([self.embed(r.tolist())[0].numpy() for r in ids_b]))
        if scale:
            emb = emb * self.embed_scale
        empty = torch.zeros(b, 0, len(self.tap_ids) * self.hidden)
        try:
            with torch.inference_mode():
                h = m(position_ids=qpos, noise_embedding=emb, target_hidden=empty, past_key_values=cache,
                      use_cache=True)
                tokens, _, _ = m.propose(h[:, 1:], ids_b[:, 0], self.head, 0.0)
        finally:
            self.layer_masks = None
            self.active["layer"] = None
        self._last_hidden = h
        return [[int(x) for x in row] for row in tokens]

    def draft_block_uncached(self, feats: np.ndarray, anchor: int, s: int, rows: int, scale: bool,
                             lo: int = 0) -> tuple[list[int], "torch.Tensor"]:
        """The same with the upstream forward on the full context window (no cache): the reference for the cache."""
        torch, m = self.torch, self.model
        ids = [anchor] + [self.mask_id] * (rows - 1)
        emb = self.embed(ids)
        if scale:
            emb = emb * self.embed_scale
        f = torch.from_numpy(np.ascontiguousarray(feats[lo:s]))[None]
        with torch.inference_mode():
            h = m(position_ids=torch.arange(lo, s + rows)[None], noise_embedding=emb, target_hidden=f,
                  past_key_values=None, use_cache=False)
            tokens, _, _ = m.propose(h[:, 1:], torch.tensor([anchor]), self.head, 0.0)
        self.active["layer"] = None
        return [int(x) for x in tokens[0]], h

    # ---- a whole prompt ----------------------------------------------------------------
    def draft_many(self, cache, anchor_pos, ids, rows, scale, windowed):
        """One anchor: the upstream forward with the upstream mask. Several: draft_blocks."""
        if len(anchor_pos) == 1:
            a = anchor_pos[0]
            return [self.draft_block(cache, int(ids[a]), a, rows, scale, windowed)]
        return self.draft_blocks(cache, anchor_pos, ids, rows, scale, windowed)

    def run_prompt(self, ids: np.ndarray, taps: np.ndarray, first_anchor: int, variants: list[str],
                   l_ds: list[int], stride: int = 1, limit: int | None = None, log=print,
                   batch: int = 1) -> dict:
        """ids [N] int; taps [N][K][H]. Anchors first_anchor, first_anchor+stride, .. <= N-2."""
        n = len(ids)
        k_taps = taps.shape[1]
        feats = taps.reshape(n, k_taps * taps.shape[2])
        anchors = list(range(first_anchor, n - 1, stride))
        if limit:
            anchors = anchors[:limit]
        out: dict[str, np.ndarray] = {"anchors": np.array(anchors, dtype=np.int32)}
        out["avail"] = np.array([min(MAX_D, n - 1 - s) for s in anchors], dtype=np.int16)
        seconds = {}
        cache_for: dict[bool, CtxCache] = {}
        for variant in variants:
            spec = VARIANTS[variant]
            t0 = time.time()
            key = spec["nope"]
            if key not in cache_for:
                cache_for[key] = self.build_ctx(feats[:n - 1], variant)
            cache = cache_for[key]
            self.set_variant_rope(variant)
            t_ctx = time.time() - t0
            t1 = time.time()
            if spec["block"] == "fixed":
                drafts = np.zeros((len(anchors), MAX_D), dtype=np.int32)
                acc = np.full((len(anchors), MAX_D), -1, dtype=np.int16)
                for lo in range(0, len(anchors), batch):
                    sel = list(range(lo, min(len(anchors), lo + batch)))
                    got = self.draft_many(cache, [anchors[i] for i in sel], ids, self.block, spec["scale"], True)
                    for i, d in zip(sel, got):
                        drafts[i, :len(d)] = d
                        acc[i] = accepted_by_d(d, ids, anchors[i], MAX_D)
                    log(f"    {variant}: {sel[-1] + 1}/{len(anchors)} anchors, {(time.time() - t1) / (sel[-1] + 1):.3f} s each")
                out[f"{variant}_drafts"] = drafts
                out[f"{variant}_acc"] = acc
            else:
                acc = np.full((len(anchors), MAX_D), -1, dtype=np.int16)
                for d_len in l_ds:
                    if d_len + 1 < self.conv_kernel - 1:
                        # upstream _grouped_dynamic_convolve slices blocks[:, :-offset] and pads by offset: a block
                        # shorter than kernel_size - 1 rows comes out too long and the forward raises. Not run.
                        log(f"    {variant} d={d_len}: block of {d_len + 1} rows is shorter than the convolution's "
                            f"{self.conv_kernel - 1}-row minimum in the upstream file; skipped")
                        continue
                    drafts = np.zeros((len(anchors), d_len), dtype=np.int32)
                    for lo in range(0, len(anchors), batch):
                        sel = list(range(lo, min(len(anchors), lo + batch)))
                        got = self.draft_many(cache, [anchors[i] for i in sel], ids, d_len + 1, spec["scale"], False)
                        for i, d in zip(sel, got):
                            drafts[i] = d
                            acc[i, d_len - 1] = accepted_by_d(d, ids, anchors[i], d_len)[d_len - 1]
                        log(f"    {variant} d={d_len}: {sel[-1] + 1}/{len(anchors)} anchors, "
                            f"{(time.time() - t1) / (sel[-1] + 1):.3f} s each")
                    out[f"{variant}_drafts_d{d_len}"] = drafts
                    t1 = time.time()
                out[f"{variant}_acc"] = acc
            seconds[variant] = {"ctx": t_ctx, "anchors": time.time() - t1}
        out["_seconds"] = seconds
        return out


def accepted_by_d(drafts: list[int], ids: np.ndarray, s: int, max_d: int) -> np.ndarray:
    """acc[d-1] = how many of the first d drafts equal the continuation ids[s+1 ..], for d <= available
    continuation; -1 where the continuation is shorter than d (unknown)."""
    n = len(ids)
    avail = min(max_d, n - 1 - s)
    acc = np.full(max_d, -1, dtype=np.int16)
    run = 0
    broken = False
    for i in range(avail):
        if not broken and i < len(drafts) and drafts[i] == int(ids[s + 1 + i]):
            run += 1
        else:
            broken = True
        if i < len(drafts):
            acc[i] = run
    return acc


def load_drafter(args) -> Drafter:
    up = load_upstream(Path(args.upstream))
    t = time.time()
    d = Drafter(up, Gguf(Path(args.draft)), Gguf(Path(args.target)), args.threads)
    print(f"loaded drafter: {d.tensor_count} tensors, {d.param_count:,} parameters in {time.time() - t:.1f} s")
    return d


# --------------------------------------------------------------------------------------
# self-checks
# --------------------------------------------------------------------------------------
def random_blocks(rng: np.random.Generator, gtype: int, n_blocks: int) -> np.ndarray:
    kind, per, size = TYPES[gtype]
    raw = rng.integers(0, 256, size=(n_blocks, size), dtype=np.uint8)
    scale_at = {"Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208], "IQ4_XS": [0]}[kind]
    for at in scale_at:
        raw[:, at:at + 2] = rng.normal(0, 0.02, size=(n_blocks, 1)).astype(np.float16).view(np.uint8)
    return raw


def check_dequant_vs_gguf_py(g: Gguf, names: list[str], results: list) -> None:
    import gguf
    for name in names:
        t = g.tensors[name]
        mine = g.read(name)
        theirs = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(mine.shape)
        results.append((f"dequantized {name} ({t.tensor_type.name}, {mine.size:,} values) equals gguf-py exactly",
                        bool(np.array_equal(mine, theirs))))


def selfcheck(args) -> int:
    import gguf
    results: list[tuple[str, bool]] = []
    notes: list[str] = []

    def check(label: str, ok: bool, note: str = "") -> None:
        results.append((label, bool(ok)))
        if note:
            notes.append(note)

    # 1. dequantizers on random blocks of every type used by either GGUF, against gguf-py.
    rng = np.random.default_rng(7)
    for gtype in (8, 12, 13, 14, 23):
        raw = random_blocks(rng, gtype, 3000)
        kind, per, size = TYPES[gtype]
        mine = dequantize(gtype, raw, 3000 * per)
        theirs = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(gtype)).reshape(-1)
        check(f"{kind}: own dequantization equals gguf-py on 3000 random blocks", np.array_equal(mine, theirs))

    # 2. upstream pin (load_upstream refuses otherwise) and the model.
    up = load_upstream(Path(args.upstream))
    check(f"upstream file sha256 equals the pin {UPSTREAM_SHA256[:12]}...", True)
    draft, target = Gguf(Path(args.draft)), Gguf(Path(args.target))
    t0 = time.time()
    d = Drafter(up, draft, target, args.threads)
    notes.append(f"model build and dequantization: {time.time() - t0:.1f} s")
    total = sum(int(np.prod(Gguf.shape(t))) for t in draft.tensors.values())
    check("every GGUF tensor is bound to an upstream parameter; counts agree",
          d.tensor_count == len(draft.tensors) and d.param_count == total,
          f"tensors {d.tensor_count}, parameters {d.param_count:,}")
    if args.card:
        check(f"tensor count equals the card's {CARD_TENSORS}", d.tensor_count == CARD_TENSORS)
        check(f"parameter sum equals the card's {CARD_PARAMS:,}", d.param_count == CARD_PARAMS)
        card = json.loads(Path(args.card_config).read_text())
        mismatch = []
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "vocab_size", "sliding_window", "layer_types",
                    "no_rope_layers", "rms_norm_eps", "is_causal", "rope_theta", "max_position_embeddings"):
            if card.get(key) != d.cfgd.get(key):
                mismatch.append((key, card.get(key), d.cfgd.get(key)))
        for key, value in card["dflash_config"].items():
            if d.cfgd["dflash_config"].get(key) != value:
                mismatch.append(("dflash_config." + key, value, d.cfgd["dflash_config"].get(key)))
        check("configuration built from the GGUF metadata equals the card's config.json", not mismatch,
              f"config differences: {mismatch}" if mismatch else "")

    # 3. exact dequantization of the named tensors.
    check_dequant_vs_gguf_py(draft, ["fc.weight", "blk.0.attn_q.weight"], results)
    for name in ("token_embd.weight", "output.weight"):
        t = target.tensors[name]
        # whole tensor for the small fixtures, a block-aligned slice of rows for the real file
        rows = min(Gguf.shape(t)[0], 256)
        mine = target.rows(name, list(range(rows)))
        theirs = gguf.quants.dequantize(np.asarray(t.data)[:rows], t.tensor_type).reshape(mine.shape)
        check(f"target {name} ({t.tensor_type.name}) first {rows} rows equal gguf-py exactly", np.array_equal(mine, theirs))

    # 4. determinism and cache equivalence on random inputs.
    torch = d.torch
    n_ctx = int(args.ctx)
    g = np.random.default_rng(11)
    vocab = d.cfg.vocab_size
    ids = g.integers(0, vocab, size=n_ctx + 40).astype(np.int32)
    feats = (g.standard_normal((n_ctx + 40, len(d.tap_ids) * d.hidden)) * 0.5).astype(np.float32)
    taps = feats.reshape(n_ctx + 40, len(d.tap_ids), d.hidden)
    one = d.run_prompt(ids, taps, n_ctx, ["R", "Rs", "L"], [2], limit=3, log=lambda *_: None)
    two = d.run_prompt(ids, taps, n_ctx, ["R", "Rs", "L"], [2], limit=3, log=lambda *_: None)
    same = all(np.array_equal(one[k], two[k]) for k in one if k != "_seconds")
    check("a run on random inputs is deterministic (every output array identical, 3 anchors x 3 variants)", same)
    # cache == uncached upstream forward, with and without the NoPE hook, scaled and not.
    s = n_ctx + 3
    for variant in ("R", "Rs", "L"):
        spec = VARIANTS[variant]
        cache = d.build_ctx(feats[:s], variant)
        d.set_variant_rope(variant)
        rows = d.block if spec["block"] == "fixed" else 3
        tok_c = d.draft_block(cache, int(ids[s]), s, rows, spec["scale"], True)
        cache.set_range(0, s)
        with torch.inference_mode():
            ids_b = [int(ids[s])] + [d.mask_id] * (rows - 1)
            emb = d.embed(ids_b) * (d.embed_scale if spec["scale"] else 1.0)
            h_c = d.model(position_ids=torch.arange(s, s + rows)[None], noise_embedding=emb,
                          target_hidden=torch.zeros(1, 0, len(d.tap_ids) * d.hidden), past_key_values=cache,
                          use_cache=True)
        tok_u, h_u = d.draft_block_uncached(feats, int(ids[s]), s, rows, spec["scale"])
        h_u = h_u[:, -rows:]
        diff = float((h_c - h_u).abs().max())
        scale = float(h_u.abs().max())
        check(f"{variant}: cached block forward equals the uncached upstream forward (max abs {diff:.2e}, "
              f"rel {diff / max(scale, 1e-30):.2e}) and the drafts agree", diff / max(scale, 1e-30) < 1e-4 and tok_c == tok_u)
    # the comparison can fail: a cache that misses its last context row must differ from the uncached forward.
    cache = d.build_ctx(feats[:s], "R")
    d.set_variant_rope("R")
    cache.set_range(0, s - 1)
    with torch.inference_mode():
        emb = d.embed([int(ids[s])] + [d.mask_id] * (d.block - 1))
        h_bad = d.model(position_ids=torch.arange(s, s + d.block)[None], noise_embedding=emb,
                        target_hidden=torch.zeros(1, 0, len(d.tap_ids) * d.hidden), past_key_values=cache,
                        use_cache=True)
    _, h_ref = d.draft_block_uncached(feats, int(ids[s]), s, d.block, False)
    check("twin: a cache one context row short is detected by the same comparison",
          float((h_bad - h_ref[:, -d.block:]).abs().max()) / float(h_ref.abs().max()) > 1e-3)
    # context window of the recipe variant: rows older than context_window are dropped.
    if d.context_window:
        w = d.context_window
        s2 = w + 20
        big = np.random.default_rng(5)
        feats2 = (big.standard_normal((s2 + 4, len(d.tap_ids) * d.hidden)) * 0.5).astype(np.float32)
        ids2 = big.integers(0, vocab, size=s2 + 4).astype(np.int32)
        cache2 = d.build_ctx(feats2[:s2], "R")
        tok_w = d.draft_block(cache2, int(ids2[s2]), s2, d.block, False, True)
        tok_u, _ = d.draft_block_uncached(feats2, int(ids2[s2]), s2, d.block, False, lo=s2 - w)
        check(f"R: with a context of {s2} rows the last {w} (context_window) are used, equal to the "
              f"upstream forward on exactly those rows", tok_w == tok_u)
    # several anchors in one forward (the batched path run uses) equal the one-anchor upstream forward.
    if d.context_window:
        w = d.context_window
        s2 = w + 20
        big = np.random.default_rng(5)
        feats2 = (big.standard_normal((s2 + 8, len(d.tap_ids) * d.hidden)) * 0.5).astype(np.float32)
        ids2 = big.integers(0, vocab, size=s2 + 8).astype(np.int32)
        anchors = [s2 - 5, s2 - 4, s2, s2 + 3]  # contexts both below and beyond context_window
    else:
        feats2, ids2, anchors = feats, ids, [n_ctx, n_ctx + 1, n_ctx + 2, n_ctx + 5]
    for variant in ("R", "Rs", "L"):
        spec = VARIANTS[variant]
        rows = d.block if spec["block"] == "fixed" else 3
        windowed = spec["block"] == "fixed"
        cache = d.build_ctx(feats2[:max(anchors)], variant)
        d.set_variant_rope(variant)
        got = d.draft_blocks(cache, anchors, ids2, rows, spec["scale"], windowed)
        h_b = d._last_hidden
        worst, same_tokens = 0.0, True
        for bi, a in enumerate(anchors):
            lo = max(0, a - w) if (windowed and d.context_window) else 0
            tok_u, h_u = d.draft_block_uncached(feats2, int(ids2[a]), a, rows, spec["scale"], lo=lo)
            worst = max(worst, float((h_b[bi] - h_u[0]).abs().max()) / float(h_u.abs().max()))
            same_tokens &= got[bi] == tok_u
        check(f"{variant}: {len(anchors)} anchors in one batched forward equal the one-anchor upstream forward "
              f"(max rel {worst:.2e}, drafts {'equal' if same_tokens else 'DIFFER'})", worst < 1e-4 and same_tokens)
    if d.context_window:  # twin: the batched forward without the context window must differ beyond it
        cache = d.build_ctx(feats2[:max(anchors)], "R")
        d.set_variant_rope("R")
        d.draft_blocks(cache, anchors, ids2, d.block, False, True)
        h_w = d._last_hidden
        d.draft_blocks(cache, anchors, ids2, d.block, False, False)
        check("twin: the batched forward with the context window switched off is detected",
              float((h_w - d._last_hidden).abs().max()) / float(h_w.abs().max()) > 1e-3)
    # the NoPE hook acts: layer 4 keys are not rotated in R and are rotated in L.
    cr, cl = d.build_ctx(feats[:8], "R"), d.build_ctx(feats[:8], "L")
    for layer in sorted(d.no_rope):
        a = d.model.layers[layer].self_attn
        with torch.inference_mode():
            ctx = d.model.hidden_norm(d.model.fc(torch.from_numpy(feats[:8])[None]))
            raw_k = a.k_norm(a.k_proj(ctx).view(1, 8, -1, a.head_dim)).transpose(1, 2)
        check(f"R: layer {layer} context keys are unrotated; L: rotated",
              bool(torch.equal(cr.k[layer], raw_k)) and not torch.equal(cl.k[layer], raw_k))
    notes.append("the context window (16384 in the real file) is applied to R and Rs only; stage 0 prompts "
                 "stay below it, so it only runs here on random inputs")

    width = max(len(r[0]) for r in results)
    failed = 0
    for label, ok in results:
        print(("PASS  " if ok else "FAIL  ") + label)
        failed += 0 if ok else 1
    for note in notes:
        print("note: " + note)
    print(f"selfcheck: {len(results) - failed} of {len(results)} passed")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"results": results, "notes": notes}, indent=1))
    return 1 if failed else 0


# --------------------------------------------------------------------------------------
# run and timing
# --------------------------------------------------------------------------------------
def run_one(d: Drafter, dump: Path, out_dir: Path, args) -> None:
    meta = json.loads((dump / "meta.json").read_text())
    if meta["tap_layers"] != d.tap_ids:
        sys.exit(f"dump taps layers {meta['tap_layers']} differ from the drafter's {d.tap_ids}")
    if meta["hidden"] != d.hidden:
        sys.exit("dump hidden size differs from the drafter's")
    ids = np.fromfile(dump / meta["ids_file"], dtype=np.int32)
    n = meta["prompt_len"] + meta["generated"]
    if len(ids) != n:
        sys.exit("ids file has the wrong length")
    taps = np.memmap(dump / meta["taps_file"], dtype=np.float32, mode="r",
                     shape=(n, len(meta["tap_layers"]), meta["hidden"]))
    variants = args.variants.split(",")
    l_ds = [int(x) for x in args.l_d.split(",")]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = d.run_prompt(ids, taps, meta["prompt_len"], variants, l_ds, args.stride, args.limit, batch=args.batch)
    seconds = res.pop("_seconds")
    np.savez(out_dir / "oracle.tmp.npz", ids=ids, prompt_len=meta["prompt_len"], **res)
    (out_dir / "oracle.tmp.npz").rename(out_dir / "oracle.npz")  # present = complete
    info = {"variants": variants, "l_ds": l_ds, "stride": args.stride, "anchors": int(len(res["anchors"])),
            "batch": args.batch, "seconds": seconds, "wall": time.time() - t0, "upstream_sha256": UPSTREAM_SHA256,
            "draft": str(args.draft), "target": str(args.target), "threads": args.threads, "dump": str(dump)}
    (out_dir / "oracle.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {out_dir}/oracle.npz: {len(res['anchors'])} anchors in {info['wall']:.1f} s", flush=True)


def run(args) -> int:
    """One dump (--dump, --out), or every prompt of a sets file (--sets, --dumps, --out) in one model load;
    prompts whose oracle.npz exists are skipped, so an interrupted run is resumed by repeating the command."""
    if bool(args.dump) == bool(args.sets):
        sys.exit("give either --dump DIR (with --out) or --sets FILE --dumps DIR --out DIR")
    jobs = []
    if args.dump:
        jobs.append((Path(args.dump), Path(args.out)))
    else:
        for line in Path(args.sets).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                name = line.split("\t")[0]
                jobs.append((Path(args.dumps) / name, Path(args.out) / name))
    todo = [(dm, out) for dm, out in jobs if not (out / "oracle.npz").exists()]
    for dm, out in jobs:
        if (out / "oracle.npz").exists():
            print(f"skip (done): {out}")
    if not todo:
        return 0
    d = load_drafter(args)
    for dm, out in todo:
        print(f"== {dm.name}", flush=True)
        run_one(d, dm, out, args)
    return 0


def timing(args) -> int:
    """One anchor on random inputs: context cache build per row and block forward per row count."""
    d = load_drafter(args)
    torch = d.torch
    g = np.random.default_rng(3)
    n = int(args.ctx) + 64
    ids = g.integers(0, d.cfg.vocab_size, size=n).astype(np.int32)
    feats = (g.standard_normal((n, len(d.tap_ids) * d.hidden)) * 0.5).astype(np.float32)
    t = time.time()
    cache = d.build_ctx(feats[:n - 1], "R")
    per_row = (time.time() - t) / (n - 1)
    print(f"context cache: {per_row * 1000:.2f} ms per context row ({n - 1} rows)")
    for rows, label in ((16, "R/Rs block of 16"), (3, "L block of 3 (d=2)"), (9, "L block of 9 (d=8)")):
        d.draft_block(cache, int(ids[args.ctx]), int(args.ctx), rows, False, True)  # warm
        t = time.time()
        for i in range(args.anchors):
            d.draft_block(cache, int(ids[args.ctx + i]), int(args.ctx + i), rows, False, True)
        print(f"{label}: {(time.time() - t) / args.anchors:.3f} s per anchor ({args.anchors} anchors, ctx {args.ctx})")
    for rows, b in ((16, 8), (16, 16), (16, 32), (3, 32), (9, 32)):
        anchors = [int(args.ctx) + i for i in range(b)]
        d.draft_blocks(cache, anchors[:2], ids, rows, False, True)  # warm
        t = time.time()
        d.draft_blocks(cache, anchors, ids, rows, False, True)
        dt = time.time() - t
        print(f"block of {rows}, {b} anchors in one forward: {dt:.2f} s = {dt / b:.3f} s per anchor")
    return 0


def stages(args) -> int:
    """Writes FILE.bin: records `u32 name_len, name, u32 rows, u32 width, u32 kind (0 float32, 1 int32),
    data`. Names: `ctx_h`, `ctx_k.L`, `ctx_v.L` (positions 0..max anchor - 1), `anchors` ([A][1] int32) and,
    per anchor s, `a<s>/<stage>`: emb, attn_norm.L, attn_conv_in.L, attn_o.L, h_attn.L, ffn_norm.L,
    mlp_conv_in.L, mlp_o.L, h.L, final, hproj, logits, cand, score, conf, tokens."""
    import struct
    d = load_drafter(args)
    dump = Path(args.dump)
    meta = json.loads((dump / "meta.json").read_text())
    if meta["tap_layers"] != d.tap_ids:
        sys.exit(f"dump taps layers {meta['tap_layers']} differ from the drafter's {d.tap_ids}")
    if meta["hidden"] != d.hidden:
        sys.exit("dump hidden size differs from the drafter's")
    n = meta["prompt_len"] + meta["generated"]
    ids = np.fromfile(dump / meta["ids_file"], dtype=np.int32)
    if len(ids) != n:
        sys.exit("ids file has the wrong length")
    taps = np.memmap(dump / meta["taps_file"], dtype=np.float32, mode="r",
                     shape=(n, len(meta["tap_layers"]), meta["hidden"]))
    if args.anchors:
        anchors = [int(x) for x in args.anchors.split(",")]
    else:
        anchors = list(range(args.first, args.last + 1, args.stride))
    if not anchors or min(anchors) < 1 or max(anchors) > n - 1:
        sys.exit(f"anchors must lie in [1, {n - 1}]")
    feats = taps.reshape(n, -1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    t0 = time.time()
    with open(tmp, "wb") as f:
        def write(name: str, arr: np.ndarray) -> None:
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            kind = 0 if arr.dtype == np.float32 else 1
            if arr.dtype not in (np.float32, np.int32):
                raise TypeError(f"{name}: {arr.dtype}")
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<III", arr.shape[0], arr.shape[1], kind))
            f.write(arr.tobytes())

        write("anchors", np.array(anchors, dtype=np.int32))
        d.stage_ctx(np.array(feats[:max(anchors)]), write)
        cache = d.build_ctx(np.array(feats[:max(anchors)]), "R")
        for a in anchors:
            t1 = time.time()
            d.stage_block(cache, int(ids[a]), a, lambda name, arr, a=a: write(f"a{a}/{name}", arr))
            print(f"  anchor {a}: {time.time() - t1:.2f} s", flush=True)
    tmp.rename(out)
    Path(str(out) + ".json").write_text(json.dumps({
        "anchors": anchors, "draft": str(args.draft), "target": str(args.target), "dump": str(dump),
        "variant": "R", "upstream_sha256": UPSTREAM_SHA256, "wall": time.time() - t0, "threads": args.threads}, indent=1))
    print(f"wrote {out}: {len(anchors)} anchors in {time.time() - t0:.1f} s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("selfcheck", "run", "timing", "stages"):
        s = sub.add_parser(name)
        s.add_argument("--upstream", default=DEFAULT_UPSTREAM)
        s.add_argument("--draft", required=True)
        s.add_argument("--target", required=True)
        s.add_argument("--threads", type=int, default=None)
        if name == "selfcheck":
            s.add_argument("--card", action="store_true", help="also check the real file against the model card")
            s.add_argument("--card-config", default="/server/staging/ripple-gufo/perf/pr31904-build/cards/config.json")
            s.add_argument("--ctx", type=int, default=24, help="context rows for the random-input checks")
            s.add_argument("--json-out")
        if name == "run":
            s.add_argument("--dump", help="one ricursive_dump_taps directory")
            s.add_argument("--sets", help="sets.tsv: every prompt in it, in one model load")
            s.add_argument("--dumps", help="with --sets: directory of dump directories (dumps/<name>)")
            s.add_argument("--out", required=True)
            s.add_argument("--variants", default="R,Rs,L")
            s.add_argument("--l-d", default="2", help="draft lengths for variant L, comma separated (1..15)")
            s.add_argument("--stride", type=int, default=1)
            s.add_argument("--limit", type=int, default=None)
            s.add_argument("--batch", type=int, default=32,
                           help="anchors per forward (1 = the upstream mask path, one anchor at a time)")
        if name == "timing":
            s.add_argument("--ctx", type=int, default=600)
            s.add_argument("--anchors", type=int, default=3)
        if name == "stages":
            s.add_argument("--dump", required=True, help="one ricursive_dump_taps directory")
            s.add_argument("--out", required=True, help="the binary stage file to write")
            s.add_argument("--anchors", help="comma separated anchor positions")
            s.add_argument("--first", type=int, default=1)
            s.add_argument("--last", type=int, default=0)
            s.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    return {"selfcheck": selfcheck, "run": run, "timing": timing, "stages": stages}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
