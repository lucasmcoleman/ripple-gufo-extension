#!/usr/bin/env python3
"""Write a tiny random DFlash2 drafter GGUF that pairs with a tiny `ricursive` fixture.

usage: make_drafter_fixture.py TARGET.gguf OUT.gguf [--seed N]

Same tensor names, shapes and metadata keys as the real Ripple-65B-Preview-DFlash2
GGUF, at a size the CPU oracle runs in seconds: hidden size and vocabulary come
from the target (the drafter reads the target's embedding rows and output head),
5 layers (4 sliding, window 8, then 1 full layer without rotary), 8 query / 2 KV
heads of 32, convolution kernel 4 / group 16, selector rank 16 / top-k 8. The big
matrices are Q8_0, the rest F32. All weights are seeded random; the tap layers are
the target's own layers 0..4 (GGUF numbering 1..5). Run with the oracle venv
(numpy + gguf); nothing here is the real model, only the plumbing.
"""
import argparse
import sys

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q, GGUFWriter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    t = gguf.GGUFReader(args.target)
    H = int(t.fields["ricursive.embedding_length"].contents())
    V = int(t.fields["ricursive.vocab_size"].contents())
    n_target = int(t.fields["ricursive.block_count"].contents())
    L, NH, NKV, HD, FF, RANK, TOPK, KS, GS = 5, 8, 2, 32, 256, 16, 8, 4, 16
    taps = list(range(min(5, n_target)))
    assert len(taps) == 5 and H % GS == 0
    rng = np.random.default_rng(args.seed)

    def mat(rows, cols):
        return (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    w = GGUFWriter(args.out, "dflash")
    w.add_name("tiny-dflash2-random")
    w.add_block_count(L)
    w.add_context_length(4096)
    w.add_embedding_length(H)
    w.add_feed_forward_length(FF)
    w.add_head_count(NH)
    w.add_head_count_kv(NKV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_causal_attention(False)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_uint32("dflash.block_size", 16)
    w.add_uint32("dflash.conv_kernel_size", KS)
    w.add_uint32("dflash.conv_group_size", GS)
    w.add_uint32("dflash.selector_rank", RANK)
    w.add_uint32("dflash.selector_top_k", TOPK)
    w.add_array("dflash.target_layers", [i + 1 for i in taps])
    w.add_uint32("dflash.attention.sliding_window", 8)
    w.add_array("dflash.attention.sliding_window_pattern", [True] * 4 + [False])
    w.add_array("dflash.attention.rope_pattern", [True] * 4 + [False])
    w.add_uint32("dflash.attention.context_window", 64)
    w.add_uint32("tokenizer.ggml.mask_token_id", V - 1)

    def put(name, arr, big=False):
        if big:
            w.add_tensor(name, gguf.quants.quantize(arr, Q.Q8_0), raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(name, arr.astype(np.float32))

    put("fc.weight", mat(H, len(taps) * H), True)
    put("enc.output_norm.weight", 1 + 0.1 * rng.standard_normal(H))
    put("output_norm.weight", 1 + 0.1 * rng.standard_normal(H))
    put("selector_hidden.weight", mat(RANK, H), True)
    put("selector_predecessor.weight", rng.standard_normal((V, RANK)))
    put("selector_successor.weight", rng.standard_normal((V, RANK)))
    for i in range(L):
        b = f"blk.{i}."
        put(b + "attn_norm.weight", 1 + 0.1 * rng.standard_normal(H))
        put(b + "ffn_norm.weight", 1 + 0.1 * rng.standard_normal(H))
        put(b + "attn_q.weight", mat(NH * HD, H), True)
        put(b + "attn_k.weight", mat(NKV * HD, H), True)
        put(b + "attn_v.weight", mat(NKV * HD, H), True)
        put(b + "attn_output.weight", mat(H, NH * HD), True)
        put(b + "attn_q_norm.weight", 1 + 0.1 * rng.standard_normal(HD))
        put(b + "attn_k_norm.weight", 1 + 0.1 * rng.standard_normal(HD))
        put(b + "ffn_gate.weight", mat(FF, H), True)
        put(b + "ffn_up.weight", mat(FF, H), True)
        put(b + "ffn_down.weight", mat(H, FF), True)
        for side in ("attn", "ffn"):
            base = 0.1 * rng.standard_normal((2, KS, H))
            base[:, 0] += 1.0  # tap 0 near identity, as a trained kernel would be
            put(b + side + "_conv_base", base)
            put(b + side + "_conv_proj.weight", 0.1 * mat(2 * KS * (H // GS), H), True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: hidden {H}, vocab {V}, taps {taps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
