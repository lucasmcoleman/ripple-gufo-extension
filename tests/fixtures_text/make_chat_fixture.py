#!/usr/bin/env python3
"""Write chat-tiny.gguf: the real Ripple tokenizer and chat template (every
tokenizer.* / general.* key copied from fixtures/vocab-only.gguf) with tiny
random `ricursive` weights, vocab 100352, hidden size 128.

usage: make_chat_fixture.py VOCAB_ONLY.gguf OUT.gguf [SEED]
Needs numpy and llama.cpp's gguf-py (GGUF_PY, default
/server/programming/llama.cpp-ripple/gguf-py). F32 weights, about 105 MB; the
file is generated into the build directory and never committed.
"""
import os
import sys

sys.path.insert(0, os.environ.get("GGUF_PY", "/server/programming/llama.cpp-ripple/gguf-py"))
import numpy as np
import gguf
from gguf import GGUFWriter

if len(sys.argv) < 3:
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 4242

E, L, H, KV, D, FF, WINDOW = 128, 5, 2, 1, 128, 256, 8
CLAMP, ESCALE = 3.0, 11.3137085


def main():
    reader = gguf.GGUFReader(SRC)
    vocab = int(reader.fields["ricursive.vocab_size"].contents())
    rng = np.random.default_rng(SEED)
    pattern = [(i % 5) != 4 for i in range(L)]  # True = sliding, as the real model's 4:1

    w = GGUFWriter(OUT + ".tmp", "ricursive")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key.startswith("ricursive."):
            continue
        sub = field.types[1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), field.types[0], sub)
    w.add_block_count(L)
    w.add_context_length(4096)
    w.add_embedding_length(E)
    w.add_feed_forward_length(FF)
    w.add_head_count(H)
    w.add_head_count_kv(KV)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_key_length(D)
    w.add_value_length(D)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(D)
    w.add_vocab_size(vocab)
    w.add_sliding_window(WINDOW)
    w.add_sliding_window_pattern(pattern)
    w.add_rope_pattern(pattern)
    w.add_embedding_scale(ESCALE)
    w.add_float32(gguf.Keys.LLM.SWIGLU_CLAMP.format(arch="ricursive"), CLAMP)

    def mat(rows, cols, std):
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    def norm(n):
        return (1.0 + 0.25 * rng.standard_normal(n)).astype(np.float32)

    QW, KW = H * D, KV * D
    t = {
        "token_embd.weight": mat(vocab, E, 0.1),
        "output_norm.weight": norm(E),
        # Large enough that greedy decoding does not collapse onto one token.
        "output.weight": mat(vocab, E, 2.5 / np.sqrt(E)),
    }
    for i in range(L):
        p = f"blk.{i}."
        t[p + "attn_norm.weight"] = norm(E)
        t[p + "attn_q.weight"] = mat(QW, E, 1.0 / np.sqrt(E))
        t[p + "attn_k.weight"] = mat(KW, E, 1.0 / np.sqrt(E))
        t[p + "attn_v.weight"] = mat(KW, E, 1.0 / np.sqrt(E))
        t[p + "attn_output.weight"] = mat(E, QW, 1.0 / np.sqrt(QW))
        t[p + "attn_q_norm.weight"] = norm(D)
        t[p + "attn_k_norm.weight"] = norm(D)
        t[p + "attn_gate.weight"] = mat(QW, E, 1.5 / np.sqrt(E))
        t[p + "attn_vres_lambda.weight"] = rng.uniform(-2.0, 2.0, size=1).astype(np.float32)
        t[p + "post_attention_norm.weight"] = norm(E)
        t[p + "ffn_norm.weight"] = norm(E)
        t[p + "ffn_gate.weight"] = mat(FF, E, 1.6 / np.sqrt(E))
        t[p + "ffn_up.weight"] = mat(FF, E, 1.6 / np.sqrt(E))
        t[p + "ffn_down.weight"] = mat(E, FF, 1.0 / np.sqrt(FF))
        t[p + "post_ffw_norm.weight"] = norm(E)
    for name, value in t.items():
        w.add_tensor(name, value)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(OUT + ".tmp", OUT)
    size = os.path.getsize(OUT)
    assert size < 300 * 1024 * 1024, size
    print(f"wrote {OUT}: {len(t)} tensors, vocab {vocab}, {size / 1e6:.1f} MB")


main()
