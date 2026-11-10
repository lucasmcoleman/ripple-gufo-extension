#!/usr/bin/env python3
"""Seeded random conversations rendered by jinja2 exactly as oracle/tools/make_chat_goldens.py does
(ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols, tojson=json.dumps(ensure_ascii=False)).
usage: gen_chat_prop.py TEMPLATE.jinja OUT.jsonl [N=700] [SEED=20261110]
Each line: {"id", "kind": "valid"|"edge", "input": {messages, tools?, add_generation_prompt?, clear_thinking?},
            "rendered": str | null, "error": null | ExceptionName}"""
import json, random, sys
from jinja2.sandbox import ImmutableSandboxedEnvironment

tpl_path, out_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 700
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 20261110)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
    x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
tpl = env.from_string(open(tpl_path).read())

R = rng.random
words = "alpha beta gamma delta file path query city units temp name id value result list item count size mode".split()
uni = ["é", "日本語", "😀", "👨‍👩‍👧", "Привет", "مرحبا", " ", " ", " ", "​", "﻿", "\u0085", "\x7f", "\x1b", "\U0001F600", "ß", "İ", "　"]
esc = ['"', "\\", "\n", "\r", "\t", "\b", "\f", "/", "'", "\x00", "\x01", "\x1f", "</script>", "&", "<", ">", "\\u00e9", "\\n", '\\"']
ws_lead = ["", "", "", " ", "\n", "\t", " ", "　", "\x1c", "\x1f", "\x85", " ", "​", "﻿", "  \n ", " "]

def text(maxw=8):
    parts = []
    for _ in range(rng.randint(0, maxw)):
        r = R()
        if r < 0.55: parts.append(rng.choice(words))
        elif r < 0.7: parts.append(rng.choice(uni))
        elif r < 0.85: parts.append(rng.choice(esc))
        else: parts.append(rng.choice([" ", "  ", "\n", "\n\n", "\t", ": ", ", ", "<think>", "</think>", "<tool_call>", "<|im_end|>", "<|im_start|>user"]))
    return "".join(parts) if R() < 0.5 else " ".join(parts)

def number():
    r = R()
    if r < 0.3: return rng.randint(-10, 1000)
    if r < 0.4: return rng.choice([0, 1, -1, 2**31, 2**63, -2**63, 2**64 + 5, 10**30, 123456789012345678901234567890])
    if r < 0.5: return rng.choice([0.1, 0.5, 1.0, -0.0, 0.0, 100.0, 3.14159, 1e-5, 1e-4, 1.5e-7, 1e15, 1e16, 1.5e16, 123456789.125, 1e22, 1.7976931348623157e308, 5e-324, 2.5e-10, 0.30000000000000004, 1/3, 2/3, 1e100])
    if r < 0.55: return rng.choice([float("nan"), float("inf"), float("-inf")])
    return round(rng.uniform(-1000, 1000), rng.randint(0, 12))

def value(depth=0):
    r = R()
    if depth >= 3 or r < 0.35: 
        k = R()
        if k < 0.4: return text(4)
        if k < 0.65: return number()
        if k < 0.8: return rng.choice([True, False])
        return None
    if r < 0.65:
        return {rng.choice(words) + rng.choice(["", "_x", "é", "\"q", "\n", " k"]): value(depth + 1) for _ in range(rng.randint(0, 4))}
    return [value(depth + 1) for _ in range(rng.randint(0, 4))]

def schema(depth=0):
    r = R()
    t = rng.choice(["string", "integer", "number", "boolean", "array", "object"]) if depth < 3 else rng.choice(["string", "integer", "boolean"])
    s = {"type": t}
    if R() < 0.6: s["description"] = text(6)
    if t == "string" and R() < 0.4: s["enum"] = [text(2) for _ in range(rng.randint(1, 3))]
    if t == "integer" and R() < 0.4: s["minimum"] = rng.randint(0, 5); s["default"] = number()
    if t == "array": s["items"] = schema(depth + 1)
    if t == "object":
        props = {rng.choice(words) + str(i): schema(depth + 1) for i in range(rng.randint(0, 4))}
        s["properties"] = props
        if props and R() < 0.6: s["required"] = rng.sample(list(props), rng.randint(0, len(props)))
        if R() < 0.3: s["additionalProperties"] = rng.choice([True, False])
    return s

def tool():
    fn = {"name": rng.choice(words) + "_" + rng.choice(words), "description": text(8), "parameters": schema(0) if R() < 0.9 else {}}
    if R() < 0.2: fn["strict"] = True
    if R() < 0.85: return {"type": "function", "function": fn}
    return fn  # bare function definition, also valid input for tojson

def call():
    args = value(1) if R() < 0.15 else {rng.choice(words): value(1) for _ in range(rng.randint(0, 4))}
    if R() < 0.1: args = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)  # arguments given as a JSON string
    name = rng.choice(words) + "_" + rng.choice(words)
    r = R()
    if r < 0.6: return {"type": "function", "id": "call_" + str(rng.randint(0, 999)), "function": {"name": name, "arguments": args}}
    if r < 0.9: return {"name": name, "arguments": args}
    return {"id": "x", "function": {"name": name, "arguments": args}, "name": "ignored"}

def content():
    r = R()
    if r < 0.12: return ""
    if r < 0.17: return None
    s = rng.choice(ws_lead) + text(10)
    if R() < 0.12: s = rng.choice(ws_lead) + "<think>\n" + text(4) + "\n</think>\n" + text(3)
    return s

def reasoning():
    r = R()
    if r < 0.3: return ("absent",)
    if r < 0.4: return None
    if r < 0.5: return ""
    return text(10) + rng.choice(["", "\n", " "])

def conversation(edge):
    msgs = []
    if R() < 0.6: msgs.append({"role": "system", "content": text(8)})
    for _ in range(rng.randint(1, 6)):
        r = R()
        if r < 0.3 or not msgs or msgs[-1]["role"] == "system":
            msgs.append({"role": "user", "content": text(10) if R() > 0.08 else ""})
        elif r < 0.7:
            m = {"role": "assistant"}
            c = content()
            if R() < 0.1: pass  # content key missing
            else: m["content"] = c
            rc = reasoning()
            if rc != ("absent",): m["reasoning_content"] = rc
            if R() < 0.5: m["tool_calls"] = [call() for _ in range(rng.randint(1, 3))]
            elif R() < 0.05: m["tool_calls"] = []
            elif R() < 0.05: m["tool_calls"] = None
            msgs.append(m)
        elif r < 0.9:
            for _ in range(rng.randint(1, 3)):
                m = {"role": "tool", "content": text(6) if R() < 0.85 else rng.choice(["", json.dumps(value(1), ensure_ascii=False)])}
                if R() < 0.4: m["tool_call_id"] = "call_1"
                msgs.append(m)
        else:
            msgs.append({"role": rng.choice(["system", "system", "developer", "function", "ipython"]), "content": text(5)})
    if edge:
        k = rng.randint(0, 11)
        if k == 0 and msgs: msgs[0]["content"] = None
        elif k == 1: msgs.append({"role": "tool", "content": None})
        elif k == 2: msgs.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        elif k == 3: msgs = []
        elif k == 4: msgs.append({"content": "no role"})
        elif k == 5: msgs.append({"role": "assistant", "content": "x", "reasoning_content": 5})
        elif k == 6: msgs.append({"role": "assistant", "content": "x", "tool_calls": [{"function": {"arguments": {}}}]})
        elif k == 7: msgs.append({"role": "assistant", "content": "x", "tool_calls": [{"function": {"name": "f"}}]})
        elif k == 8: msgs.append({"role": "assistant", "tool_calls": [{"name": "f", "arguments": {}}]})   # content missing
        elif k == 9: msgs.append({"role": "tool"})
        elif k == 10: msgs.append({"role": "assistant", "content": 0, "tool_calls": [{"name": "f", "arguments": 1}]})
        else: msgs.append({"role": "assistant", "content": ["a"], "tool_calls": [{"name": "f", "arguments": {}}]})
    return msgs

def build(edge):
    inp = {"messages": conversation(edge)}
    r = R()
    if r < 0.5: inp["tools"] = [tool() for _ in range(rng.randint(1, 3))]
    elif r < 0.6: inp["tools"] = rng.choice([[], None])
    r = R()
    if r < 0.45: inp["add_generation_prompt"] = True
    elif r < 0.85: inp["add_generation_prompt"] = False
    r = R()
    if r < 0.2: inp["clear_thinking"] = True
    elif r < 0.35: inp["clear_thinking"] = False
    return inp

n_err = 0
with open(out_path, "w", encoding="utf-8") as o:
    for i in range(N):
        edge = i % 8 == 7
        inp = build(edge)
        try:
            rendered, err = tpl.render(**inp), None
        except Exception as e:
            rendered, err = None, type(e).__name__; n_err += 1
        o.write(json.dumps({"id": i, "kind": "edge" if edge else "valid", "input": inp, "rendered": rendered, "error": err}, ensure_ascii=False) + "\n")
print(N, "conversations,", n_err, "raise in jinja2 ->", out_path)
