#!/usr/bin/env bash
# Engine-level smoke: `gufo prompt` on the tiny chat fixture with the GPU backend
# (real tokenizer and chat template, random weights, so the text is gibberish).
# Run by tests/run-gpu-gates.sh THROUGH the GPU lock wrapper.
# Passes when gufo exits 0 and reports the ricursive GPU backend and its prompt.
set -u
GUFO=${GUFO_BIN:?GUFO_BIN must be set}
CHAT=${CHAT_FIXTURE:?CHAT_FIXTURE must be set}
[ -x "$GUFO" ] || { echo "FAIL: no gufo at $GUFO"; exit 1; }
[ -f "$CHAT" ] || { echo "FAIL: no chat fixture at $CHAT (ctest -R ricursive.chat_fixture builds it)"; exit 1; }
out=$(RICURSIVE_BACKEND=gpu "$GUFO" prompt --model "$CHAT" --prompt "What is 2+2?" \
        --max-tokens 16 --context-cap 256 --verbose 2>&1)
rc=$?
printf '%s\n' "$out"
if [ $rc -ne 0 ]; then echo "FAIL: gufo prompt exit $rc"; exit 1; fi
grep -q 'ricursive (GPU backend)' <<<"$out" || { echo "FAIL: output does not report the ricursive GPU backend"; exit 1; }
grep -q 'Prompt tokens: [1-9]' <<<"$out" || { echo "FAIL: no prompt tokens reported"; exit 1; }
exit 0
