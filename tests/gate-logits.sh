#!/usr/bin/env bash
# One engine-level logits gate: ricursive_gpu_logits on one tiny fixture, all four
# sequences, scored against llama.cpp with tools/score_logits.py. Run by
# tests/run-gpu-gates.sh THROUGH the GPU lock wrapper (it starts a GPU program);
# never run it by hand without the wrapper.
#   gate-logits.sh FIXTURE_FILE ORACLE_PREFIX MODE
#   MODE f32: --kv f32 --exact vs <prefix>.seqN.llama.f32          (max abs 2e-3, top-1 100%)
#   MODE f16: --kv f16         vs <prefix>.seqN.llama-kvf16.f32    (max abs 2e-2, top-1 100%)
set -u
MODEL=$1 PREFIX=$2 MODE=$3
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:?BUILD_DIR must be set}
STAGE=${RICURSIVE_STAGE:-/server/staging/ripple-gufo}
PY=${SCORE_PYTHON:-$STAGE/venv/bin/python}
TOOL=$BUILD_DIR/ricursive_gpu_logits
case $MODE in
  f32) FLAGS=(--kv f32 --exact); SUFFIX=llama.f32;       MAXABS=2e-3 ;;
  f16) FLAGS=(--kv f16);         SUFFIX=llama-kvf16.f32; MAXABS=2e-2 ;;
  *) echo "mode must be f32 or f16" >&2; exit 64 ;;
esac
TMP=$(mktemp -d "${TMPDIR:-/tmp}/ricursive-gate-logits.XXXXXX") || exit 64
trap 'rm -rf "$TMP"' EXIT
rc=0
for seq in 0 1 2 3; do
  tokens=$STAGE/oracle/out/$PREFIX.seq$seq.i32
  ref=$STAGE/oracle/out/$PREFIX.seq$seq.$SUFFIX
  out=$TMP/seq$seq.f32
  echo "== $PREFIX seq$seq mode $MODE"
  RICURSIVE_BACKEND=gpu "$TOOL" "$STAGE/fixtures/$MODEL" "$tokens" "$out" "${FLAGS[@]}" >"$TMP/tool.txt" 2>&1
  trc=$?
  cat "$TMP/tool.txt"
  if [ $trc -ne 0 ]; then echo "FAIL: ricursive_gpu_logits exit $trc"; rc=1; continue; fi
  vocab=$(sed -n 's/.*vocab=\([0-9][0-9]*\).*/\1/p' "$TMP/tool.txt" | head -n1)
  "$PY" "$HERE/../tools/score_logits.py" "$ref" "$out" --vocab "$vocab" --max-abs "$MAXABS" --min-top1 1.0
  src=$?
  if [ $src -ne 0 ]; then echo "FAIL: score exit $src (limit max abs $MAXABS, top-1 100%)"; rc=1; fi
done
exit $rc
