#!/usr/bin/env bash
# Runs every ricursive kernel GPU test binary through the GPU admission lock,
# one binary per lock acquisition, and prints a pass/fail/skip table.
#   usage: tests/kernels/run-gpu-tests.sh [BUILD_DIR]   (default: ./build-kernels)
# Exit status: 0 when nothing failed (skips are not failures, but a table with
# skips verifies nothing for those rows), 1 when any test failed.
# Exit 75 from the wrapper = lock or memory gate refused (a resident model or
# Scriberr holds it): reported as SKIP(lock), never retried here and never
# worked around. Exit 77 = the test itself found no GPU.
set -u
BUILD=${1:-build-kernels}
LOCK_EXEC=${GPU_LOCK_EXEC:-/server/staging/gpu-serialization-20260929/gpu-lock-exec}
export GPU_ADMISSION_LOCK=${GPU_ADMISSION_LOCK:-/server/state/gpu-admission/gpu.lock}
WAIT=${GPU_LOCK_WAIT:-120}

TESTS=(
  "kernel_ops|ricursive_kernel_ops_test"
  "attention small shapes|ricursive_attention_ops_test small"
  "attention real sliding F32|ricursive_attention_ops_test real-sliding-f32"
  "attention real sliding F16|ricursive_attention_ops_test real-sliding-f16"
  "attention real full F32|ricursive_attention_ops_test real-full-f32"
  "attention real full F16|ricursive_attention_ops_test real-full-f16"
)

fail=0
rows=()
for entry in "${TESTS[@]}"; do
  name=${entry%%|*}
  cmd=${entry#*|}
  bin=${cmd%% *}
  args=${cmd#"$bin"}
  start=$(date +%s)
  # shellcheck disable=SC2086
  "$LOCK_EXEC" -w "$WAIT" -- "$BUILD/$bin" $args > "$BUILD/$bin.${args// /_}.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - start ))
  case $rc in
    0)  status=PASS ;;
    75) status="SKIP(lock)" ;;
    77) status="SKIP(no GPU)" ;;
    *)  status=FAIL; fail=1 ;;
  esac
  rows+=("$(printf '%-30s %-14s rc=%-3s %4ss' "$name" "$status" "$rc" "$secs")")
done
echo
echo "test                           status         rc      time"
printf '%s\n' "${rows[@]}"
echo "(logs: $BUILD/*.log)"
exit $fail
