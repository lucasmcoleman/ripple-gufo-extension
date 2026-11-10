#!/usr/bin/env bash
# Runs every GPU invocation of work package A through the GPU admission lock, one invocation per lock
# acquisition, and prints a PASS/FAIL/SKIP table. Never runs a GPU test outside the lock.
#
#   BUILD_DIR   build directory of tests/project (default /server/staging/ripple-gufo/s2-project/build)
#   WAIT        seconds gpu-lock-exec waits for admission per attempt (default 120)
#   RETRY_SLEEP seconds before the second attempt after a refusal (exit 75), default 240
#   ONLY        optional grep pattern: run only invocations whose name matches
# Exit status: 0 if nothing failed (SKIPs allowed), 1 if any invocation FAILED.
set -u
BUILD_DIR=${BUILD_DIR:-/server/staging/ripple-gufo/s2-project/build}
WAIT=${WAIT:-120}
RETRY_SLEEP=${RETRY_SLEEP:-240}
LOCK_EXEC=/server/staging/gpu-serialization-20260929/gpu-lock-exec
export GPU_ADMISSION_LOCK=/server/state/gpu-admission/gpu.lock
PROJ="$BUILD_DIR/ricursive_projection_ops_test"
EMB="$BUILD_DIR/ricursive_embed_test"
LOGDIR=${LOGDIR:-$BUILD_DIR/gpu-test-logs}
mkdir -p "$LOGDIR"

# name|command (each under 20 s of run time and 4 GB of GPU memory by construction)
mapfile -t CASES < <(
  { "$PROJ" --list | sed "s|^|projection_ops $PROJ |" ; echo "embed_device $EMB gpu"; } )

names=(); results=(); times=(); notes=()
for entry in "${CASES[@]}"; do
  name=${entry%% *}; rest=${entry#* }
  if [ "$name" = projection_ops ]; then
    bin=${rest%% *}; arg=${rest#* }; name="projection_ops:$arg"; cmd=("$bin" "$arg")
  else
    cmd=($rest)
  fi
  if [ -n "${ONLY:-}" ] && ! grep -q -- "$ONLY" <<<"$name"; then continue; fi
  log="$LOGDIR/${name//[:\/ ]/_}.log"
  result=FAIL; note=""
  start=$(date +%s)
  for attempt in 1 2; do
    "$LOCK_EXEC" -w "$WAIT" -- "${cmd[@]}" >"$log" 2>&1
    rc=$?
    if [ $rc -eq 75 ]; then
      note="lock or memory gate refused (exit 75), attempt $attempt"
      [ $attempt -eq 1 ] && sleep "$RETRY_SLEEP"
      continue
    fi
    break
  done
  secs=$(( $(date +%s) - start ))
  case $rc in
    0)  result=PASS; note=$(grep -m1 -o 'done in [0-9.]* s' "$log" || true) ;;
    75) result=SKIP; note="not run on GPU: lock held ($note)" ;;
    77) result=SKIP; note="no GPU device" ;;
    *)  result=FAIL; note="exit $rc, see $log: $(grep -m1 'FAIL' "$log")" ;;
  esac
  names+=("$name"); results+=("$result"); times+=("${secs}s"); notes+=("$note")
done

printf '\n%-44s %-5s %-6s %s\n' INVOCATION RESULT WALL NOTE
fail=0
for i in "${!names[@]}"; do
  printf '%-44s %-5s %-6s %s\n' "${names[$i]}" "${results[$i]}" "${times[$i]}" "${notes[$i]}"
  [ "${results[$i]}" = FAIL ] && fail=1
done
exit $fail
