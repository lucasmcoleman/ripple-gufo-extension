#!/usr/bin/env bash
# Every GPU gate of the ricursive extension, in dependency order, one invocation
# per GPU-lock acquisition, one log per invocation, one table at the end.
#
#   tests/run-gpu-gates.sh            run everything
#   LIST=1 tests/run-gpu-gates.sh     print the invocations, run nothing
#   ONLY='attention|kernel' tests/run-gpu-gates.sh   only "gate:invocation" names matching the regex
#
# Gates and what each needs first (a gate runs only when every prerequisite
# PASSED; the table still lists the rest as NOT-RUN):
#   kernel_ops        -
#   attention_ops     kernel_ops          (small, real-sliding/full x f32/f16)
#   projection_ops    -                   (every shape of tests/project/run-gpu-tests.sh)
#   embed             -                   (embed_test gpu: host lookup + device copy)
#   executor_test     kernel_ops attention_ops projection_ops embed   (every filter)
#   executor_contract executor_test       (contract suite on the GPU backend)
#   engine_tests      executor_contract   (engine_test, continuation_test and draft_session_test
#                                          (docs/DRAFTER-S3.md: Session::Verify / AcceptVerified / taps, 4
#                                          invocations: verify and rest of tinyA and tinyB) with the GPU backend,
#                                          RICURSIVE_TEST_BACKEND=gpu-exact: F32 caches, no activation
#                                          rounding, so they stay comparable with the scalar Reference;
#                                          their twins still run on the CPU executor. output_parser,
#                                          chat_e2e and the rest are CPU-only by design)
#   engine_logits     executor_contract   (ricursive_gpu_logits on tinyA/tinyB, 4 sequences each:
#                                          --kv f32 --exact vs llama.cpp F32 (2e-3, top-1 100%) and
#                                          --kv f16 vs llama.cpp --kv-f16 (2e-2, top-1 100%), scored by
#                                          tools/score_logits.py; tests/gate-logits.sh)
#   engine_prompt     engine_logits      (`gufo prompt` smoke with the GPU backend on the tiny chat
#                                          fixture; tests/gate-prompt.sh)
#
# Rules: never a GPU program outside the lock wrapper:
#   GPU_ADMISSION_LOCK=/server/state/gpu-admission/gpu.lock gpu-lock-exec -w WAIT -- CMD
# An exit 75 (lock or memory gate refused) is recorded as LOCKED(75), is never
# retried and ends the run's attempts (the remaining invocations are listed as
# LOCKED(75) "not tried"): waiting out the lock once per invocation would only
# stack waits. Nothing is unloaded and nothing is killed except the gate's own
# test process when it exceeds INVOCATION_TIMEOUT while holding the lock.
#
# Environment: BUILD_DIR (directory holding the ricursive_*_test binaries),
# GPU_LOCK_WAIT (seconds the wrapper waits for admission, default 120),
# INVOCATION_TIMEOUT (seconds a test may run, default 180; the design budget is
# 20 s), GPU_LOCK_EXEC (wrapper), GPU_ADMISSION_LOCK, GATES_OUT (log root).
#
# Statuses: PASS | FAIL | SKIP(77) (the test found no device) | LOCKED(75) |
# NOT-RUN (a prerequisite did not PASS).
# Exit: 0 all PASS; 1 at least one FAIL; 2 no FAIL but something unverified.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BUILD_DIR=${BUILD_DIR:-/server/programming/gufo-ripple/build-test/extensions/build/ricursive/build}
export GUFO_BIN=${GUFO_BIN:-/server/programming/gufo-ripple/build-ric-release/gufo}
export CHAT_FIXTURE=${CHAT_FIXTURE:-$BUILD_DIR/chat-tiny.gguf}
WAIT=${GPU_LOCK_WAIT:-120}
INV_TIMEOUT=${INVOCATION_TIMEOUT:-180}
LOCK_EXEC=${GPU_LOCK_EXEC:-/server/staging/gpu-serialization-20260929/gpu-lock-exec}
export GPU_ADMISSION_LOCK=${GPU_ADMISSION_LOCK:-/server/state/gpu-admission/gpu.lock}
OUT_ROOT=${GATES_OUT:-/server/staging/ripple-gufo/gpu-gates}
ONLY=${ONLY:-}

GATES=(kernel_ops attention_ops projection_ops embed executor_test executor_contract
       engine_tests engine_logits engine_prompt)
declare -A DEPS=(
  [kernel_ops]=""
  [attention_ops]="kernel_ops"
  [projection_ops]=""
  [embed]=""
  [executor_test]="kernel_ops attention_ops projection_ops embed"
  [executor_contract]="executor_test"
  [engine_tests]="executor_contract"
  [engine_logits]="executor_contract"
  [engine_prompt]="engine_logits"
)

# gate -> lines "label|binary|arguments|NAME=value ..." ("binary" is relative to BUILD_DIR,
# "@name" is a helper script next to this one, an absolute path is used as is; the lists are fixed here on
# purpose: asking a GPU binary for its list would run it outside the lock).
invocations() {
  case $1 in
    kernel_ops)
      echo "all|ricursive_kernel_ops_test||" ;;
    attention_ops)
      for m in small real-sliding-f32 real-sliding-f16 real-full-f32 real-full-f16; do
        echo "$m|ricursive_attention_ops_test|$m|"
      done ;;
    projection_ops)
      # tests/project/run-gpu-tests.sh (`--list`) and extension.cmake
      for c in q4k_10240x5120 q8_0_1024x5120 q6k_5120x10240 q4k_40960x5120 q5k_40960x5120 \
               q4k_5120x40960 q5k_5120x40960 q6k_100352x5120 f32-tiny twins reject \
               route_q4k_47x16640 route_q4k_47x16384 route_q4k_47x16128 route_q4k_5123x16640 route_q4k_5123x16384 \
               route_q4k_5123x16128 route_q5k_47x16640 route_q5k_47x16384 route_q5k_47x16128 route_q5k_5123x16640 \
               route_q5k_5123x16384 route_q5k_5123x16128 \
               route_q4k_1x256 route_q5k_1x256 route_q4k_1x512 route_q5k_1x512 route_q4k_7x256 route_q5k_7x256 route_q4k_7x512 route_q5k_7x512; do
        echo "$c|ricursive_projection_ops_test|$c|"
      done ;;
    embed)
      echo "gpu|ricursive_embed_test|gpu|" ;;
    executor_test)
      for i in taps:tinyA taps:tinyB llama-f32:tinyA llama-f32:tinyB llama-f16:tinyA llama-f16:tinyB \
               chunk:tinyA chunk:tinyB rewind:tinyA rewind:tinyB twins:tinyA twins:tinyB limits:tinyA \
               quant:tinyB-Q4_K_M quant:tinyB-Q5_K_M quant:tinyB-Q6_K quant:tinyB-Q8_0; do
        echo "$i|ricursive_executor_gpu_test|$i|"
      done ;;
    executor_contract)
      # The contract test's main() has no filter; the backends file honours this variable.
      for b in gpu-f32-exact gpu-f16; do
        echo "$b|ricursive_executor_contract_gpu_test||RICURSIVE_CONTRACT_BACKEND=$b"
      done ;;
    engine_tests)
      echo "engine|ricursive_engine_test||RICURSIVE_TEST_BACKEND=gpu-exact"
      echo "continuation|ricursive_continuation_test||RICURSIVE_TEST_BACKEND=gpu-exact"
      # docs/DRAFTER-S3.md; "verify" = check 1 (k 1..16 x every a across W, 2W+3 and the ring wrap),
      # "rest" = checks 2 to 6 (cycles, taps, twins, snapshots, refusals). Long: see perf/STAGE3-GPU.md.
      for g in verify rest; do
        for f in tinyA tinyB; do
          echo "draft-$g-$f|ricursive_draft_session_test|$g:$f|RICURSIVE_TEST_BACKEND=gpu-exact"
        done
      done ;;
    engine_logits)
      for m in f32 f16; do
        echo "tinyA-$m|@gate-logits.sh|tinyA.gguf tinyA $m|"
        echo "tinyB-$m|@gate-logits.sh|tinyB.gguf tinyB $m|"
      done ;;
    engine_prompt)
      echo "gufo-prompt|@gate-prompt.sh||" ;;
  esac
}

if [ "${LIST:-}" = 1 ]; then
  for g in "${GATES[@]}"; do
    while IFS='|' read -r label bin args envs; do
      [ -n "$label" ] && printf '%-18s %-22s %s %s %s\n' "$g" "$label" "$bin" "$args" "$envs"
    done < <(invocations "$g")
  done
  exit 0
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="$OUT_ROOT/$STAMP"
mkdir -p "$LOGDIR" || { echo "cannot create $LOGDIR" >&2; exit 64; }
echo "gpu gates $STAMP  logs: $LOGDIR  build: $BUILD_DIR  wrapper: $LOCK_EXEC  wait ${WAIT}s"

rows=()          # tab separated: gate, invocation, status, seconds, note
declare -A GATE_STATUS=()
locked_seen=0
n=0
any_fail=0
any_unverified=0

first_line() {  # log, status -> the first telling line, one line, trimmed
  local log=$1 status=$2 line=""
  case $status in
    LOCKED*) line=$(grep -m1 'gpu-lock-exec' "$log") ;;
    FAIL*)   line=$(grep -m1 -E '^FAIL' "$log")  # the test's own verdict first: "error" also appears in passing lines
             [ -z "$line" ] && line=$(grep -m1 -E 'FAIL|rror|what\(\)|terminate|Segmentation|timed out' "$log")
             [ -z "$line" ] && line=$(tail -n 1 "$log") ;;
    SKIP*)   line=$(grep -m1 -i 'skip' "$log") ;;
  esac
  printf '%s' "$line" | tr '\t\r' '  ' | cut -c1-160
}

add_row() { rows+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"); }

for gate in "${GATES[@]}"; do
  blocked=""
  for dep in ${DEPS[$gate]}; do
    # A gate the ONLY filter left out is not a prerequisite to wait for.
    case "${GATE_STATUS[$dep]:-}" in PASS|FILTERED) ;; *) blocked="$blocked $dep" ;; esac
  done
  gate_fail=0
  gate_pass=0
  gate_total=0
  while IFS='|' read -r label bin args envs; do
    [ -z "$label" ] && continue
    if [ -n "$ONLY" ] && ! grep -Eq -- "$ONLY" <<<"$gate:$label"; then continue; fi
    gate_total=$((gate_total + 1))
    if [ -n "$blocked" ]; then
      add_row "$gate" "$label" NOT-RUN 0 "prerequisite not PASS:$blocked"
      any_unverified=1
      continue
    fi
    if [ $locked_seen -eq 1 ]; then
      add_row "$gate" "$label" "LOCKED(75)" 0 "not tried: the lock was refused earlier in this run (no retry)"
      any_unverified=1
      continue
    fi
    n=$((n + 1))
    log="$LOGDIR/$(printf '%02d' "$n")-$gate-${label//[^A-Za-z0-9._-]/_}.log"
    case $bin in
      /*) exe=$bin ;;
      @*) exe=${GATE_HELPER_DIR:-$HERE}/${bin#@} ;;
      *)  exe=$BUILD_DIR/$bin ;;
    esac
    if [ ! -x "$exe" ]; then
      echo "binary not built: $exe" >"$log"
      add_row "$gate" "$label" FAIL 0 "binary not built: $exe"
      gate_fail=1
      any_fail=1
      continue
    fi
    start=$(date +%s%N)
    # shellcheck disable=SC2086
    env OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} $envs "$LOCK_EXEC" -w "$WAIT" -- \
      timeout --kill-after=10 "$INV_TIMEOUT" "$exe" $args >"$log" 2>&1
    rc=$?
    secs=$(awk -v a="$start" -v b="$(date +%s%N)" 'BEGIN{printf "%.1f", (b-a)/1e9}')
    case $rc in
      0)   status=PASS; gate_pass=$((gate_pass + 1)) ;;
      75)  status="LOCKED(75)"; locked_seen=1; any_unverified=1 ;;
      77)  status="SKIP(77)"; any_unverified=1 ;;
      124|137) status=FAIL; gate_fail=1; any_fail=1; echo "timed out after ${INV_TIMEOUT}s" >>"$log" ;;
      *)   status=FAIL; gate_fail=1; any_fail=1
           # a crash prints nothing: say so, or the table shows the lock line
           [ "$rc" -gt 128 ] && echo "FAIL: killed by signal $((rc - 128)) (exit $rc)" >>"$log" ;;
    esac
    note=$(first_line "$log" "$status")
    add_row "$gate" "$label" "$status" "$secs" "$note"
    echo "  [$gate:$label] $status ${secs}s"
  done < <(invocations "$gate")
  if [ $gate_total -eq 0 ]; then GATE_STATUS[$gate]=FILTERED
  elif [ $gate_fail -eq 1 ]; then GATE_STATUS[$gate]=FAIL
  elif [ $gate_pass -eq $gate_total ]; then GATE_STATUS[$gate]=PASS
  else GATE_STATUS[$gate]=UNVERIFIED; fi
done

{
  printf '\n%-18s %-24s %-11s %7s  %s\n' GATE INVOCATION STATUS SECONDS "FIRST FAILURE LINE / NOTE"
  for r in "${rows[@]}"; do
    IFS=$'\t' read -r g i s t note <<<"$r"
    printf '%-18s %-24s %-11s %7s  %s\n' "$g" "$i" "$s" "$t" "$note"
  done
  echo
  for g in "${GATES[@]}"; do printf 'gate %-18s %s\n' "$g" "${GATE_STATUS[$g]}"; done
  echo "logs: $LOGDIR"
} | tee "$LOGDIR/summary.txt"

if [ $any_fail -eq 1 ]; then exit 1; fi
if [ $any_unverified -eq 1 ]; then exit 2; fi
exit 0
