#!/usr/bin/env bash
# Tests tests/run-gpu-gates.sh with a fake lock wrapper and fake test binaries
# (shell scripts), so the PASS / FAIL / SKIP(77) / LOCKED(75) / NOT-RUN paths,
# the prerequisite cut-off, the no-retry rule and the timeout are exercised
# without touching the GPU. Prints one line per scenario; exit 0 when all hold.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GATES=$HERE/run-gpu-gates.sh
ROOT=${SELFTEST_DIR:-/server/staging/ripple-gufo/s2-exec/gates-selftest}
rm -rf "$ROOT"; mkdir -p "$ROOT/bin" "$ROOT/out"
CALLS=$ROOT/wrapper-calls.log

# Fake test binaries: behaviour from FAKE_FAIL / FAKE_SKIP / FAKE_SLEEP (regex on "name args").
for b in kernel_ops attention_ops projection_ops embed executor_gpu executor_contract_gpu; do :; done
mkdir -p "$ROOT/helpers"
for bin in ricursive_kernel_ops_test ricursive_attention_ops_test ricursive_projection_ops_test \
           ricursive_embed_test ricursive_executor_gpu_test ricursive_executor_contract_gpu_test \
           ricursive_engine_test ricursive_continuation_test; do
  cat >"$ROOT/bin/$bin" <<'FAKE'
#!/usr/bin/env bash
me="$(basename "$0") $* ${RICURSIVE_CONTRACT_BACKEND:-} ${RICURSIVE_TEST_BACKEND:-}"
[ -n "${FAKE_SLEEP:-}" ] && grep -Eq -- "$FAKE_SLEEP" <<<"$me" && sleep 30
if [ -n "${FAKE_SKIP:-}" ] && grep -Eq -- "$FAKE_SKIP" <<<"$me"; then echo "SKIP: no GPU device"; exit 77; fi
if [ -n "${FAKE_FAIL:-}" ] && grep -Eq -- "$FAKE_FAIL" <<<"$me"; then echo "FAIL: injected failure in $me"; exit 1; fi
echo PASS; exit 0
FAKE
  chmod +x "$ROOT/bin/$bin"
done
# Fake helper scripts (the real ones start GPU programs): same behaviour switches.
for h in gate-logits.sh gate-prompt.sh; do
  cp "$ROOT/bin/ricursive_embed_test" "$ROOT/helpers/$h"
  chmod +x "$ROOT/helpers/$h"
done

# Fake wrapper: logs every call; FAKE_LOCK_REFUSE (regex on the command line) -> exit 75 like the real one.
cat >"$ROOT/fake-lock-exec" <<'FAKE'
#!/usr/bin/env bash
[ "$1" = -w ] && shift 2
[ "$1" = -- ] && shift
echo "$*" >>"$FAKE_CALLS"
if [ -n "${FAKE_LOCK_REFUSE:-}" ] && grep -Eq -- "$FAKE_LOCK_REFUSE" <<<"$*"; then
  echo "gpu-lock-exec: refusing to start '$1': GPU admission lock held (fail-closed)" >&2
  exit 75
fi
exec "$@"
FAKE
chmod +x "$ROOT/fake-lock-exec"

fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1   [$2]"; fail=1; fi; }

run() {  # extra env assignments as arguments; output to $ROOT/last.txt, exit code in $RC
  : >"$CALLS"
  env BUILD_DIR="$ROOT/bin" GATE_HELPER_DIR="$ROOT/helpers" GPU_LOCK_EXEC="$ROOT/fake-lock-exec" FAKE_CALLS="$CALLS" \
      GATES_OUT="$ROOT/out" GPU_LOCK_WAIT=1 "$@" "$GATES" >"$ROOT/last.txt" 2>&1
  RC=$?
}
row() { grep -E "^$1 +$2 " "$ROOT/last.txt"; }

run
check "all pass: exit 0, 64 PASS rows" '[ $RC -eq 0 ] && [ "$(grep -cE " PASS +[0-9.]+  ?" "$ROOT/last.txt")" -ge 64 ] && ! grep -E "^[a-z_]+ +[A-Za-z0-9_:.-]+ +(FAIL|NOT-RUN|LOCKED|SKIP)" "$ROOT/last.txt" | grep -q .'
check "all pass: one wrapper call per invocation (64)" '[ "$(wc -l <"$CALLS")" -eq 64 ]'
check "all pass: one log per invocation + summary" '[ "$(ls "$(ls -d "$ROOT"/out/* | tail -1)" | wc -l)" -eq 65 ]'

run FAKE_FAIL='kernel_ops'
check "kernel_ops FAIL: exit 1" '[ $RC -eq 1 ]'
check "kernel_ops FAIL: first failure line shown" 'row kernel_ops all | grep -q "FAIL: injected failure in ricursive_kernel_ops_test"'
check "kernel_ops FAIL: attention, executor test and contract NOT-RUN" 'row attention_ops small | grep -q NOT-RUN && row executor_test taps:tinyA | grep -q NOT-RUN && row executor_contract gpu-f16 | grep -q NOT-RUN'
check "kernel_ops FAIL: projection_ops and embed still run and PASS" 'row projection_ops reject | grep -q " PASS " && row embed gpu | grep -q " PASS "'
check "kernel_ops FAIL: not-run gates never reach the wrapper (1+31+1 calls)" '[ "$(wc -l <"$CALLS")" -eq 33 ]'

run FAKE_FAIL='executor_gpu_test twins:tinyB'
check "one executor invocation FAIL: others PASS, contract NOT-RUN, exit 1" '[ $RC -eq 1 ] && row executor_test twins:tinyA | grep -q " PASS " && row executor_test twins:tinyB | grep -q " FAIL " && row executor_contract gpu-f16 | grep -q NOT-RUN'

run FAKE_FAIL='executor_contract_gpu_test'
check "contract FAIL: every engine gate NOT-RUN" '[ $RC -eq 1 ] && row engine_tests engine | grep -q NOT-RUN && row engine_tests continuation | grep -q NOT-RUN && row engine_logits tinyB-f16 | grep -q NOT-RUN && row engine_prompt gufo-prompt | grep -q NOT-RUN'

run FAKE_FAIL='engine_test.*gpu-exact'
check "engine test FAIL (backend env reaches it): logits and prompt gates still run" '[ $RC -eq 1 ] && row engine_tests engine | grep -q " FAIL " && row engine_tests continuation | grep -q " PASS " && row engine_logits tinyA-f32 | grep -q " PASS " && row engine_prompt gufo-prompt | grep -q " PASS "'

run FAKE_FAIL='gate-logits.sh tinyA.gguf tinyA f32'
check "one logits invocation FAIL: prompt smoke NOT-RUN, other logits PASS" '[ $RC -eq 1 ] && row engine_logits tinyA-f32 | grep -q " FAIL " && row engine_logits tinyB-f32 | grep -q " PASS " && row engine_prompt gufo-prompt | grep -q NOT-RUN'

run FAKE_SKIP='embed'
check "embed SKIP(77): exit 2, executor_test NOT-RUN" '[ $RC -eq 2 ] && row embed gpu | grep -q "SKIP(77)" && row executor_test limits:tinyA | grep -q NOT-RUN'

run FAKE_LOCK_REFUSE='.'
check "lock refused: exit 2, LOCKED(75) with the wrapper line" '[ $RC -eq 2 ] && row kernel_ops all | grep -q "LOCKED(75).*refusing to start"'
check "lock refused: exactly one wrapper call, no retry" '[ "$(wc -l <"$CALLS")" -eq 1 ]'
check "lock refused: independent gates not tried (LOCKED), dependents NOT-RUN" '[ "$(grep -Ec "^[a-z_]+ .*LOCKED\(75\)" "$ROOT/last.txt")" -eq 33 ] && [ "$(grep -Ec "^[a-z_]+ .* NOT-RUN" "$ROOT/last.txt")" -eq 31 ] && grep -q "not tried" "$ROOT/last.txt"'

run FAKE_LOCK_REFUSE='ricursive_embed_test'
check "lock lost mid-run: earlier gates PASS, embed LOCKED, dependents NOT-RUN, no retry" 'row kernel_ops all | grep -q " PASS " && row embed gpu | grep -q "LOCKED(75)" && row executor_test taps:tinyA | grep -q "NOT-RUN" && [ "$(wc -l <"$CALLS")" -eq 38 ]'

run FAKE_SLEEP='attention_ops_test small' INVOCATION_TIMEOUT=1
check "timeout: FAIL with timed out note" 'row attention_ops small | grep -q " FAIL " && row attention_ops small | grep -q "timed out"'

run ONLY='^embed:|^executor_contract:'
check "ONLY filter: just those invocations run; filtered prerequisites ignored" '[ "$(wc -l <"$CALLS")" -eq 3 ] && row executor_contract gpu-f16 | grep -q " PASS "'

mv "$ROOT/bin/ricursive_embed_test" "$ROOT/bin/hidden"
run ONLY='^embed:'
check "missing binary: FAIL, wrapper never called" '[ $RC -eq 1 ] && row embed gpu | grep -q "binary not built" && [ "$(wc -l <"$CALLS")" -eq 0 ]'
mv "$ROOT/bin/hidden" "$ROOT/bin/ricursive_embed_test"

run LIST=1
check "LIST=1: prints 64 invocations, runs nothing" '[ "$(grep -cE "^[a-z_]+ +[A-Za-z0-9_:.-]+ +(ricursive_|@)" "$ROOT/last.txt")" -eq 64 ] && [ "$(wc -l <"$CALLS")" -eq 0 ]'

[ $fail -eq 0 ] && echo "ALL SELFTESTS PASS" || echo "SELFTEST FAILURES"
exit $fail
