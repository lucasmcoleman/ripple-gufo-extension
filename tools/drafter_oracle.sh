#!/bin/sh
# Runs tools/drafter_oracle.py the only way it may run: it executes the downloaded upstream model file, so
# the interpreter is isolated (-I), the environment is empty (no tokens, offline), CPU torch only.
# usage: tools/drafter_oracle.sh selfcheck|run|timing ARGS...   (see the script's docstring)
VENV=${ORACLE_VENV:-/server/staging/ripple-gufo/venv-oracle}
SCRATCH=${ORACLE_HOME:-/server/staging/ripple-gufo/oracle-home}
mkdir -p "$SCRATCH" || exit 1
exec nice -n "${ORACLE_NICE:-10}" env -i PATH=/usr/bin:/bin HOME="$SCRATCH" HF_HOME="$SCRATCH" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  "$VENV/bin/python" -I "$(dirname "$0")/drafter_oracle.py" "$@"
