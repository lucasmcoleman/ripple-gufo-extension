#!/bin/sh
# Regenerates the oracle side of the stage 1 test (tests/drafter_cpu_test.cpp, docs/DRAFTER-S1.md): the stage
# files of the tiny drafters. The drafter GGUFs and the tap dump (perf/stage1/tiny/{drafter-*.gguf,dump}) are
# written by tests/fixtures_text/make_drafter_fixture.py (--taps 4) and a layer 0..3 cut of a ricursive_dump_taps --backend cpu
# dump (commands: docs/DRAFTER-S1.md, "Regenerating").
# usage: tools/drafter_stage1_fixtures.sh [DIR]     DIR defaults to /server/staging/ripple-gufo/perf/stage1/tiny
set -e
HERE=$(dirname "$0")
T=${1:-/server/staging/ripple-gufo/perf/stage1/tiny}
TARGET=${RICURSIVE_TINY_TARGET:-/server/staging/ripple-gufo/fixtures/tinyB.gguf}
# q8_0: a walk over 79 positions (8..86); q4_k: a short walk, for the Q4_K loading path.
"$HERE/drafter_oracle.sh" stages --draft "$T/drafter-q8_0.gguf" --target "$TARGET" --dump "$T/dump" \
  --first 8 --last 86 --out "$T/stages-q8_0.bin"
"$HERE/drafter_oracle.sh" stages --draft "$T/drafter-q4_k.gguf" --target "$TARGET" --dump "$T/dump" \
  --first 60 --last 75 --out "$T/stages-q4_k.bin"
