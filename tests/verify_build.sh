#!/usr/bin/env bash
# Builds <build-dir> and runs every check that must pass before a binary is deployed.
# Usage (inside tools/wsl_env.sh): bash tests/verify_build.sh build-bot
set -uo pipefail
DIR="${1:-build-bot}"
cd "$(dirname "$0")/.."
cmake -S . -B "$DIR" -G Ninja -DTSURUGI_NATIVE=ON > /dev/null || exit 1
cmake --build "$DIR" 2>&1 | grep -E "error|warning" && { echo "BUILD: warnings or errors"; exit 1; }
fail=0
"./$DIR/perft_test" --quick | tail -1 | grep -q "ALL PASSED" && echo "perft:   ok" || { echo "perft:   FAIL"; fail=1; }
"./$DIR/eval_test" | tail -1 | grep -q "ALL PASSED" && echo "eval:    ok" || { echo "eval:    FAIL"; fail=1; }
bash tests/uci_smoke.sh "./$DIR/tsurugi" > /tmp/smoke.txt && echo "uci:     ok" || { echo "uci:     FAIL"; cat /tmp/smoke.txt; fail=1; }
echo "time per move by time control:"
bash tests/tm_check.sh "./$DIR/tsurugi" | sed 's/^/         /'
echo "bench:   $("./$DIR/tsurugi" bench 8)"
exit $fail
