#!/usr/bin/env bash
# UCI protocol and time-management smoke test. Usage: tests/uci_smoke.sh <engine>
set -u
ENGINE="$1"
fail=0

# Waits for the engine to answer isready (so process startup is excluded), sends the commands,
# keeps stdin open for `hold` seconds, and reports ms from the commands to the first line matching pattern.
check() {  # name, commands, max_ms, pattern, hold_seconds
    local start out line_ts ms
    out=$({ echo isready; sleep 1; date +%s%N >&3; printf "%b" "$2"; sleep "${5:-3}"; echo quit; } 3>/tmp/uci_smoke_t0 |
          timeout 30 "$ENGINE" | while IFS= read -r l; do echo "$(date +%s%N) $l"; done)
    start=$(cat /tmp/uci_smoke_t0)
    line_ts=$(grep -E "^[0-9]+ $4" <<<"$out" | awk -v s="$start" '$1 >= s' | head -1 | cut -d' ' -f1)
    if [ -z "$line_ts" ]; then echo "FAIL $1: missing /$4/"; fail=1; return; fi
    ms=$(( (line_ts - start) / 1000000 ))
    if [ "$ms" -gt "$3" ]; then echo "FAIL $1: took ${ms}ms > $3ms"; fail=1; return; fi
    echo "ok   $1 (${ms}ms)"
}

check uciok        "uci\n"                                                   1000 "uciok" 1
check readyok      "isready\n"                                               1000 "readyok" 1
check movetime     "position startpos\ngo movetime 500\n"                    700  "bestmove [a-h][1-8][a-h][1-8]" 2
check lowclock     "position startpos moves e2e4\ngo wtime 300 btime 300\n"  300  "bestmove [a-h][1-8]" 2
check clockinc     "position startpos\ngo wtime 5000 btime 5000 winc 100 binc 100\n" 2600 "bestmove" 4
check movestogo1   "position startpos\ngo wtime 1000 btime 1000 movestogo 1\n" 600 "bestmove" 2
check depth        "position startpos\ngo depth 5\n"                         3000 "bestmove" 3
check nodes        "position startpos\ngo nodes 20000\n"                     3000 "bestmove" 3
check mate_in_1    "position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1\ngo depth 4\n" 3000 "info .*score mate 1 " 2
check stalemate    "position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\ngo depth 3\n" 3000 "bestmove 0000" 2
check options      "setoption name Hash value 64\nsetoption name NullMove value true\nisready\n" 2000 "readyok" 1
check illegal_mv   "position startpos moves e2e5\n"                          1000 "info string illegal move e2e5" 1

# go infinite must keep searching until stop arrives.
out=$({ printf "position startpos\ngo infinite\n"; sleep 1.5; echo stop; sleep 0.5; echo quit; } | timeout 10 "$ENGINE")
if grep -q "^bestmove" <<<"$out" && grep -q "info depth" <<<"$out"; then echo "ok   infinite+stop"; else echo "FAIL infinite+stop"; fail=1; fi

exit $fail
