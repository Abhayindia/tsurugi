#!/usr/bin/env bash
# Prints the time the engine spends on one middlegame move at each Lichess time control.
# Usage: tests/tm_check.sh <engine>
ENGINE="$1"
FEN="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
for tc in "60 0" "60 1" "180 0" "180 2" "300 3" "600 0" "600 5" "900 10" "1800 0" "1800 20"; do
    base=$(( ${tc% *} * 1000 )); inc=$(( ${tc#* } * 1000 ))
    start=$(date +%s%N)
    out=$({ printf "setoption name Hash value 128\nposition fen %s\ngo wtime %d btime %d winc %d binc %d\n" \
              "$FEN" "$base" "$base" "$inc" "$inc"; sleep 0.1; while ! grep -q ^bestmove /tmp/tm_out 2>/dev/null; do sleep 0.2; done; echo quit; } | "$ENGINE" | tee /tmp/tm_out)
    rm -f /tmp/tm_out
    ms=$(( ($(date +%s%N) - start) / 1000000 - 200 ))
    d=$(grep "info depth" <<<"$out" | tail -1 | grep -oE "info depth [0-9]+" | grep -oE "[0-9]+$")
    printf "%-8s move took ~%6d ms, depth %s\n" "${tc% *}s+${tc#* }" "$ms" "$d"
done
