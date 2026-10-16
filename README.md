# Tsurugi

Tsurugi is a UCI chess engine written from scratch in C++20, released under GPL-3.0.

Status: early development. Move generation is complete and perft-verified; the search and
hand-crafted evaluation are a baseline that will be improved one SPRT-tested change at a time,
followed by an NNUE evaluation trained on Tsurugi's own self-play data.

## Building

Requires CMake 3.16+ and a C++20 compiler (GCC 11+ or Clang 14+).

```
cmake -S . -B build -DTSURUGI_NATIVE=ON
cmake --build build
./build/tsurugi
```

`-DTSURUGI_NATIVE=OFF` (the default) builds a portable binary that needs only POPCNT.

## Testing

```
./build/perft_test --quick      # standard perft positions, shallow (CI)
./build/perft_test              # full suite, depth 6-7 (about 17 billion nodes)
tests/uci_smoke.sh ./build/tsurugi
./build/tsurugi bench 8         # node-count signature for each commit
```

Full perft results are in `paper/results/perft.json`.

## UCI options

| Option | Default | Notes |
|---|---|---|
| Hash | 16 | Transposition table size in MB |
| Threads | 1 | Single-threaded for now |
| Move Overhead | 50 | Milliseconds reserved per move for GUI/network lag |
| NullMove, LMR, RFP, Razoring, Futility, LMP, Aspiration, CheckExtension, SEEPruning, CounterMoves | see `uci` | Search technique toggles used for SPRT tests and the ablation study |

Extra commands: `bench [depth]`, `perft <depth>`, `go perft <depth>`, `d` (print FEN), `eval`.

## Project layout

```
src/     engine source
tests/   perft suite, UCI smoke test
tools/   match/SPRT scripts, perft cross-check, data generation and NNUE trainer
nets/    released network files
bot/     lichess-bot configuration template
paper/   research paper source and results
docs/    design notes and Elo history
```

## License

GPL-3.0. See `LICENSE` and `CREDITS.md`.
