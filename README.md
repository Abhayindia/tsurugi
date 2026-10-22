<h1 align="center">Tsurugi</h1>

<p align="center">
  A UCI chess engine written from scratch in C++20, with an NNUE evaluation and an SPRT-tested search.
</p>

<p align="center">
  <a href="https://github.com/Abhayindia/tsurugi/actions/workflows/ci.yml"><img src="https://github.com/Abhayindia/tsurugi/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg" alt="License: GPL-3.0"></a>
  <a href="https://lichess.org/@/romeogonerogue"><img src="https://img.shields.io/badge/lichess-romeogonerogue-black.svg" alt="Lichess BOT"></a>
</p>

## Overview

Tsurugi is a chess engine that speaks the [Universal Chess Interface](https://www.chessprogramming.org/UCI) and runs in
any UCI GUI (Cute Chess, Arena, BanksiaGUI, En Croissant, ...). It does not include a graphical interface.

Every change that affects playing strength is accepted only after a sequential probability ratio test (SPRT) against
the previous version; the full record is in [docs/elo_history.md](docs/elo_history.md).

## Features

**Board and move generation**
- Bitboards with PEXT (BMI2) or magic-bitboard sliding attacks
- Legal move generation verified by perft to depth 6-7 on the standard test positions

**Search**
- Iterative deepening, principal variation search, aspiration windows
- Transposition table with prefetching, quiescence search
- Null-move pruning, late move reductions, late move pruning
- Reverse futility pruning, futility pruning, razoring
- Move ordering by TT move, MVV-LVA, killer moves and history

**Evaluation**
- NNUE: 768 → 2×1536 → 1 network with clipped-ReLU activations, int16 quantisation and incrementally
  updated accumulators, trained with Tsurugi's own PyTorch trainer ([scripts/nnue](scripts/nnue))
- A hand-crafted tapered evaluation (material, piece-square tables, mobility, king safety, rook files) is used when
  no network is loaded

## Strength

| Change | Elo gain (SPRT, 5+0.05) |
|---|---|
| NNUE evaluation (network v1) | +241.6 ± 36.9 |
| Network v2 over v1 | +97.1 ± 13.4 |
| Network v3 over v2 (8+0.08) | +50.2 ± 8.1 |
| Network v4 over v3 (8+0.08) | +47.9 ± 8.2 |
| Network v5 over v4 (8+0.08) | +10.4 ± 5.5 |
| Network v6 over v5 (8+0.08) | +12.1 ± 7.5 |
| Network v7 over v6 (8+0.08) | +27.2 ± 7.5 |
| Late move reductions | +144.8 ± 27.7 |
| Null-move pruning | +137.6 ± 28.1 |
| Aspiration windows | +78.3 ± 21.7 |
| Mobility evaluation | +77.8 ± 21.3 |

Gains are measured against the version before each change, so they are not additive.

**Calibration:** 3157 ± 19 Elo with network v6 (3123 ± 18 with v5, 3066 ± 18 with v4, 3030 ± 19 with v3, 2941 ± 19 with v2) on Stockfish's `UCI_Elo` scale (1200 games at 10+0.1 against Stockfish 17.1 limited to
2850, 3000 and 3190). That scale is calibrated by Stockfish at longer time controls, so treat it as an
approximation rather than a rating-list figure.

**Ablation:** how much the current engine loses when one technique is switched off (3200 games each, 8+0.08):

| Technique removed | Elo lost |
|---|---|
| NNUE evaluation | 272 ± 14 |
| Late move reductions | 155 ± 10 |
| SEE pruning | 112 ± 9 |
| Reverse futility pruning | 71 ± 8 |
| Late move pruning | 69 ± 8 |
| Aspiration windows | 53 ± 8 |
| Null-move pruning | 34 ± 8 |
| Futility pruning | 14 ± 8 |
| Counter-move heuristic | 14 ± 8 |
| Check extension | 13 ± 8 |
| Razoring | 7 ± 8 |

Techniques overlap: null-move pruning was worth +138 when it was added to a bare search, but only 34 once the
other pruning methods are present.

## Building

Requires CMake 3.16+ and a C++20 compiler (GCC 11+ or Clang 14+).

```
git clone https://github.com/Abhayindia/tsurugi.git
cd tsurugi
cmake -S . -B build -DTSURUGI_NATIVE=ON
cmake --build build -j
./build/tsurugi
```

`-DTSURUGI_NATIVE=ON` tunes the binary for the build machine and uses PEXT where available. Leave it off for a
portable binary. The default network is copied next to the binary as `tsurugi.nnue` and loaded at start-up.

## UCI options

| Option | Default | Description |
|---|---|---|
| `Hash` | 16 | Transposition table size in MB (1-4096) |
| `Threads` | 1 | Search threads (single-threaded for now) |
| `Move Overhead` | 50 | Milliseconds reserved per move for GUI or network latency |
| `UseNNUE` | true | Use the neural network evaluation |
| `EvalFile` | `tsurugi.nnue` | Network file to load |

The remaining options switch individual search and evaluation techniques on or off; they exist for testing and
ablation studies and should normally be left at their defaults.

Non-standard commands: `bench [depth]` (node-count signature), `perft <depth>`, `d` (print FEN), `eval`.

## Testing

```
./build/perft_test --quick          # perft on the standard positions (full suite: no argument)
./build/eval_test                   # evaluation symmetry and static exchange evaluation
bash tests/uci_smoke.sh ./build/tsurugi
./build/tsurugi bench 8
```

Strength changes are tested with [fastchess](https://github.com/Disservin/fastchess) on the UHO_Lichess opening
book using SPRT (elo0 = 0, elo1 = 5, α = β = 0.05). Tests run locally and on GitHub Actions
([.github/workflows/sprt.yml](.github/workflows/sprt.yml)).

## Networks

Released networks are in [nets/](nets), each with a JSON file describing its training data and loss history. They are
trained on Stockfish evaluations of positions from the [Lichess evaluation database](https://database.lichess.org/#evals)
(CC0); see [docs/data-sources.md](docs/data-sources.md). The trainer and the Kaggle notebook used to train them are
in [scripts/nnue](scripts/nnue).

## Acknowledgements

- The [Chess Programming Wiki](https://www.chessprogramming.org/), the reference for most techniques used here.
- [Stockfish](https://github.com/official-stockfish/Stockfish), studied for ideas and used as a test opponent.
  Tsurugi contains no Stockfish code. See [CREDITS.md](CREDITS.md).
- [Lichess](https://lichess.org/) for its open databases, and [fastchess](https://github.com/Disservin/fastchess).

## License

Tsurugi is free software, distributed under the [GNU General Public License v3.0](LICENSE).
