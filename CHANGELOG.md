# Changelog

All strength figures are SPRT results against the previous version (fastchess, UHO_Lichess_4852_v1,
5+0.05 unless noted), standard logistic Elo with 95% error.

## Unreleased

### Evaluation
- NNUE evaluation (768 → 2×256 → 1, clipped ReLU, int16, incremental accumulators), on by default.
  Network v1 (30M Lichess positions): +241.6 ± 36.9 over the hand-crafted evaluation (444 games).
- Network v2 (60M positions, 20 epochs, trained on a Kaggle GPU): +97.1 ± 13.4 over v1 (1600 games).
- Network v3 (120M Lichess positions, 25 epochs, trained on a Kaggle GPU): +50.2 ± 8.1 over v2
  (3200 games, 8+0.08).
- Network v4 (768 → 2×512 → 1, 160M Lichess positions, 20 epochs, trained on a Kaggle GPU):
  +47.9 ± 8.2 over v3 (3200 games, 8+0.08).
- Network v5 (768 → 2×1024 → 1, 220M Lichess positions, 15 epochs, trained on a Kaggle GPU) is the default:
  +10.4 ± 5.5 over v4 (6400 games, 8+0.08). The default build now uses a 1024-neuron hidden layer.
- Hand-crafted evaluation terms (used without a network): mobility +77.8 ± 21.3, rook on open and semi-open files
  +37.6 ± 15.0, king safety +18.1 ± 10.0. Passed pawns (−18.8 ± 11.8) was rejected. Pawn structure failed on top
  of five terms (−7.3 ± 7.9) but passed a 12800-game retest on top of nine (+5.8 ± 4.6, 8+0.08); it stays off by
  default because the network replaces the hand-crafted evaluation.
- Reworked passed pawns (halved bonuses plus an endgame king-distance term to the stop square), on by default
  for the hand-crafted evaluation: +33.5 ± 13.5 (1600 games, 8+0.08, without a network).

### Search
- Late move reductions +144.8 ± 27.7 (492 games).
- Null-move pruning +137.6 ± 28.1 (518 games).
- SEE pruning of losing captures and quiets +109.0 ± 23.4 (474 games).
- Aspiration windows +78.3 ± 21.7 (528 games).
- Reverse futility pruning +32.6 ± 13.9 (1644 games).
- Futility pruning +29.3 ± 13.1 (1582 games).
- Late move pruning +19.8 ± 10.6 (2528 games).
- Razoring +10.7 ± 7.3 (5650 games).
- Counter-move heuristic (reply to the previous move ordered after killers) +10.5 ± 7.2 (4416 games).
- Check extension +5.7 ± 4.6 (12376 games).
- Transposition-table move searched before move generation +9.3 ± 6.6 (4800 games, 8+0.08).
- Continuation history (1- and 2-ply) for quiet move ordering and late move reductions +15.3 ± 7.6 (3200 games, 8+0.08).
- Singular extensions with multi-cut and negative extensions +36.4 ± 7.3 (3200 games, 8+0.08).
- Pawn-structure correction history of the static evaluation +15.3 ± 7.7 (3200 games, 8+0.08).
- Time management from the share of nodes on the best root move, best-move stability and score drops +29.5 ± 7.3 (3200 games, 8+0.08).
- Expected cut nodes passed down the search (harder late move reductions, null move only at cut nodes) +7.1 ± 5.2 (6400 games, 8+0.08).
- ProbCut +7.6 ± 5.2 (6400 games, 8+0.08).
- Double singular extensions (capped per line) and stronger negative extensions +7.7 ± 5.0 (6400 games, 8+0.08).

### Engine
- AVX2 kernel for the network output and single-pass accumulator updates: the same search in roughly half the
  CPU time with the 1024-neuron network (identical evaluations and node counts). Actions builds now use the
  runner's native instruction set.
- Fixed: a `stop` that arrived before the search thread started was lost, so the engine could search forever
  and ignore later commands (seen under heavy CPU load).
- Time management scales with the time control and never starts an iteration it cannot finish.
- PEXT slider attacks on BMI2 builds, transposition-table prefetching, incremental material and
  piece-square totals.
- The default network is copied next to the binary at build time and loaded on start-up.
- `datagen <games> <nodes> <seed> <file>`: self-play at a fixed node budget from random openings, writing quiet
  positions with the search score and the game result for network training.

### Foundations
- Bitboard board representation, magic bitboards, legal move generation, make/unmake, Zobrist hashing.
- Perft suite exact to depth 6 on all standard positions (depth 7 for position 3).
- UCI protocol: `uci`, `isready`, `ucinewgame`, `position`, `go` (`wtime/btime/winc/binc/movestogo/depth/nodes/
  movetime/infinite`), `stop`, `quit`, `setoption`.
- Baseline search: iterative deepening, PVS, quiescence search, transposition table, MVV-LVA, killer and history
  move ordering; tapered hand-crafted evaluation.
