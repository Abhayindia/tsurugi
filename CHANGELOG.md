# Changelog

## Unreleased

- Bitboard board representation, magic bitboards, legal move generation, make/unmake, Zobrist hashing.
- Perft suite: all standard positions exact to depth 6 (depth 7 for position 3).
- UCI protocol with time management (`wtime/btime/winc/binc/movestogo/movetime/depth/nodes/infinite`).
- Baseline search: iterative deepening, PVS, quiescence search, transposition table,
  MVV-LVA, killer and history move ordering.
- Hand-crafted tapered evaluation: material, piece-square tables, bishop pair.
- Runtime toggles for each search technique, all off until SPRT-tested.
- Time management scaled per time control; no new iteration once 40% of the move budget is used.
- Null-move pruning enabled: SPRT +162.8 ± 29.9 Elo (518 games, 5+0.05).
- Late move reductions enabled: SPRT +180.6 ± 30.7 Elo (492 games, 5+0.05).
- Reverse futility pruning enabled: SPRT +39.6 ± 16.8 Elo (1644 games, 5+0.05).
- King safety evaluation enabled: SPRT +20.2 ± 11.2 Elo (3714 games, 5+0.05).
- Mobility evaluation enabled: SPRT +94.5 ± 25.0 Elo (740 games, 5+0.05).
- Passed-pawn evaluation tested and rejected: SPRT −23.0 ± 14.4 Elo (2252 games, 5+0.05).
- Pawn-structure evaluation tested and rejected: SPRT −9.2 ± 10.0 Elo (4640 games, 5+0.05).
- Rook open/semi-open file evaluation enabled: SPRT +49.3 ± 19.5 Elo (1216 games, 5+0.05).
- SPRT runner clears stale snapshots left by the old snapshot layout instead of crashing.
