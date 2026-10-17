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
- Null-move pruning enabled: SPRT +137.6 ± 28.1 Elo (518 games, 5+0.05).
- Late move reductions enabled: SPRT +144.8 ± 27.7 Elo (492 games, 5+0.05).
- Reverse futility pruning enabled: SPRT +32.6 ± 13.9 Elo (1644 games, 5+0.05).
- King safety evaluation enabled: SPRT +18.1 ± 10.0 Elo (3714 games, 5+0.05).
- Mobility evaluation enabled: SPRT +77.8 ± 21.3 Elo (740 games, 5+0.05).
- Passed-pawn evaluation tested and rejected: SPRT −18.8 ± 11.8 Elo (2252 games, 5+0.05).
- Pawn-structure evaluation tested and rejected: SPRT −7.3 ± 7.9 Elo (4640 games, 5+0.05).
- Rook open/semi-open file evaluation enabled: SPRT +37.6 ± 15.0 Elo (1216 games, 5+0.05).
- Futility pruning enabled: SPRT +29.3 ± 13.1 Elo (1582 games, 5+0.05).
- Late move pruning enabled: SPRT +19.8 ± 10.6 Elo (2528 games, 5+0.05).
- SPRT runner clears stale snapshots left by the old snapshot layout instead of crashing.
- Correction: SPRT Elo figures were previously recorded as fastchess's normalised Elo (nElo); all entries above now show standard logistic Elo (nElo kept in paper/results/sprt_*.json).
