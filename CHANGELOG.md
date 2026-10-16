# Changelog

## Unreleased

- Bitboard board representation, magic bitboards, legal move generation, make/unmake, Zobrist hashing.
- Perft suite: all standard positions exact to depth 6 (depth 7 for position 3).
- UCI protocol with time management (`wtime/btime/winc/binc/movestogo/movetime/depth/nodes/infinite`).
- Baseline search: iterative deepening, PVS, quiescence search, transposition table,
  MVV-LVA, killer and history move ordering.
- Hand-crafted tapered evaluation: material, piece-square tables, bishop pair.
- Runtime toggles for each search technique, all off until SPRT-tested.
