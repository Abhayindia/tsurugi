# Credits

Tsurugi's code is written from scratch. No source code from Stockfish or any other engine has been
copied or adapted. If that ever changes, each adapted file is listed here with the original file and
commit link, and marked in the source.

## Ideas and references

The techniques used are standard in computer chess and were learned from public descriptions:

- [Chess Programming Wiki](https://www.chessprogramming.org/): bitboards, magic bitboards,
  perft, principal variation search, quiescence search, transposition tables, null-move pruning,
  late move reductions, futility pruning, static exchange evaluation, history heuristics.
- [Stockfish](https://github.com/official-stockfish/Stockfish) (GPL-3.0): studied as a reference for
  search ideas and used as a calibration opponent and comparison target. Not a code source.

## Test data

- Perft node counts: Chess Programming Wiki,
  [Perft Results](https://www.chessprogramming.org/Perft_Results).
- python-chess (GPL-3.0) is used only as an independent reference in `tools/perft_check.py`.

Opening books, game databases and training data are recorded with source URL and licence in
`docs/data-sources.md` when they are added.
