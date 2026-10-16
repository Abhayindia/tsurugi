# Data sources

Every external dataset, book or binary used for testing, training or the paper. Large files are
downloaded by `tools/setup_testing.sh` and are not committed.

| Item | Use | Source | Licence |
|---|---|---|---|
| UHO_Lichess_4852_v1.epd | Opening positions for SPRT and gauntlet matches | https://github.com/official-stockfish/books | CC0-1.0 |
| 8moves_v3.pgn | Balanced openings, secondary book | https://github.com/official-stockfish/books | CC0-1.0 |
| fastchess | Match runner | https://github.com/Disservin/fastchess (alpha 1.8.2, ebfd64c) | MIT |
| Stockfish (commit recorded in `~/tsurugi-deps/bin/stockfish.commit`) | Calibration opponent and comparison target; not a code source | https://github.com/official-stockfish/Stockfish | GPL-3.0 |
| Perft node counts | Move generator verification | https://www.chessprogramming.org/Perft_Results | Reference values |
| python-chess | Independent perft reference | https://github.com/niklasf/python-chess | GPL-3.0 |
| Lichess evaluations (`lichess_db_eval.jsonl.zst`, 416,442,401 positions, 22.4 GB) | NNUE training labels (Stockfish evaluations at varying depth, mostly 20-40, computed in users' browsers) | https://database.lichess.org/#evals | CC0 |
| Lichess puzzles (`lichess_db_puzzle.csv.zst`, 6,157,341 puzzles) | Tactics benchmark: solve rate by puzzle rating | https://database.lichess.org/#puzzles | CC0 |
| Lichess broadcast games (`lichess_db_broadcast_2026-04..09.pgn.zst`) | Over-the-board games of titled players: move-agreement benchmark | https://database.lichess.org/#broadcasts | CC0 |

Training on Stockfish-labelled positions means the network learns to imitate Stockfish's judgement
(knowledge distillation). The engine's code stays original, but this is disclosed in the README and the
paper, and a network trained only on Tsurugi's own self-play labels is kept as a comparison.
