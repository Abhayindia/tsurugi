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
