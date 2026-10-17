# Contributing to Tsurugi

Thanks for your interest in improving Tsurugi.

## Reporting bugs

Open an issue with the engine version (`./build/tsurugi` prints it on `uci`), your OS and compiler, and the exact UCI
commands or position (FEN plus moves) that reproduce the problem.

## Code changes

- Fork the repository and branch from `main`.
- Keep each pull request to one change. Match the existing style (C++20, 4-space indent, short functions).
- `./build/perft_test --quick`, `./build/eval_test` and `bash tests/uci_smoke.sh ./build/tsurugi` must pass.
- State the `bench 8` node count in the pull request. A change that alters it changes search behaviour.

## Strength-affecting changes

Any change that alters the bench node count must pass an SPRT against current `main` before it is merged:
fastchess, UHO_Lichess_4852_v1 openings, 5+0.05 (or longer), elo0 = 0, elo1 = 5, α = β = 0.05. Include the final
fastchess summary (Elo, games, pentanomial counts, LLR) in the pull request.

Pure speed-ups must leave the bench node count unchanged and should report the nodes-per-second gain.

## Licence

By contributing you agree that your contribution is licensed under the GNU General Public License v3.0.
