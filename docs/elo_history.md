# Elo history

Each row is one SPRT test of a single change against the previous version (fastchess, UHO_Lichess_4852_v1 openings, logistic Elo, 95% error).

| Date | Commit | Test | Change | TC | Games | W-L-D | Elo | LLR | Result |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-17 | 0fc77a1 | sanity_selfplay | new binary | 8+0.08 | 200 | 66-81-53 | -31.51 ± 48.15 | ? | inconclusive |
| 2026-10-17 | b5a7bb9 | nullmove | NullMove=true | 5+0.05 | 518 | 307-112-99 | 162.84 ± 29.92 | 2.96 | H1 |
| 2026-10-17 | c1c24a3 | lmr | LMR=true NullMove=true vs NullMove=true | 5+0.05 | 492 | 275-81-136 | 180.63 ± 30.70 | 2.94 | H1 |
| 2026-10-17 | 401f0cf | rfp | RFP=true NullMove=true LMR=true vs NullMove=true LMR=true | 5+0.05 | 1644 | 637-483-524 | 39.62 ± 16.79 | 2.95 | H1 |
| 2026-10-17 | 7642bc6 | evalpassedpawns | EvalPassedPawns=true NullMove=true LMR=true RFP=true vs NullMove=true LMR=true RFP=true | 5+0.05 | 2252 | 736-858-658 | -23.04 ± 14.35 | 0.09 | H0 |
| 2026-10-17 | 302983c | evalkingsafety | EvalKingSafety=true NullMove=true LMR=true RFP=true vs NullMove=true LMR=true RFP=true | 5+0.05 | 3714 | 1611-1418-685 | 20.2 ± 11.17 | 2.97 | H1 |
| 2026-10-17 | 302983c | evalmobility | EvalMobility=true NullMove=true LMR=true RFP=true EvalKingSafety=true vs NullMove=true LMR=true RFP=true EvalKingSafety=true | 5+0.05 | 740 | 350-187-203 | 94.51 ± 25.03 | 2.97 | H1 |
| 2026-10-17 | 11f9078 | evalpawnstructure | EvalPawnStructure=true NullMove=true LMR=true RFP=true EvalKingSafety=true EvalMobility=true vs NullMove=true LMR=true RFP=true EvalKingSafety=true EvalMobility=true | 5+0.05 | 4640 | 1549-1646-1445 | -9.16 ± 10.00 | ? | H0 |
| 2026-10-17 | 11f9078 | evalrookfiles | EvalRookFiles=true NullMove=true LMR=true RFP=true EvalKingSafety=true EvalMobility=true vs NullMove=true LMR=true RFP=true EvalKingSafety=true EvalMobility=true | 5+0.05 | 1216 | 473-342-401 | 49.3 ± 19.53 | 2.96 | H1 |
