"""Turn the Lichess evaluation database into NNUE training shards.

Usage: python tools/nnue/prepare_data.py --out ~/tsurugi-data/nnue --positions 30000000 [--shard 1000000]
Streams ~/tsurugi-data/lichess_db_eval.jsonl.zst (works on a partial download too). For each position
it takes the deepest evaluation's first line, keeps only quiet positions (side to move not in check,
best move not a capture or promotion) with |eval| <= 3000 cp and no forced mate, and stores:
  features  int16[N, 32]  piece-square index from White's view (colour*6 + type)*64 + square, padded with -1
  stm       int8[N]       0 = White to move, 1 = Black
  cp        int16[N]      Stockfish evaluation in centipawns from White's point of view
"""
import argparse
import io
import json
from pathlib import Path

import chess
import numpy as np
import zstandard

SRC = Path.home() / "tsurugi-data" / "lichess_db_eval.jsonl.zst"


def features(board):
    out = np.full(32, -1, dtype=np.int16)
    for i, (sq, piece) in enumerate(board.piece_map().items()):
        out[i] = ((0 if piece.color == chess.WHITE else 1) * 6 + piece.piece_type - 1) * 64 + sq
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=30_000_000)
    ap.add_argument("--shard", type=int, default=1_000_000)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    feats, stms, cps = [], [], []
    kept = seen = shard_no = 0
    skipped = {"mate": 0, "extreme": 0, "check": 0, "noisy": 0, "bad": 0}

    def flush():
        nonlocal feats, stms, cps, shard_no
        if not feats:
            return
        np.savez(out / f"shard_{shard_no:04d}.npz", features=np.stack(feats), stm=np.array(stms, dtype=np.int8),
                 cp=np.array(cps, dtype=np.int16))
        print(f"shard {shard_no}: {len(feats)} positions (seen {seen}, kept {kept}, skipped {skipped})", flush=True)
        feats, stms, cps = [], [], []
        shard_no += 1

    with SRC.open("rb") as fh:
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        try:
            for line in stream:
                seen += 1
                try:
                    rec = json.loads(line)
                    ev = max(rec["evals"], key=lambda e: e["depth"])
                    pv = ev["pvs"][0]
                    board = chess.Board(rec["fen"] + " 0 1")
                    first = chess.Move.from_uci(pv["line"].split()[0])
                except (ValueError, KeyError, IndexError):
                    skipped["bad"] += 1
                    continue
                if not board.is_valid():  # editor positions: >32 pieces, missing kings, pawns on back ranks...
                    skipped["bad"] += 1
                    continue
                if "mate" in pv:
                    skipped["mate"] += 1
                    continue
                if abs(pv["cp"]) > 3000:
                    skipped["extreme"] += 1
                    continue
                if board.is_check():
                    skipped["check"] += 1
                    continue
                if board.is_capture(first) or first.promotion:
                    skipped["noisy"] += 1
                    continue
                feats.append(features(board))
                stms.append(0 if board.turn == chess.WHITE else 1)
                cps.append(pv["cp"])
                kept += 1
                if len(feats) == a.shard:
                    flush()
                if kept >= a.positions:
                    break
        except zstandard.ZstdError as e:
            print(f"stream ended early ({e}); keeping what was read", flush=True)
    flush()
    (out / "meta.json").write_text(json.dumps({"source": str(SRC), "seen": seen, "kept": kept, "skipped": skipped,
                                               "shards": shard_no}, indent=1))
    print(f"done: kept {kept} of {seen}")


if __name__ == "__main__":
    main()
