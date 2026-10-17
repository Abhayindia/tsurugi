"""Turn the Lichess evaluation database into NNUE training shards.

Usage: python tools/nnue/prepare_data.py --out ~/tsurugi-data/nnue --positions 30000000 [--shard 1000000]
           [--src <path or https URL>] [--workers 4]
Streams the zstd-compressed JSONL (a local file, possibly still downloading, or straight from
https://database.lichess.org/lichess_db_eval.jsonl.zst). For each position it takes the deepest evaluation's
first line, keeps only quiet positions (side to move not in check, best move not a capture or promotion) with
|eval| <= 3000 cp and no forced mate, and stores:
  features  int16[N, 32]  piece-square index from White's view (colour*6 + type)*64 + square, padded with -1
  stm       int8[N]       0 = White to move, 1 = Black
  cp        int16[N]      Stockfish evaluation in centipawns from White's point of view
"""
import argparse
import io
import json
import urllib.request
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import zstandard

DEFAULT_SRC = str(Path.home() / "tsurugi-data" / "lichess_db_eval.jsonl.zst")
CHUNK = 20000


def features(board):
    out = np.full(32, -1, dtype=np.int16)
    for i, (sq, piece) in enumerate(board.piece_map().items()):
        out[i] = ((0 if piece.color == chess.WHITE else 1) * 6 + piece.piece_type - 1) * 64 + sq
    return out


def process(lines):
    feats, stms, cps, skipped = [], [], [], Counter()
    for line in lines:
        try:
            rec = json.loads(line)
            pv = max(rec["evals"], key=lambda e: e["depth"])["pvs"][0]
            board = chess.Board(rec["fen"] + " 0 1")
            first = chess.Move.from_uci(pv["line"].split()[0])
        except (ValueError, KeyError, IndexError):
            skipped["bad"] += 1
            continue
        if not board.is_valid():  # editor positions: >32 pieces, missing kings, pawns on back ranks...
            skipped["bad"] += 1
        elif "mate" in pv:
            skipped["mate"] += 1
        elif abs(pv["cp"]) > 3000:
            skipped["extreme"] += 1
        elif board.is_check():
            skipped["check"] += 1
        elif board.is_capture(first) or first.promotion:
            skipped["noisy"] += 1
        else:
            feats.append(features(board))
            stms.append(0 if board.turn == chess.WHITE else 1)
            cps.append(pv["cp"])
    return feats, stms, cps, skipped, len(lines)


def chunks(stream):
    buf = []
    try:
        for line in stream:
            buf.append(line)
            if len(buf) == CHUNK:
                yield buf
                buf = []
    except zstandard.ZstdError as e:
        print(f"stream ended early ({e}); keeping what was read", flush=True)
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=30_000_000)
    ap.add_argument("--shard", type=int, default=1_000_000)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    raw = urllib.request.urlopen(a.src) if a.src.startswith("http") else open(a.src, "rb")
    stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")

    feats, stms, cps = [], [], []
    kept = seen = shard_no = 0
    skipped = Counter()

    def flush():
        nonlocal feats, stms, cps, shard_no
        if not feats:
            return
        np.savez(out / f"shard_{shard_no:04d}.npz", features=np.stack(feats[:a.shard]),
                 stm=np.array(stms[:a.shard], dtype=np.int8), cp=np.array(cps[:a.shard], dtype=np.int16))
        print(f"shard {shard_no}: {min(len(feats), a.shard)} positions (seen {seen}, kept {kept}, "
              f"skipped {dict(skipped)})", flush=True)
        feats, stms, cps = feats[a.shard:], stms[a.shard:], cps[a.shard:]
        shard_no += 1

    with Pool(a.workers) as pool:
        for f, s, c, sk, n in pool.imap(process, chunks(stream)):
            seen += n
            skipped.update(sk)
            take = min(len(f), a.positions - kept)
            feats += f[:take]
            stms += s[:take]
            cps += c[:take]
            kept += take
            while len(feats) >= a.shard:
                flush()
            if kept >= a.positions:
                pool.terminate()
                break
    flush()
    raw.close()
    (out / "meta.json").write_text(json.dumps({"source": a.src, "seen": seen, "kept": kept, "skipped": dict(skipped),
                                               "shards": shard_no}, indent=1))
    print(f"done: kept {kept} of {seen}")


if __name__ == "__main__":
    main()
