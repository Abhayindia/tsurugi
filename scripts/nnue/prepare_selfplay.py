"""Turn `tsurugi datagen` output into NNUE training shards.

Usage: python scripts/nnue/prepare_selfplay.py --out ~/tsurugi-data/selfplay <files...> [--shard 1000000]
Input lines are "fen | white-view cp | white-view result (1.0/0.5/0.0)", plain or gzipped. The shards have the
same fields as prepare_data.py plus
  result    int8[N]       game result from White's point of view: 0 = loss, 1 = draw, 2 = win
With --td-lambda L every score is replaced by a lambda-return over the later scores of the same game (datagen writes a
game's positions in order): in win-probability space, g[i] = (1 - L^d) * w[i] + L^d * g[i+1], d plies to the next
recorded position, and g = w at a game's last position. A later score is a search from deeper in the game.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

PIECES = {c: i for i, c in enumerate("PNBRQKpnbrqk")}
RESULTS = {"0.0": 0, "0.5": 1, "1.0": 2}


def parse(line):
    fen, cp, result = (x.strip() for x in line.split("|"))
    fields = fen.split()
    feats = np.full(32, -1, dtype=np.int16)
    i = 0
    for r, row in enumerate(fields[0].split("/")):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                feats[i] = PIECES[ch] * 64 + (7 - r) * 8 + f
                i, f = i + 1, f + 1
    return feats, 0 if fields[1] == "w" else 1, int(cp), RESULTS[result], 2 * int(fields[5]) + (fields[1] == "b"), i


def scale_cp(cp, scale):
    """Scores times `scale`, rounded; mate-range scores (|cp| >= 10000) are left as they are."""
    if scale == 1.0:
        return cp.astype(np.int16)
    return np.where(np.abs(cp) < 10000, np.rint(cp * scale), cp).astype(np.int16)


def td_lambda(cp, ply, pieces, result, lam):
    """Lambda-returns of one file's scores (already on the training scale), game by game."""
    w = 1.0 / (1.0 + np.exp(-np.clip(cp, -4000, 4000) / 400.0))
    g = w.copy()
    for i in range(len(w) - 2, -1, -1):
        d = ply[i + 1] - ply[i]
        # The next line belongs to another game when the ply does not advance, a piece reappears or the result differs.
        if d <= 0 or pieces[i + 1] > pieces[i] or result[i + 1] != result[i]:
            continue
        a = lam ** d
        g[i] = (1 - a) * w[i] + a * g[i + 1]
    g = np.clip(g, 1e-4, 1 - 1e-4)
    return np.where(np.abs(cp) < 4000, np.rint(400.0 * np.log(g / (1 - g))), cp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=1_000_000)
    ap.add_argument("--cp-scale", type=float, default=1.0,
                    help="multiply scores below 10000 by this (to put labels from different networks on one scale)")
    ap.add_argument("--td-lambda", type=float, default=0.0,
                    help="replace each score by its lambda-return over the game's later scores (0 = off)")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    buf, shard_no, kept, bad = [], 0, 0, 0

    def flush():
        nonlocal buf, shard_no
        if not buf:
            return
        f, s, c, r, ply, pieces = zip(*buf)
        cp = scale_cp(np.array(c), a.cp_scale)
        if a.td_lambda > 0:
            cp = td_lambda(cp.astype(np.float64), np.array(ply), np.array(pieces), np.array(r), a.td_lambda).astype(np.int16)
        np.savez(out / f"shard_{shard_no:04d}.npz", features=np.stack(f), stm=np.array(s, dtype=np.int8),
                 cp=cp, result=np.array(r, dtype=np.int8))
        print(f"shard {shard_no}: {len(buf)} positions", flush=True)
        buf, shard_no = [], shard_no + 1

    for path in a.files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    buf.append(parse(line))
                    kept += 1
                except (ValueError, KeyError, IndexError):
                    bad += 1
                if len(buf) == a.shard:
                    flush()
    flush()
    (out / "meta.json").write_text(json.dumps({"source": a.files, "kept": kept, "bad": bad, "shards": shard_no,
                                               "cp_scale": a.cp_scale, "td_lambda": a.td_lambda}, indent=1))
    print(f"done: {kept} positions, {bad} bad lines")


if __name__ == "__main__":
    main()
