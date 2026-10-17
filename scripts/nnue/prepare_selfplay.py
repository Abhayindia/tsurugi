"""Turn `tsurugi datagen` output into NNUE training shards.

Usage: python scripts/nnue/prepare_selfplay.py --out ~/tsurugi-data/selfplay <files...> [--shard 1000000]
Input lines are "fen | white-view cp | white-view result (1.0/0.5/0.0)", plain or gzipped. The shards have the
same fields as prepare_data.py plus
  result    int8[N]       game result from White's point of view: 0 = loss, 1 = draw, 2 = win
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
    return feats, 0 if fields[1] == "w" else 1, int(cp), RESULTS[result]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=1_000_000)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    buf, shard_no, kept, bad = [], 0, 0, 0

    def flush():
        nonlocal buf, shard_no
        if not buf:
            return
        f, s, c, r = zip(*buf)
        np.savez(out / f"shard_{shard_no:04d}.npz", features=np.stack(f), stm=np.array(s, dtype=np.int8),
                 cp=np.array(c, dtype=np.int16), result=np.array(r, dtype=np.int8))
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
    (out / "meta.json").write_text(json.dumps({"source": a.files, "kept": kept, "bad": bad, "shards": shard_no},
                                              indent=1))
    print(f"done: {kept} positions, {bad} bad lines")


if __name__ == "__main__":
    main()
