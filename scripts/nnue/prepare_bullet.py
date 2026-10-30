"""Turn bullet-format chess records (32 bytes each, e.g. Lc0-derived training data) into NNUE training shards.

Usage: python scripts/nnue/prepare_bullet.py --out ~/tsurugi-data/lc0 [--limit N] [--shard 1000000] <file|->
The input is raw or zstd-compressed (.zst); "-" reads raw records from stdin. A record is stored from the side to
move's point of view: occupancy u64, 32 piece nibbles (bit 3 = opponent, low 3 bits = P N B R Q K) in square order,
score i16 and result u8 (0/1/2) for the side to move, our king square, the opponent's king square flipped, 3 spare
bytes. Each record becomes the same position with White to move, so the shards have the fields of
prepare_selfplay.py with stm = 0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

RECORD = 32


def convert(raw, max_cp=0):
    """Records -> (features int16[N, 32], cp int16[N], result int8[N], ok bool[N])."""
    rec = raw.reshape(-1, RECORD)
    n = len(rec)
    bits = np.unpackbits(rec[:, :8], axis=1, bitorder="little")
    pcs = rec[:, 8:24]
    nib = np.empty((n, 32), dtype=np.uint8)
    nib[:, 0::2], nib[:, 1::2] = pcs & 15, pcs >> 4
    count = bits.sum(axis=1, dtype=np.int64)
    rows, sqs = np.nonzero(bits)
    idx = np.arange(len(rows)) - np.repeat(np.cumsum(count) - count, count)
    pc = nib[rows, idx]
    kind, theirs = pc & 7, pc >> 3
    piece = theirs.astype(np.int16) * 6 + kind
    feats = np.full((n, 32), -1, dtype=np.int16)
    feats[rows, idx] = piece * 64 + sqs

    ok = (count <= 32) & (count >= 2)
    np.logical_and.at(ok, rows, kind <= 5)
    king = kind == 5
    ok &= np.bincount(rows[king & (theirs == 0)], minlength=n) == 1
    ok &= np.bincount(rows[king & (theirs == 1)], minlength=n) == 1
    our_k = np.full(n, 255, dtype=np.int64)
    opp_k = np.full(n, 255, dtype=np.int64)
    our_k[rows[king & (theirs == 0)]] = sqs[king & (theirs == 0)]
    opp_k[rows[king & (theirs == 1)]] = sqs[king & (theirs == 1)]
    ok &= (our_k == rec[:, 27]) & ((opp_k ^ 56) == rec[:, 28])
    result = rec[:, 26]
    ok &= result <= 2
    cp = rec[:, 24:26].copy().view(np.int16).ravel()
    if max_cp:
        ok &= np.abs(cp.astype(np.int32)) < max_cp
    return feats, cp, result.astype(np.int8), ok


def convert_kept(args):
    raw, max_cp = args
    feats, cp, result, ok = convert(raw, max_cp)
    return feats[ok], cp[ok], result[ok], int((~ok).sum())


def records(path, chunk):
    if path == "-":
        fh = sys.stdin.buffer
    elif path.endswith(".zst"):
        import zstandard
        fh = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
    else:
        fh = open(path, "rb")
    while True:
        buf = fh.read(chunk * RECORD)
        whole = len(buf) - len(buf) % RECORD
        if whole == 0:
            return
        yield np.frombuffer(buf[:whole], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=1_000_000)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many kept positions (0 = all)")
    ap.add_argument("--max-cp", type=int, default=0, help="drop positions with |score| >= this (0 = keep all)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("file")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    kept = bad = shard_no = 0
    pool = None
    if a.jobs > 1:
        from multiprocessing import Pool
        pool = Pool(a.jobs)

    def batches():
        # At most `jobs` chunks in memory at once.
        batch = []
        for raw in records(a.file, a.shard):
            batch.append((raw, a.max_cp))
            if len(batch) == a.jobs:
                yield from (pool.map(convert_kept, batch) if pool else map(convert_kept, batch))
                batch = []
        yield from (pool.map(convert_kept, batch) if pool else map(convert_kept, batch))

    for feats, cp, result, nbad in batches():
        bad += nbad
        if a.limit:
            k = min(len(cp), a.limit - kept)
            feats, cp, result = feats[:k], cp[:k], result[:k]
        if len(cp):
            np.savez(out / f"shard_{shard_no:04d}.npz", features=feats, stm=np.zeros(len(cp), dtype=np.int8),
                     cp=cp, result=result)
            print(f"shard {shard_no}: {len(cp)} positions", flush=True)
            shard_no += 1
            kept += len(cp)
        if a.limit and kept >= a.limit:
            break
    if pool:
        pool.terminate()
    (out / "meta.json").write_text(json.dumps({"source": [a.file], "kept": kept, "bad": bad, "shards": shard_no,
                                               "max_cp": a.max_cp}, indent=1))
    print(f"done: {kept} positions, {bad} dropped (invalid or |cp| >= max-cp)")


if __name__ == "__main__":
    main()
