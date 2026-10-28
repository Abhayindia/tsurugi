"""Fit an uncertainty head on top of a frozen network: a second linear output that predicts how far the network's
own evaluation is from the training target in this position.

Usage: python scripts/nnue/fit_uncertainty.py --net nets/tsurugi-v6.nnue --data <shards> --out <net.nnue>
       [--positions 5000000] [--ridge 1.0]

The network's weights are left untouched, so its evaluation is unchanged; the output file (format version 3, or 6
for networks of version 4 and later) appends
  i16 aux_w[B][2H] | i32 aux_b[B] | i32 aux_q
and the engine computes the predicted error in centipawns as (dot(clamped hidden, aux_w[b]) + aux_b[b]) / aux_q,
using the same clipped-ReLU activations and output bucket as the evaluation. Because the activations are fixed,
the head is an ordinary least-squares fit per bucket, solved exactly.
"""
import argparse
import glob
import json
import struct
from pathlib import Path

import numpy as np
import torch

QA, QB, SCALE = 255, 64, 400
AUX_MAX = 1024  # largest quantised weight; keeps the engine's int32 sums far from overflow
ERROR_CAP = 500


def read_net(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"TSRG"
    version, hidden = struct.unpack_from("<II", raw, 4)
    buckets, o = (struct.unpack_from("<I", raw, 12)[0], 16) if version >= 2 else (1, 12)
    layout = act = has_aux = 0
    if version >= 4:
        layout = struct.unpack_from("<I", raw, o)[0]; o += 4
    if version >= 5:
        act = struct.unpack_from("<I", raw, o)[0]; o += 4
    if version == 6:
        has_aux = struct.unpack_from("<I", raw, o)[0]; o += 4
    assert version != 3 and not has_aux, "the network already has an uncertainty head"
    assert layout == 0, "king-bucket networks are not supported"
    start = o
    ft_w = np.frombuffer(raw, np.int16, 768 * hidden, o).reshape(768, hidden); o += 768 * hidden * 2
    ft_b = np.frombuffer(raw, np.int16, hidden, o); o += hidden * 2
    out_w = np.frombuffer(raw, np.int16, buckets * 2 * hidden, o).reshape(buckets, 2 * hidden); o += buckets * 4 * hidden
    out_b = np.array(struct.unpack_from(f"<{buckets}i", raw, o), dtype=np.int64); o += buckets * 4
    return dict(version=version, act=act, hidden=hidden, buckets=buckets, ft_w=ft_w, ft_b=ft_b, out_w=out_w,
                out_b=out_b, body=raw[start:o])


def black_view(f):
    c, rest = f // 384, f % 384
    t, sq = rest // 64, rest % 64
    return (1 - c) * 384 + t * 64 + (sq ^ 56)


def activations(net, feats, stm):
    """Clipped hidden activations [N, 2H] (side to move first), output bucket, and the network's eval in cp."""
    table = torch.cat([torch.from_numpy(net["ft_w"].astype(np.int32)),
                       torch.zeros(1, net["hidden"], dtype=torch.int32)])
    ft_b = torch.from_numpy(net["ft_b"].astype(np.int32))
    f = torch.from_numpy(feats.astype(np.int64))
    valid = f >= 0
    pad = torch.full_like(f, 768)
    white = torch.where(valid, f, pad)
    black = torch.where(valid, torch.from_numpy(black_view(feats.astype(np.int64))), pad)
    acc_w = table[white].sum(1) + ft_b
    acc_b = table[black].sum(1) + ft_b
    s = torch.from_numpy(stm.astype(np.bool_)).unsqueeze(1)
    us, them = torch.where(s, acc_b, acc_w), torch.where(s, acc_w, acc_b)
    x = torch.clamp(torch.cat([us, them], 1), 0, QA)
    pieces = valid.sum(1)
    B = net["buckets"]
    bucket = torch.clamp((pieces - 2) * B // 32, 0, B - 1) if B > 1 else torch.zeros_like(pieces)
    out_w = torch.from_numpy(net["out_w"].astype(np.int64))
    out_b = torch.from_numpy(net["out_b"])
    if net["act"] == 1:
        raw = (x.long() * x.long() * out_w[bucket]).sum(1) + out_b[bucket] * QA
        cp = torch.trunc(raw.double() * SCALE / (QA * QA * QB))
    else:
        raw = (x.long() * out_w[bucket]).sum(1) + out_b[bucket]
        cp = torch.trunc(raw.double() * SCALE / (QA * QB))
    return x, bucket, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=5_000_000)
    ap.add_argument("--val", type=int, default=200_000)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    net = read_net(a.net)
    H2, B = 2 * net["hidden"], net["buckets"]

    # Gram matrices of [activations, 1] per bucket, accumulated in float64. The first --val positions are held out.
    xtx = torch.zeros(B, H2 + 1, H2 + 1, dtype=torch.float64)
    xty = torch.zeros(B, H2 + 1, dtype=torch.float64)
    val, seen = [], 0
    for path in sorted(glob.glob(str(Path(a.data).expanduser() / "shard_*.npz"))):
        d = np.load(path)
        feats, stm, cp = d["features"], d["stm"], d["cp"].astype(np.float64)
        target = np.where(stm == 0, cp, -cp)
        for i in range(0, len(cp), a.batch):
            x, bucket, pred = activations(net, feats[i:i + a.batch], stm[i:i + a.batch])
            err = torch.clamp((pred - torch.from_numpy(target[i:i + a.batch])).abs(), max=ERROR_CAP)
            xa = torch.cat([x.double(), torch.ones(len(x), 1, dtype=torch.float64)], 1)
            if seen < a.val:
                val.append((x, bucket, err))
            else:
                for k in range(B):
                    m = bucket == k
                    if m.any():
                        xm = xa[m]
                        xtx[k] += xm.T @ xm
                        xty[k] += xm.T @ err[m]
            seen += len(x)
            if seen >= a.positions + a.val:
                break
        print(f"{path}: {seen} positions", flush=True)
        if seen >= a.positions + a.val:
            break

    reg = torch.eye(H2 + 1, dtype=torch.float64) * a.ridge
    reg[-1, -1] = 0
    beta = torch.stack([torch.linalg.solve(xtx[k] + reg, xty[k]) for k in range(B)])  # [B, 2H+1], cp per activation

    # Quantise: aux_w = round(beta * q), aux_b = round(beta0 * q); the engine divides the sum by q.
    q = int(AUX_MAX / beta[:, :-1].abs().max().item())
    aux_w = torch.round(beta[:, :-1] * q).to(torch.int16).numpy()
    aux_b = torch.round(beta[:, -1] * q).to(torch.int64).numpy()

    # Held-out check of the quantised head: does the predicted error track the actual error?
    w64, b64 = torch.from_numpy(aux_w.astype(np.int64)), torch.from_numpy(aux_b)
    p = torch.cat([torch.trunc(((x.long() * w64[bk]).sum(1) + b64[bk]).double() / q) for x, bk, _ in val])
    e = torch.cat([err for _, _, err in val])
    corr = float(torch.corrcoef(torch.stack([p, e]))[0, 1])
    lo, hi = p <= p.quantile(0.25), p >= p.quantile(0.75)
    report = {"positions": seen - a.val, "val_positions": len(e), "q": q, "corr": round(corr, 4),
              "mean_err": round(float(e.mean()), 1), "err_low_quartile": round(float(e[lo].mean()), 1),
              "err_high_quartile": round(float(e[hi].mean()), 1)}
    print(json.dumps(report), flush=True)

    with open(a.out, "wb") as fh:
        fh.write(b"TSRG")
        if net["version"] <= 2:
            fh.write(struct.pack("<III", 3, net["hidden"], B))
        else:
            fh.write(struct.pack("<IIIIII", 6, net["hidden"], B, 0, net["act"], 1))
        fh.write(net["body"])
        fh.write(aux_w.tobytes())
        fh.write(struct.pack(f"<{B}i", *aux_b.tolist()))
        fh.write(struct.pack("<i", q))
    Path(a.out).with_suffix(".json").write_text(json.dumps({"base": a.net, "data": a.data, "ridge": a.ridge, **report},
                                                           indent=1))


if __name__ == "__main__":
    main()
