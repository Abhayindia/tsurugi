"""Train Tsurugi's NNUE (768 -> 2x256 -> 1, clipped ReLU) on shards from prepare_data.py and export it.

Usage: python scripts/nnue/train.py --data ~/tsurugi-data/nnue --out nets/tsurugi-v1.nnue [--epochs 20] [--hidden 256]
Target: sigmoid(cp_stm / SCALE); prediction: sigmoid(net); MSE loss. Export quantises the feature
transformer by QA and the output layer by QB (format documented in src/nnue.h).
"""
import argparse
import glob
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCALE, QA, QB = 400.0, 255, 64
PAD = 768
WEIGHT_CLIP = 127 / QB  # keeps every quantised weight inside int16 and the output sum inside int32


def views(f):
    """Feature indices from White's and Black's point of view (Black's: colours swapped, ranks flipped)."""
    valid = f >= 0
    c, rest = f // 384, f % 384
    t, sq = rest // 64, rest % 64
    black = (1 - c) * 384 + t * 64 + (sq ^ 56)
    return np.where(valid, f, PAD).astype(np.int64), np.where(valid, black, PAD).astype(np.int64)


class Net(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.ft = nn.EmbeddingBag(PAD + 1, hidden, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, 1)
        nn.init.normal_(self.ft.weight, std=0.01)

    def forward(self, white, black, stm):
        w = self.ft(white) + self.ft_bias
        b = self.ft(black) + self.ft_bias
        s = stm.unsqueeze(1).bool()
        us, them = torch.where(s, b, w), torch.where(s, w, b)
        return self.out(torch.clamp(torch.cat([us, them], 1), 0, 1)).squeeze(1)


def load(data, limit):
    feats, stm, cp, res = [], [], [], []
    for path in sorted(glob.glob(str(Path(data).expanduser() / "shard_*.npz"))):
        d = np.load(path)
        feats.append(d["features"]), stm.append(d["stm"]), cp.append(d["cp"])
        # Game result from White's view (0/1/2); shards without one (Stockfish labels) get -1.
        res.append(d["result"] if "result" in d.files else np.full(len(d["cp"]), -1, dtype=np.int8))
        if limit and sum(len(x) for x in cp) >= limit:
            break
    f, s, c = np.concatenate(feats), np.concatenate(stm), np.concatenate(cp).astype(np.float32)
    r = np.concatenate(res).astype(np.float32)
    c_stm = np.where(s == 0, c, -c)
    r_stm = np.where(r < 0, -1.0, np.where(s == 0, r, 2 - r) / 2).astype(np.float32)
    n = limit or len(c)
    return f[:n], s[:n], c_stm[:n], r_stm[:n]


def export(net, path, hidden):
    ft_w = np.round(net.ft.weight.detach().cpu().numpy()[:PAD] * QA).astype(np.int16)  # [768, H]
    ft_b = np.round(net.ft_bias.detach().cpu().numpy() * QA).astype(np.int16)           # [H]
    out_w = np.round(net.out.weight.detach().cpu().numpy()[0] * QB).astype(np.int16)    # [2H]
    out_b = int(round(float(net.out.bias.detach().cpu()[0]) * QA * QB))
    with open(path, "wb") as fh:
        fh.write(b"TSRG")
        fh.write(struct.pack("<II", 1, hidden))
        fh.write(ft_w.tobytes())
        fh.write(ft_b.tobytes())
        fh.write(out_w.tobytes())
        fh.write(struct.pack("<i", out_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--wdl", type=float, default=0.0, help="weight of the game result in the target (0 = score only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    dev = torch.device(a.device)
    print(f"training on {dev}", flush=True)

    f, s, c, r = load(a.data, a.limit)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(c))
    n_val = int(len(c) * a.val)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"{len(train_idx)} training / {n_val} validation positions", flush=True)

    net = Net(a.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, a.epochs // 3), gamma=0.3)

    def batch(idx):
        w, b = views(f[idx])
        return (torch.from_numpy(w).to(dev), torch.from_numpy(b).to(dev),
                torch.from_numpy(s[idx].astype(np.int64)).to(dev),
                make_target(idx))

    def make_target(idx):
        score = torch.sigmoid(torch.from_numpy(c[idx]).to(dev) / SCALE)
        result = torch.from_numpy(r[idx]).to(dev)
        return torch.where(result < 0, score, (1 - a.wdl) * score + a.wdl * result)

    def val_loss():
        net.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(val_idx), a.batch):
                w, b, st, target = batch(val_idx[i:i + a.batch])
                losses.append(torch.mean((torch.sigmoid(net(w, b, st)) - target) ** 2).item())
        net.train()
        return float(np.mean(losses))

    history = []
    for epoch in range(a.epochs):
        t0, tot, nb = time.time(), 0.0, 0
        rng.shuffle(train_idx)
        for i in range(0, len(train_idx), a.batch):
            w, b, st, target = batch(train_idx[i:i + a.batch])
            loss = torch.mean((torch.sigmoid(net(w, b, st)) - target) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                net.ft.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
                net.out.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
            tot, nb = tot + loss.item(), nb + 1
        sched.step()
        history.append({"epoch": epoch + 1, "train_loss": tot / nb, "val_loss": val_loss(),
                        "seconds": round(time.time() - t0)})
        print(json.dumps(history[-1]), flush=True)
        export(net, a.out, a.hidden)

    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"data": a.data, "positions": len(c), "hidden": a.hidden, "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "wdl": a.wdl,
         "scale": SCALE, "qa": QA, "qb": QB, "history": history}, indent=1))


if __name__ == "__main__":
    main()
