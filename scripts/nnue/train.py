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

from shards import load

SCALE, QA, QB = 400.0, 255, 64
WEIGHT_CLIP = 127 / QB  # keeps every quantised weight inside int16 and the output sum inside int32


KING_BUCKETS = {0: 1, 1: 4}  # king layout -> number of input buckets (see src/nnue.h)


def king_bucket(kv):
    return torch.where(kv < 8, torch.where((kv & 7) < 2, 0, 1), torch.where(kv < 16, 2, 3))


def views(f, layout=0):
    """Feature indices from White's and Black's point of view (Black's: colours swapped, ranks flipped), exactly as
    the engine computes them; layout 1 adds king input buckets with the board mirrored onto the king's half.

    Works on a torch tensor of int16 indices (-1 = empty slot) on any device; empty slots map to the padding row."""
    f = f.long()
    valid = f >= 0
    c, rest = f // 384, f % 384
    t, sq = rest // 64, rest % 64
    pad = torch.full_like(f, 768 * KING_BUCKETS[layout])
    if layout == 0:
        black = (1 - c) * 384 + t * 64 + (sq ^ 56)
        return torch.where(valid, f, pad), torch.where(valid, black, pad)
    p = c * 6 + t
    wk = torch.where((f >= 320) & (f < 384), f - 320, 0).max(1, keepdim=True).values
    bk = torch.where((f >= 704) & (f < 768), f - 704, 0).max(1, keepdim=True).values ^ 56
    mw, mb = (wk & 7) >= 4, (bk & 7) >= 4
    white = king_bucket(torch.where(mw, wk ^ 7, wk)) * 768 + p * 64 + torch.where(mw, sq ^ 7, sq)
    sb = sq ^ 56
    black = king_bucket(torch.where(mb, bk ^ 7, bk)) * 768 + ((p + 6) % 12) * 64 + torch.where(mb, sb ^ 7, sb)
    return torch.where(valid, white, pad), torch.where(valid, black, pad)


class Net(nn.Module):
    def __init__(self, hidden, buckets=1, layout=0, factorize=False, screlu=False):
        super().__init__()
        self.buckets, self.layout, self.screlu = buckets, layout, screlu
        self.pad = 768 * KING_BUCKETS[layout]
        self.ft = nn.EmbeddingBag(self.pad + 1, hidden, mode="sum", padding_idx=self.pad)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, buckets)
        nn.init.normal_(self.ft.weight, std=0.01)
        # Factoriser: piece-square weights shared by every king bucket, trained alongside and merged into each
        # bucket at export, so rarely seen buckets still learn from all positions. The file format is unchanged.
        self.fact = None
        if factorize and layout:
            self.fact = nn.EmbeddingBag(769, hidden, mode="sum", padding_idx=768)
            nn.init.zeros_(self.fact.weight)

    def inputs(self, idx):
        acc = self.ft(idx) + self.ft_bias
        if self.fact is not None:
            acc = acc + self.fact(torch.where(idx == self.pad, 768, idx % 768))
        return acc

    def merged_ft(self):
        """The bucketed feature weights with the factoriser folded in: what the engine sees."""
        w = self.ft.weight[:self.pad]
        return w if self.fact is None else w + self.fact.weight[:768].repeat(KING_BUCKETS[self.layout], 1)

    def clip(self):
        with torch.no_grad():
            if self.fact is None:
                self.ft.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
            else:
                self.fact.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
                shared = self.fact.weight[:768].repeat(KING_BUCKETS[self.layout], 1)
                self.ft.weight[:self.pad] = (self.ft.weight[:self.pad] + shared).clamp(-WEIGHT_CLIP, WEIGHT_CLIP) - shared
            self.out.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

    def forward(self, white, black, stm, with_hidden=False):
        w = self.inputs(white)
        b = self.inputs(black)
        s = stm.unsqueeze(1).bool()
        us, them = torch.where(s, b, w), torch.where(s, w, b)
        x = torch.clamp(torch.cat([us, them], 1), 0, 1)
        x = x * x if self.screlu else x
        out = self.out(x)
        # Same bucket as the engine: the piece count (kings included) split evenly over 2..32.
        pieces = (white != self.pad).sum(1)
        bucket = torch.clamp((pieces - 2) * self.buckets // 32, 0, self.buckets - 1)
        out = out.squeeze(1) if self.buckets == 1 else out.gather(1, bucket.unsqueeze(1)).squeeze(1)
        return (out, x, bucket) if with_hidden else out


def export(net, path, hidden):
    ft_w = np.round(net.merged_ft().detach().cpu().numpy() * QA).astype(np.int16)  # [K*768, H]
    ft_b = np.round(net.ft_bias.detach().cpu().numpy() * QA).astype(np.int16)           # [H]
    out_w = np.round(net.out.weight.detach().cpu().numpy() * QB).astype(np.int16)       # [B, 2H]
    out_b = [int(round(float(x) * QA * QB)) for x in net.out.bias.detach().cpu()]       # [B]
    with open(path, "wb") as fh:
        fh.write(b"TSRG")
        if net.screlu:
            fh.write(struct.pack("<IIIII", 5, hidden, net.buckets, net.layout, 1))
        elif net.layout:
            fh.write(struct.pack("<IIII", 4, hidden, net.buckets, net.layout))
        elif net.buckets == 1:
            fh.write(struct.pack("<II", 1, hidden))
        else:
            fh.write(struct.pack("<III", 2, hidden, net.buckets))
        fh.write(ft_w.tobytes())
        fh.write(ft_b.tobytes())
        fh.write(out_w.tobytes())
        fh.write(struct.pack(f"<{len(out_b)}i", *out_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--buckets", type=int, default=1, help="output buckets by piece count (1 or 8)")
    ap.add_argument("--king-layout", type=int, default=0, help="0 = no king buckets, 1 = 4 mirrored king buckets")
    ap.add_argument("--factorize", action="store_true", help="train king buckets with a shared factoriser")
    ap.add_argument("--screlu", action="store_true", help="squared clipped ReLU activation (network format version 5)")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--wdl", type=float, default=0.0, help="weight of the game result in the target (0 = score only)")
    ap.add_argument("--hetero", type=float, default=None, metavar="BETA",
                    help="train a log-variance head jointly with the value under the beta-NLL loss (not exported)")
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

    net = Net(a.hidden, a.buckets, a.king_layout, a.factorize, a.screlu).to(dev)
    params = list(net.parameters())
    var = None
    if a.hetero is not None:
        # Log-variance head on the (gradient-stopped) hidden layer, started at the targets' variance.
        score = 1.0 / (1.0 + np.exp(-c[train_idx[:1000000]] / SCALE))
        res = r[train_idx[:1000000]]
        tgt0 = np.where(res < 0, score, (1 - a.wdl) * score + a.wdl * res)
        var = nn.Linear(2 * a.hidden, a.buckets).to(dev)
        nn.init.zeros_(var.weight)
        nn.init.constant_(var.bias, float(np.log(np.var(tgt0))))
        params += list(var.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, a.epochs // 3), gamma=0.3)

    # Only the compact int16 features cross to the device; the index arithmetic runs there.
    def batch(idx):
        w, b = views(torch.from_numpy(f[idx]).to(dev, non_blocking=True), a.king_layout)
        return w, b, torch.from_numpy(s[idx].astype(np.int64)).to(dev), make_target(idx)

    def make_target(idx):
        score = torch.sigmoid(torch.from_numpy(c[idx]).to(dev) / SCALE)
        result = torch.from_numpy(r[idx]).to(dev)
        return torch.where(result < 0, score, (1 - a.wdl) * score + a.wdl * result)

    def residual(w, b, st, target):
        out, x, k = net(w, b, st, with_hidden=True)
        res = torch.sigmoid(out) - target
        if var is None:
            return res, None
        return res, var(x.detach()).gather(1, k.unsqueeze(1)).squeeze(1)

    def loss_fn(res, logvar):
        if logvar is None:
            return torch.mean(res ** 2)
        # beta-NLL (Seitzer et al. 2022): each position's NLL weighted by its own predicted variance ** beta.
        nll = 0.5 * (res ** 2 * torch.exp(-logvar) + logvar)
        return torch.mean(torch.exp(a.hetero * logvar).detach() * nll)

    def val_stats():
        net.eval()
        stats = []
        with torch.no_grad():
            for i in range(0, len(val_idx), a.batch):
                res, logvar = residual(*batch(val_idx[i:i + a.batch]))
                if logvar is None:
                    stats.append([torch.mean(res ** 2).item()])
                    continue
                sigma, err = torch.exp(0.5 * logvar), res.abs()
                corr = torch.mean((sigma - sigma.mean()) * (err - err.mean())) / (sigma.std() * err.std() + 1e-12)
                stats.append([torch.mean(res ** 2).item(),
                              torch.mean(0.5 * (res ** 2 * torch.exp(-logvar) + logvar)).item(), corr.item()])
        net.train()
        return np.mean(stats, axis=0)

    history = []
    for epoch in range(a.epochs):
        t0, tot, nb = time.time(), torch.zeros((), device=dev), 0
        rng.shuffle(train_idx)
        for i in range(0, len(train_idx), a.batch):
            w, b, st, target = batch(train_idx[i:i + a.batch])
            loss = loss_fn(*residual(w, b, st, target))
            opt.zero_grad()
            loss.backward()
            opt.step()
            net.clip()
            tot, nb = tot + loss.detach(), nb + 1
        sched.step()
        stats = val_stats()
        history.append({"epoch": epoch + 1, "train_loss": tot.item() / nb, "val_loss": float(stats[0]),
                        "seconds": round(time.time() - t0)})
        if var is not None:
            history[-1].update(val_nll=float(stats[1]), val_sigma_corr=float(stats[2]))
        print(json.dumps(history[-1]), flush=True)
        export(net, a.out, a.hidden)

    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"data": a.data, "positions": len(c), "hidden": a.hidden, "buckets": a.buckets, "king_layout": a.king_layout, "factorize": a.factorize, "screlu": a.screlu, "hetero": a.hetero, "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "wdl": a.wdl,
         "scale": SCALE, "qa": QA, "qb": QB, "history": history}, indent=1))


if __name__ == "__main__":
    main()
