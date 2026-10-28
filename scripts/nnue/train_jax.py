"""Train Tsurugi's NNUE with JAX, for TPUs (Kaggle TPU VMs); same model, data, targets and file format as train.py.

Usage: python scripts/nnue/train_jax.py --data <dir[,dir:N...]> --out net.nnue [--hidden 1536] [--buckets 8]
       [--epochs 15] [--wdl 0.25] [--hetero 0.5]
The batch is split across all local devices (data parallel); weights are exported after every epoch.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shards import load  # noqa: E402

SCALE, QA, QB = 400.0, 255, 64
KING_BUCKETS = {0: 1, 1: 4}  # king layout -> number of input buckets (see src/nnue.h)
WEIGHT_CLIP = 127 / QB


def king_bucket(kv):
    return jnp.where(kv < 8, jnp.where((kv & 7) < 2, 0, 1), jnp.where(kv < 16, 2, 3))


def views(f, layout=0):
    """Feature indices from White's and Black's point of view, exactly as the engine computes them (layout 1: king
    input buckets, board mirrored onto the king's half); empty slots map to the zero padding row."""
    f = f.astype(jnp.int32)
    valid = f >= 0
    c, rest = f // 384, f % 384
    t, sq = rest // 64, rest % 64
    pad = 768 * KING_BUCKETS[layout]
    if layout == 0:
        black = (1 - c) * 384 + t * 64 + (sq ^ 56)
        return jnp.where(valid, f, pad), jnp.where(valid, black, pad)
    p = c * 6 + t
    wk = jnp.max(jnp.where((f >= 320) & (f < 384), f - 320, 0), axis=1, keepdims=True)
    bk = jnp.max(jnp.where((f >= 704) & (f < 768), f - 704, 0), axis=1, keepdims=True) ^ 56
    mw, mb = (wk & 7) >= 4, (bk & 7) >= 4
    white = king_bucket(jnp.where(mw, wk ^ 7, wk)) * 768 + p * 64 + jnp.where(mw, sq ^ 7, sq)
    sb = sq ^ 56
    black = king_bucket(jnp.where(mb, bk ^ 7, bk)) * 768 + ((p + 6) % 12) * 64 + jnp.where(mb, sb ^ 7, sb)
    return jnp.where(valid, white, pad), jnp.where(valid, black, pad)


def init(key, hidden, buckets, layout=0, factorize=False):
    k1, k2, k3 = jax.random.split(key, 3)
    bound = 1.0 / np.sqrt(2 * hidden)
    pad = 768 * KING_BUCKETS[layout]
    ft_w = jax.random.normal(k1, (pad + 1, hidden)) * 0.01
    p = {"ft_w": ft_w.at[pad].set(0.0), "ft_b": jnp.zeros(hidden),
         "out_w": jax.random.uniform(k2, (buckets, 2 * hidden), minval=-bound, maxval=bound),
         "out_b": jax.random.uniform(k3, (buckets,), minval=-bound, maxval=bound)}
    if factorize and layout:
        # Factoriser: piece-square weights shared by every king bucket, merged into each bucket at export.
        p["fact"] = jnp.zeros((769, hidden))
    return p


def merged_ft(p, layout=0):
    """The bucketed feature weights with the factoriser folded in: what the engine sees."""
    pad = 768 * KING_BUCKETS[layout]
    w = p["ft_w"][:pad]
    return w if "fact" not in p else w + jnp.tile(p["fact"][:768], (KING_BUCKETS[layout], 1))


def accumulate(p, idx, pad):
    acc = p["ft_w"][idx].sum(1) + p["ft_b"]
    if "fact" in p:
        acc = acc + p["fact"][jnp.where(idx == pad, 768, idx % 768)].sum(1)
    return acc


def forward(p, feats, stm, buckets, layout=0, screlu=False, with_hidden=False):
    pad = 768 * KING_BUCKETS[layout]
    white, black = views(feats, layout)
    w = accumulate(p, white, pad)
    b = accumulate(p, black, pad)
    s = (stm == 1)[:, None]
    x = jnp.clip(jnp.concatenate([jnp.where(s, b, w), jnp.where(s, w, b)], 1), 0.0, 1.0)
    if screlu:
        x = x * x
    pieces = (white != pad).sum(1)
    k = jnp.clip((pieces - 2) * buckets // 32, 0, buckets - 1) if buckets > 1 else jnp.zeros_like(pieces)
    out = (x * p["out_w"][k]).sum(1) + p["out_b"][k]
    return (out, x, k) if with_hidden else out


def export(p, path, hidden, buckets, layout=0, screlu=False):
    ft_w = np.round(np.asarray(merged_ft(p, layout)) * QA).astype(np.int16)
    ft_b = np.round(np.asarray(p["ft_b"]) * QA).astype(np.int16)
    out_w = np.round(np.asarray(p["out_w"]) * QB).astype(np.int16)
    out_b = [int(round(float(x) * QA * QB)) for x in np.asarray(p["out_b"])]
    with open(path, "wb") as fh:
        fh.write(b"TSRG")
        if screlu:
            fh.write(struct.pack("<IIIII", 5, hidden, buckets, layout, 1))
        elif layout:
            fh.write(struct.pack("<IIII", 4, hidden, buckets, layout))
        else:
            fh.write(struct.pack("<II", 1, hidden) if buckets == 1 else struct.pack("<III", 2, hidden, buckets))
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
    ap.add_argument("--buckets", type=int, default=1)
    ap.add_argument("--king-layout", type=int, default=0, help="0 = no king buckets, 1 = 4 mirrored king buckets")
    ap.add_argument("--factorize", action="store_true", help="train king buckets with a shared factoriser")
    ap.add_argument("--screlu", action="store_true", help="squared clipped ReLU activation (network format version 5)")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--wdl", type=float, default=0.0)
    ap.add_argument("--hetero", type=float, default=None, metavar="BETA",
                    help="train a log-variance head jointly with the value under the beta-NLL loss (not exported)")
    a = ap.parse_args()

    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("d",))
    shard, repl = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    batch = a.batch - a.batch % len(devices)
    print(f"training on {len(devices)} x {devices[0].platform}", flush=True)

    f, s, c, r = load(a.data, a.limit)
    score = 1.0 / (1.0 + np.exp(-c / SCALE))
    target = np.where(r < 0, score, (1 - a.wdl) * score + a.wdl * r).astype(np.float32)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(c))
    n_val = int(len(c) * a.val)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"{len(train_idx)} training / {n_val} validation positions", flush=True)

    steps_per_epoch = len(train_idx) // batch
    sched = optax.piecewise_constant_schedule(
        a.lr, {steps_per_epoch * max(1, a.epochs // 3) * i: 0.3 for i in range(1, 4)})
    opt = optax.adam(sched)
    params = init(jax.random.PRNGKey(1), a.hidden, a.buckets, a.king_layout, a.factorize)
    if a.hetero is not None:
        # Log-variance head on the (gradient-stopped) hidden layer, started at the targets' variance.
        params["var_w"] = jnp.zeros((a.buckets, 2 * a.hidden))
        params["var_b"] = jnp.full((a.buckets,), float(np.log(np.var(target[train_idx]))))
    params = jax.device_put(params, repl)
    pad = 768 * KING_BUCKETS[a.king_layout]
    opt_state = jax.device_put(opt.init(params), repl)

    def residual(p, feats, stm, tgt):
        out, x, k = forward(p, feats, stm, a.buckets, a.king_layout, a.screlu, with_hidden=True)
        r = jax.nn.sigmoid(out) - tgt
        if a.hetero is None:
            return r, None
        return r, (jax.lax.stop_gradient(x) * p["var_w"][k]).sum(1) + p["var_b"][k]

    def loss_fn(p, feats, stm, tgt):
        r, logvar = residual(p, feats, stm, tgt)
        if logvar is None:
            return jnp.mean(r ** 2)
        # beta-NLL (Seitzer et al. 2022): each position's NLL weighted by its own predicted variance ** beta.
        nll = 0.5 * (r ** 2 * jnp.exp(-logvar) + logvar)
        return jnp.mean(jax.lax.stop_gradient(jnp.exp(a.hetero * logvar)) * nll)

    @jax.jit
    def eval_stats(p, feats, stm, tgt):
        r, logvar = residual(p, feats, stm, tgt)
        if logvar is None:
            return jnp.mean(r ** 2), 0.0, 0.0
        sigma, err = jnp.exp(0.5 * logvar), jnp.abs(r)
        corr = jnp.mean((sigma - sigma.mean()) * (err - err.mean())) / (sigma.std() * err.std() + 1e-12)
        return jnp.mean(r ** 2), jnp.mean(0.5 * (r ** 2 * jnp.exp(-logvar) + logvar)), corr

    @jax.jit
    def step(p, st, feats, stm, tgt):
        loss, g = jax.value_and_grad(loss_fn)(p, feats, stm, tgt)
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        if "fact" in p:
            # Clip the merged weights (the ones exported), keeping the shared part inside the bound as well.
            fact = jnp.clip(p["fact"], -WEIGHT_CLIP, WEIGHT_CLIP).at[768].set(0.0)
            shared = jnp.tile(fact[:768], (KING_BUCKETS[a.king_layout], 1))
            ft = jnp.clip(p["ft_w"][:pad] + shared, -WEIGHT_CLIP, WEIGHT_CLIP) - shared
            p = {**p, "fact": fact, "ft_w": p["ft_w"].at[:pad].set(ft).at[pad].set(0.0)}
        else:
            p = {**p, "ft_w": jnp.clip(p["ft_w"], -WEIGHT_CLIP, WEIGHT_CLIP).at[pad].set(0.0)}
        p = {**p, "out_w": jnp.clip(p["out_w"], -WEIGHT_CLIP, WEIGHT_CLIP)}
        return p, st, loss

    def put(idx):
        return (jax.device_put(f[idx], shard), jax.device_put(s[idx].astype(np.int32), shard),
                jax.device_put(target[idx], shard))

    history = []
    for epoch in range(a.epochs):
        t0 = time.time()
        rng.shuffle(train_idx)
        losses = []
        for i in range(steps_per_epoch):
            params, opt_state, loss = step(params, opt_state, *put(train_idx[i * batch:(i + 1) * batch]))
            losses.append(loss)
        vb = [val_idx[i:i + batch] for i in range(0, len(val_idx) - batch + 1, batch)] or [val_idx]
        stats = np.mean([[float(x) for x in eval_stats(params, *put(v))] for v in vb], axis=0)
        history.append({"epoch": epoch + 1, "train_loss": float(jnp.mean(jnp.stack(losses))), "val_loss": stats[0],
                        "seconds": round(time.time() - t0)})
        if a.hetero is not None:
            history[-1].update(val_nll=stats[1], val_sigma_corr=stats[2])
        print(json.dumps(history[-1]), flush=True)
        export(jax.device_get({k: v for k, v in params.items() if not k.startswith("var_")}), a.out, a.hidden, a.buckets, a.king_layout, a.screlu)

    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"data": a.data, "positions": len(c), "hidden": a.hidden, "buckets": a.buckets, "king_layout": a.king_layout, "factorize": a.factorize, "screlu": a.screlu, "hetero": a.hetero, "epochs": a.epochs,
         "batch": batch, "lr": a.lr, "wdl": a.wdl, "trainer": "jax", "devices": len(devices),
         "scale": SCALE, "qa": QA, "qb": QB, "history": history}, indent=1))


if __name__ == "__main__":
    main()
