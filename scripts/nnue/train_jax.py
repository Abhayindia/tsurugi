"""Train Tsurugi's NNUE with JAX, for TPUs (Kaggle TPU VMs); same model, data, targets and file format as train.py.

Usage: python scripts/nnue/train_jax.py --data <dir[,dir:N...]> --out net.nnue [--hidden 1536] [--buckets 8]
       [--epochs 15] [--wdl 0.25]
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
PAD = 768
WEIGHT_CLIP = 127 / QB


def views(f):
    """Feature indices from White's and Black's point of view; empty slots map to the zero row PAD."""
    f = f.astype(jnp.int32)
    valid = f >= 0
    c, rest = f // 384, f % 384
    t, sq = rest // 64, rest % 64
    black = (1 - c) * 384 + t * 64 + (sq ^ 56)
    return jnp.where(valid, f, PAD), jnp.where(valid, black, PAD)


def init(key, hidden, buckets):
    k1, k2, k3 = jax.random.split(key, 3)
    bound = 1.0 / np.sqrt(2 * hidden)
    ft_w = jax.random.normal(k1, (PAD + 1, hidden)) * 0.01
    return {"ft_w": ft_w.at[PAD].set(0.0), "ft_b": jnp.zeros(hidden),
            "out_w": jax.random.uniform(k2, (buckets, 2 * hidden), minval=-bound, maxval=bound),
            "out_b": jax.random.uniform(k3, (buckets,), minval=-bound, maxval=bound)}


def forward(p, feats, stm, buckets):
    white, black = views(feats)
    w = p["ft_w"][white].sum(1) + p["ft_b"]
    b = p["ft_w"][black].sum(1) + p["ft_b"]
    s = (stm == 1)[:, None]
    x = jnp.clip(jnp.concatenate([jnp.where(s, b, w), jnp.where(s, w, b)], 1), 0.0, 1.0)
    pieces = (white != PAD).sum(1)
    k = jnp.clip((pieces - 2) * buckets // 32, 0, buckets - 1) if buckets > 1 else jnp.zeros_like(pieces)
    return (x * p["out_w"][k]).sum(1) + p["out_b"][k]


def export(p, path, hidden, buckets):
    ft_w = np.round(np.asarray(p["ft_w"])[:PAD] * QA).astype(np.int16)
    ft_b = np.round(np.asarray(p["ft_b"]) * QA).astype(np.int16)
    out_w = np.round(np.asarray(p["out_w"]) * QB).astype(np.int16)
    out_b = [int(round(float(x) * QA * QB)) for x in np.asarray(p["out_b"])]
    with open(path, "wb") as fh:
        fh.write(b"TSRG")
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
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--wdl", type=float, default=0.0)
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
    params = jax.device_put(init(jax.random.PRNGKey(1), a.hidden, a.buckets), repl)
    opt_state = jax.device_put(opt.init(params), repl)

    def loss_fn(p, feats, stm, tgt):
        return jnp.mean((jax.nn.sigmoid(forward(p, feats, stm, a.buckets)) - tgt) ** 2)

    @jax.jit
    def step(p, st, feats, stm, tgt):
        loss, g = jax.value_and_grad(loss_fn)(p, feats, stm, tgt)
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        p = {**p, "ft_w": jnp.clip(p["ft_w"], -WEIGHT_CLIP, WEIGHT_CLIP).at[PAD].set(0.0),
             "out_w": jnp.clip(p["out_w"], -WEIGHT_CLIP, WEIGHT_CLIP)}
        return p, st, loss

    eval_loss = jax.jit(loss_fn)

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
        val = float(np.mean([float(eval_loss(params, *put(v))) for v in vb]))
        history.append({"epoch": epoch + 1, "train_loss": float(jnp.mean(jnp.stack(losses))), "val_loss": val,
                        "seconds": round(time.time() - t0)})
        print(json.dumps(history[-1]), flush=True)
        export(jax.device_get(params), a.out, a.hidden, a.buckets)

    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"data": a.data, "positions": len(c), "hidden": a.hidden, "buckets": a.buckets, "epochs": a.epochs,
         "batch": batch, "lr": a.lr, "wdl": a.wdl, "trainer": "jax", "devices": len(devices),
         "scale": SCALE, "qa": QA, "qb": QB, "history": history}, indent=1))


if __name__ == "__main__":
    main()
