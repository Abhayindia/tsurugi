"""Training data shared by the NNUE trainers: shards from prepare_data.py / prepare_selfplay.py."""
import glob
import json
from pathlib import Path

import numpy as np


def load(data, limit):
    """Read shards into preallocated arrays, so peak memory is the final size rather than twice it."""
    # Several data directories may be given, comma-separated and read in that order; "dir:N" takes at most N
    # positions from that directory.
    sources, total = [], 0
    for spec in data.split(","):
        head, _, cap = spec.rpartition(":")
        d, cap = (head, cap) if cap.isdigit() else (spec, "")
        d = Path(d).expanduser()
        dpaths = sorted(glob.glob(str(d / "shard_*.npz")))
        meta = d / "meta.json"
        kept = json.loads(meta.read_text()).get("kept", 0) if meta.exists() else 0
        kept = kept or sum(len(np.load(p)["cp"]) for p in dpaths)
        kept = min(kept, int(cap)) if cap else kept
        sources.append((dpaths, kept))
        total += kept
    n = min(limit, total) if limit else total
    f = np.empty((n, 32), dtype=np.int16)
    s = np.empty(n, dtype=np.int8)
    c_stm = np.empty(n, dtype=np.float32)
    r_stm = np.empty(n, dtype=np.float32)
    at = 0
    for dpaths, kept in sources:
        left = kept
        for path in dpaths:
            if at >= n or left <= 0:
                break
            d = np.load(path)
            k = min(len(d["cp"]), n - at, left)
            left -= k
            f[at:at + k] = d["features"][:k]
            st = d["stm"][:k]
            s[at:at + k] = st
            cp = d["cp"][:k].astype(np.float32)
            c_stm[at:at + k] = np.where(st == 0, cp, -cp)
            # Game result from White's view (0/1/2); shards without one (Stockfish labels) get -1.
            r = d["result"][:k].astype(np.float32) if "result" in d.files else np.full(k, -1.0, dtype=np.float32)
            r_stm[at:at + k] = np.where(r < 0, -1.0, np.where(st == 0, r, 2 - r) / 2)
            at += k
    return f[:at], s[:at], c_stm[:at], r_stm[:at]
