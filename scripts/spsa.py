"""SPSA tuning of the search constants exposed by a -DTSURUGI_TUNE=ON build.

Usage: python scripts/spsa.py --engine build/tsurugi --fastchess fastchess/fastchess --book book.epd
       [--hours 5] [--iterations 600] [--pairs 8] [--tc 5+0.05] [--seed 1] [--out spsa.json]

Each iteration moves every parameter by +-c_k with random signs, plays --pairs game pairs between the two perturbed
engines and steps the parameters along the score difference, with the schedule Fishtest uses:
c_k = c_end * (N / k)^0.101, r_k = r_end * ((A + N) / (A + k))^0.602, A = N / 10, c_end = (max - min) / 20.
The JSON written after every iteration holds the current values and the per-iteration results.
Parallel chains that start from the same values are combined by summing their moves (start + sum of
(final - start)), which matches one chain with all of their iterations; averaging would keep only 1/chains of it.
"""
import argparse
import json
import os
import random
import re
import subprocess
import time

R_END = 0.002


def read_params(engine):
    out = subprocess.run([engine], input="uci\nquit\n", capture_output=True, text=True).stdout
    params = []
    for m in re.finditer(r"option name T_(\w+) type spin default (-?\d+) min (-?\d+) max (-?\d+)", out):
        params.append({"name": m[1], "value": float(m[2]), "min": int(m[3]), "max": int(m[4])})
    if not params:
        raise SystemExit("no T_ options: build with -DTSURUGI_TUNE=ON")
    return params


def play(a, plus, minus, seed):
    def side(name, values):
        return ["-engine", f"cmd={a.engine}", f"name={name}"] + [f"option.T_{k}={v}" for k, v in values.items()]
    cmd = [a.fastchess, *side("plus", plus), *side("minus", minus),
           "-each", f"tc={a.tc}", "proto=uci", "option.Hash=16",
           "-openings", f"file={a.book}", "format=epd", "order=random", "-srand", str(seed),
           "-repeat", "-games", "2", "-rounds", str(a.pairs), "-concurrency", str(a.concurrency), "-recover"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.findall(r"Games: (\d+), Wins: (\d+), Losses: (\d+)", out)
    if not m:
        raise RuntimeError("no result in fastchess output:\n" + out[-2000:])
    games, wins, losses = map(int, m[-1])
    return wins, losses, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fastchess", required=True)
    ap.add_argument("--book", required=True)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--iterations", type=int, default=600, help="N of the schedule; the run stops earlier at --hours")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--concurrency", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="spsa.json")
    ap.add_argument("--start", default="", help="JSON object of starting values (default: the engine's defaults)")
    a = ap.parse_args()

    params = read_params(a.engine)
    start = json.loads(a.start) if a.start else {}
    for p in params:
        if p["name"] in start:
            p["value"] = float(min(max(start[p["name"]], p["min"]), p["max"]))
    rng = random.Random(a.seed)
    n, big_a = a.iterations, a.iterations / 10
    history = []
    deadline = time.time() + a.hours * 3600
    for k in range(1, n + 1):
        if time.time() >= deadline:
            break
        ck = (n / k) ** 0.101
        rk = R_END * ((big_a + n) / (big_a + k)) ** 0.602
        signs = [rng.choice((-1, 1)) for _ in params]
        plus, minus = {}, {}
        for p, s in zip(params, signs):
            c = (p["max"] - p["min"]) / 20 * ck
            plus[p["name"]] = min(max(round(p["value"] + c * s), p["min"]), p["max"])
            minus[p["name"]] = min(max(round(p["value"] - c * s), p["min"]), p["max"])
        wins, losses, games = play(a, plus, minus, a.seed * 100000 + k)
        result = wins - losses
        for p, s in zip(params, signs):
            c = (p["max"] - p["min"]) / 20 * ck
            p["value"] = min(max(p["value"] + rk * c * result * s, p["min"]), p["max"])
        history.append({"k": k, "wins": wins, "losses": losses, "games": games})
        with open(a.out, "w") as fh:
            json.dump({"iterations": k, "tc": a.tc, "pairs": a.pairs, "seed": a.seed,
                       "values": {p["name"]: round(p["value"], 3) for p in params}, "history": history}, fh, indent=1)
        print(f"{k}: +{wins} -{losses} ={games - wins - losses}", flush=True)


if __name__ == "__main__":
    main()
