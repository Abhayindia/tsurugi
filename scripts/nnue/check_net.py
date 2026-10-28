"""Cross-check the engine's NNUE evaluation against an independent numpy implementation of the same file.

Usage: python scripts/nnue/check_net.py <engine> <net.nnue>
Any difference means the trainer and the engine disagree about features, perspective or scaling.
"""
import random
import struct
import subprocess
import sys

import chess
import numpy as np

QA, QB, SCALE = 255, 64, 400
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
    "6k1/5p2/6p1/8/7p/8/6PP/6K1 b - - 0 1",
]


def load(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"TSRG"
    version, H = struct.unpack_from("<II", raw, 4)
    B, o = (struct.unpack_from("<I", raw, 12)[0], 16) if version >= 2 else (1, 12)
    layout = act = 0
    if version >= 4:
        layout = struct.unpack_from("<I", raw, o)[0]; o += 4
    if version == 5:
        act = struct.unpack_from("<I", raw, o)[0]; o += 4
    rows = 768 * (4 if layout == 1 else 1)
    ft_w = np.frombuffer(raw, np.int16, rows * H, o).reshape(rows, H).astype(np.int64); o += rows * H * 2
    ft_b = np.frombuffer(raw, np.int16, H, o).astype(np.int64); o += H * 2
    out_w = np.frombuffer(raw, np.int16, B * 2 * H, o).reshape(B, 2 * H).astype(np.int64); o += B * 2 * H * 2
    out_b = struct.unpack_from(f"<{B}i", raw, o); o += B * 4
    aux = None
    if version == 3:
        aux_w = np.frombuffer(raw, np.int16, B * 2 * H, o).reshape(B, 2 * H).astype(np.int64); o += B * 2 * H * 2
        aux_b = struct.unpack_from(f"<{B}i", raw, o); o += B * 4
        aux = (aux_w, aux_b, struct.unpack_from("<i", raw, o)[0])
    assert o == len(raw), "trailing or missing bytes"
    return ft_w, ft_b, out_w, out_b, H, aux, layout, act


def tdiv(n, d):
    """Integer division truncating toward zero, as in C++."""
    q = abs(n) // abs(d)
    return q if (n >= 0) == (d > 0) else -q


def feature(layout, persp_white, piece, sq, ksq):
    """Row of piece `piece` (engine index, white pawn = 0) on `sq` for one perspective whose king is on `ksq`."""
    pc, s, k = (piece, sq, ksq) if persp_white else ((piece + 6) % 12, sq ^ 56, ksq ^ 56)
    if layout == 0:
        return pc * 64 + s
    if (k & 7) >= 4:
        s, k = s ^ 7, k ^ 7
    bucket = (0 if (k & 7) < 2 else 1) if k < 8 else 2 if k < 16 else 3
    return bucket * 768 + pc * 64 + s


def reference(net, board):
    ft_w, ft_b, out_w, out_b, H, aux, layout, act = net
    acc = {c: ft_b.copy() for c in (chess.WHITE, chess.BLACK)}
    for sq, p in board.piece_map().items():
        idx = ((0 if p.color else 1) * 6 + p.piece_type - 1)
        acc[chess.WHITE] += ft_w[feature(layout, True, idx, sq, board.king(chess.WHITE))]
        acc[chess.BLACK] += ft_w[feature(layout, False, idx, sq, board.king(chess.BLACK))]
    us, them = acc[board.turn], acc[not board.turn]
    B = len(out_b)
    k = min(max((len(board.piece_map()) - 2) * B // 32, 0), B - 1)
    if act == 1:
        u, t = np.clip(us, 0, QA), np.clip(them, 0, QA)
        s = int(np.dot(u * u, out_w[k][:H]) + np.dot(t * t, out_w[k][H:]))
        return tdiv((s + out_b[k] * QA) * SCALE, QA * QA * QB), None
    s = int(np.dot(np.clip(us, 0, QA), out_w[k][:H]) + np.dot(np.clip(them, 0, QA), out_w[k][H:]))
    v = (s + out_b[k]) * SCALE
    unc = None
    if aux:
        aux_w, aux_b, q = aux
        u = int(np.dot(np.clip(us, 0, QA), aux_w[k][:H]) + np.dot(np.clip(them, 0, QA), aux_w[k][H:]))
        unc = max(0, int((u + aux_b[k]) / q))
    return int(v / (QA * QB)), unc  # C++ integer division truncates toward zero


# Positions reached by playing moves, so the engine's incremental (and lazy) accumulator updates are checked too:
# king moves across input buckets and across the board's middle, both castlings, captures, promotions, en passant.
SEQUENCES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4 e7e5 e1e2 e8e7 e2d3 e7d6 d3c4 d6c6"),
    ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1 e8c8 g1h1 c8b8"),
    ("8/1P6/7k/8/8/8/6p1/4K2R w K - 0 1", "b7b8q g2h1q e1e2 h1h2"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6 e8d7 e1d2 d7d6"),
]


def engine_eval(engine, net_path, position):
    cmds = (f"setoption name EvalFile value {net_path}\nsetoption name UseNNUE value true\n"
            f"position {position}\neval\nquit\n")
    out = subprocess.run([engine], input=cmds, capture_output=True, text=True).stdout
    words = next(l for l in out.splitlines() if l.startswith("eval ")).split()
    unc = int(words[words.index("uncertainty") + 1]) if "uncertainty" in words else None
    return int(words[1]), unc


def cases():
    for fen in FENS:
        yield f"fen {fen}", chess.Board(fen), fen
    for fen, moves in SEQUENCES:
        board = chess.Board(fen)
        for m in moves.split():
            board.push_uci(m)
        yield f"fen {fen} moves {moves}", board, f"{fen} + {moves}"
    # A random game from the start, checked every 10 plies.
    rng = random.Random(1)
    board, played = chess.Board(), []
    for ply in range(1, 61):
        moves = list(board.legal_moves)
        if not moves:
            break
        m = rng.choice(moves)
        board.push(m)
        played.append(m.uci())
        if ply % 10 == 0:
            yield "startpos moves " + " ".join(played), board.copy(), f"random game, ply {ply}"


def main():
    engine, net_path = sys.argv[1], sys.argv[2]
    net = load(net_path)
    bad = 0
    for position, board, label in cases():
        ref, eng = reference(net, board), engine_eval(engine, net_path, position)
        bad += ref != eng
        extra = f"  uncertainty ref {ref[1]} engine {eng[1]}" if ref[1] is not None else ""
        print(f"{'ok  ' if ref == eng else 'FAIL'} ref {ref[0]:6} engine {eng[0]:6}{extra}  {label}")
    print("ALL MATCH" if not bad else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
