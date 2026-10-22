"""Cross-check the engine's NNUE evaluation against an independent numpy implementation of the same file.

Usage: python scripts/nnue/check_net.py <engine> <net.nnue>
Any difference means the trainer and the engine disagree about features, perspective or scaling.
"""
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
    ft_w = np.frombuffer(raw, np.int16, 768 * H, o).reshape(768, H).astype(np.int64); o += 768 * H * 2
    ft_b = np.frombuffer(raw, np.int16, H, o).astype(np.int64); o += H * 2
    out_w = np.frombuffer(raw, np.int16, B * 2 * H, o).reshape(B, 2 * H).astype(np.int64); o += B * 2 * H * 2
    out_b = struct.unpack_from(f"<{B}i", raw, o); o += B * 4
    aux = None
    if version == 3:
        aux_w = np.frombuffer(raw, np.int16, B * 2 * H, o).reshape(B, 2 * H).astype(np.int64); o += B * 2 * H * 2
        aux_b = struct.unpack_from(f"<{B}i", raw, o); o += B * 4
        aux = (aux_w, aux_b, struct.unpack_from("<i", raw, o)[0])
    return ft_w, ft_b, out_w, out_b, H, aux


def reference(net, board):
    ft_w, ft_b, out_w, out_b, H, aux = net
    acc = {c: ft_b.copy() for c in (chess.WHITE, chess.BLACK)}
    for sq, p in board.piece_map().items():
        idx = ((0 if p.color else 1) * 6 + p.piece_type - 1)
        acc[chess.WHITE] += ft_w[idx * 64 + sq]
        acc[chess.BLACK] += ft_w[((idx + 6) % 12) * 64 + (sq ^ 56)]
    us, them = acc[board.turn], acc[not board.turn]
    B = len(out_b)
    k = min(max((len(board.piece_map()) - 2) * B // 32, 0), B - 1)
    s = int(np.dot(np.clip(us, 0, QA), out_w[k][:H]) + np.dot(np.clip(them, 0, QA), out_w[k][H:]))
    v = (s + out_b[k]) * SCALE
    unc = None
    if aux:
        aux_w, aux_b, q = aux
        u = int(np.dot(np.clip(us, 0, QA), aux_w[k][:H]) + np.dot(np.clip(them, 0, QA), aux_w[k][H:]))
        unc = max(0, int((u + aux_b[k]) / q))
    return int(v / (QA * QB)), unc  # C++ integer division truncates toward zero


def engine_eval(engine, net_path, fen):
    cmds = f"setoption name EvalFile value {net_path}\nsetoption name UseNNUE value true\nposition fen {fen}\neval\nquit\n"
    out = subprocess.run([engine], input=cmds, capture_output=True, text=True).stdout
    words = next(l for l in out.splitlines() if l.startswith("eval ")).split()
    unc = int(words[words.index("uncertainty") + 1]) if "uncertainty" in words else None
    return int(words[1]), unc


def main():
    engine, net_path = sys.argv[1], sys.argv[2]
    net = load(net_path)
    bad = 0
    for fen in FENS:
        ref, eng = reference(net, chess.Board(fen)), engine_eval(engine, net_path, fen)
        bad += ref != eng
        extra = f"  uncertainty ref {ref[1]} engine {eng[1]}" if ref[1] is not None else ""
        print(f"{'ok  ' if ref == eng else 'FAIL'} ref {ref[0]:6} engine {eng[0]:6}{extra}  {fen}")
    print("ALL MATCH" if not bad else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
