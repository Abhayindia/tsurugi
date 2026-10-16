"""Find the first perft divergence between Tsurugi and python-chess by recursive divide.

Usage: python tools/perft_check.py <divide-binary> "<fen>" <depth>
"""
import subprocess
import sys

import chess


def ref_divide(board: chess.Board, depth: int) -> dict[str, int]:
    def perft(b: chess.Board, d: int) -> int:
        if d == 0:
            return 1
        if d == 1:
            return b.legal_moves.count()
        n = 0
        for m in b.legal_moves:
            b.push(m)
            n += perft(b, d - 1)
            b.pop()
        return n

    out = {}
    for m in board.legal_moves:
        board.push(m)
        out[m.uci()] = perft(board, depth - 1)
        board.pop()
    return out


def our_divide(binary: str, fen: str, depth: int) -> dict[str, int]:
    text = subprocess.run([binary, fen, str(depth)], capture_output=True, text=True, check=True).stdout
    out = {}
    for line in text.splitlines():
        if ": " in line and not line.startswith(("Nodes", "Time", "NPS")):
            mv, n = line.split(": ")
            out[mv] = int(n)
    return out


def bisect(binary: str, fen: str, depth: int) -> None:
    board = chess.Board(fen)
    ours, ref = our_divide(binary, fen, depth), ref_divide(board, depth)
    if ours == ref:
        print(f"match at depth {depth}: {sum(ref.values())} nodes")
        return
    for mv in sorted(set(ours) | set(ref)):
        if mv not in ours:
            print(f"{fen}\n  MISSING move {mv} (reference has it)")
            return
        if mv not in ref:
            print(f"{fen}\n  ILLEGAL move {mv} generated (reference lacks it)")
            return
    for mv in sorted(ours):
        if ours[mv] != ref[mv]:
            print(f"{mv}: ours {ours[mv]} ref {ref[mv]} -> descending")
            board.push_uci(mv)
            bisect(binary, board.fen(), depth - 1)
            return


if __name__ == "__main__":
    bisect(sys.argv[1], sys.argv[2], int(sys.argv[3]))
