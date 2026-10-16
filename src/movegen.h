#pragma once

#include "types.h"

namespace tsurugi {

class Position;

struct MoveList {
    Move moves[256];
    int size = 0;
    void add(Move m) { moves[size++] = m; }
    Move* begin() { return moves; }
    Move* end() { return moves + size; }
    const Move* begin() const { return moves; }
    const Move* end() const { return moves + size; }
};

// NOISY: captures, en passant, queen promotions. QUIET: everything else. Pseudo-legal.
enum GenType { GEN_NOISY, GEN_QUIET, GEN_ALL };

template <GenType T>
void generate(const Position& pos, MoveList& list);

void generate_legal(const Position& pos, MoveList& list);

}  // namespace tsurugi
