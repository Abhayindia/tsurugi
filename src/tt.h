#pragma once

#include "types.h"

#include <cstddef>
#include <vector>

namespace tsurugi {

__extension__ using u128 = unsigned __int128;

enum Bound : std::uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    std::uint32_t key32;
    Move move;
    std::int16_t score;
    std::int16_t eval;
    std::uint8_t depth;
    std::uint8_t gen_bound;  // generation in high 6 bits, bound in low 2

    Bound bound() const { return Bound(gen_bound & 3); }
};

class TranspositionTable {
public:
    void resize(std::size_t mb);
    void clear();
    void new_search() { generation_ = (generation_ + 4) & 0xFC; }

    bool probe(U64 key, TTEntry& out) const;
    void store(U64 key, Move move, Value score, Value eval, int depth, Bound bound);
    int hashfull() const;

private:
    std::size_t index(U64 key) const {
        return std::size_t((static_cast<u128>(key) * table_.size()) >> 64);
    }

    std::vector<TTEntry> table_;
    std::uint8_t generation_ = 0;
};

extern TranspositionTable TT;

// Mate scores are stored relative to the node, not the root.
inline Value score_to_tt(Value v, int ply) {
    return v >= VALUE_MATE_IN_MAX_PLY ? v + ply : v <= -VALUE_MATE_IN_MAX_PLY ? v - ply : v;
}
inline Value score_from_tt(Value v, int ply) {
    return v >= VALUE_MATE_IN_MAX_PLY ? v - ply : v <= -VALUE_MATE_IN_MAX_PLY ? v + ply : v;
}

}  // namespace tsurugi
