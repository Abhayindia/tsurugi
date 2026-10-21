#include "tt.h"

#include <algorithm>

namespace tsurugi {

TranspositionTable TT;

void TranspositionTable::resize(std::size_t mb) {
    std::size_t entries = std::max<std::size_t>(1, mb * 1024 * 1024 / sizeof(TTEntry));
    table_.assign(entries, TTEntry{});
    generation_ = 0;
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTEntry{});
    generation_ = 0;
}

bool TranspositionTable::probe(U64 key, TTEntry& out) const {
    const TTEntry& e = table_[index(key)];
    if (e.key32 != std::uint32_t(key >> 32) || e.bound() == BOUND_NONE) return false;
    out = e;
    return true;
}

void TranspositionTable::store(U64 key, Move move, Value score, Value eval, int depth, Bound bound, bool pv) {
    TTEntry& e = table_[index(key)];
    std::uint32_t k = std::uint32_t(key >> 32);
    bool same = e.key32 == k;
    if (move != NO_MOVE || !same) e.move = move;
    bool stale = (e.gen_bound & 0xF8) != generation_;
    if (!same || stale || bound == BOUND_EXACT || depth + 3 >= e.depth) {
        e.key32 = k;
        e.score = std::int16_t(score);
        e.eval = std::int16_t(eval);
        e.depth = std::uint8_t(std::max(depth, 0));
        e.gen_bound = std::uint8_t(generation_ | (pv ? 4 : 0) | bound);
    }
}

int TranspositionTable::hashfull() const {
    int used = 0;
    std::size_t n = std::min<std::size_t>(1000, table_.size());
    for (std::size_t i = 0; i < n; ++i)
        if (table_[i].bound() != BOUND_NONE && (table_[i].gen_bound & 0xF8) == generation_) ++used;
    return int(used * 1000 / n);
}

}  // namespace tsurugi
