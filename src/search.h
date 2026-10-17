#pragma once

#include "position.h"

#include <atomic>
#include <cstdint>

namespace tsurugi {

struct SearchLimits {
    std::int64_t time[COLOR_NB] = {-1, -1};
    std::int64_t inc[COLOR_NB] = {0, 0};
    std::int64_t movetime = -1;
    int movestogo = 0;
    int depth = MAX_PLY - 1;
    std::uint64_t nodes = 0;
    bool infinite = false;
};

// Each technique can be switched off at runtime (UCI options) for SPRT tests and ablation.
struct SearchFeatures {
    bool null_move = true;
    bool lmr = true;
    bool rfp = true;
    bool razoring = true;
    bool futility = true;
    bool lmp = true;
    bool aspiration = true;
    bool check_ext = false;
    bool see_pruning = false;
    bool counter_moves = false;
    bool staged_tt = false;
};

extern SearchFeatures Features;

struct SearchResult {
    Move best = NO_MOVE;
    Value score = 0;
    int depth = 0;
    std::uint64_t nodes = 0;
};

class Searcher {
public:
    SearchResult search(Position& pos, const SearchLimits& limits, bool verbose);
    void new_game();
    void request_stop() { stop_.store(true, std::memory_order_relaxed); }
    int move_overhead = 50;

private:
    struct Frame {
        Move killers[2];
        Move current;
        Piece moved;
        Value static_eval;
    };

    Value negamax(Position& pos, Value alpha, Value beta, int depth, int ply);
    Value qsearch(Position& pos, Value alpha, Value beta, int ply);
    void score_moves(const Position& pos, const Move* moves, int n, int* scores, Move tt_move, int ply) const;
    void update_quiet_stats(const Position& pos, Move best, const Move* quiets, int nq, int depth, int ply);
    bool out_of_time();
    void setup_time(const Position& pos, const SearchLimits& limits);
    std::int64_t elapsed() const;

    std::atomic<bool> stop_{false};
    std::uint64_t nodes_ = 0;
    std::uint64_t node_limit_ = 0;
    std::int64_t start_ = 0, soft_ = -1, hard_ = -1;
    int seldepth_ = 0;
    Move root_best_ = NO_MOVE;

    int history_[COLOR_NB][SQUARE_NB][SQUARE_NB] = {};
    Move counter_[PIECE_NB][SQUARE_NB] = {};
    Frame stack_[MAX_PLY + 4] = {};
    Move pv_[MAX_PLY + 1][MAX_PLY + 1] = {};
    int pv_len_[MAX_PLY + 1] = {};
};

bool see_ge(const Position& pos, Move m, int threshold);
std::uint64_t bench(Searcher& searcher, int depth);

}  // namespace tsurugi
