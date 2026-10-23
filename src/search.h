#pragma once

#include "position.h"

#include <atomic>
#include <cstdint>
#include <vector>

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
    bool check_ext = true;
    bool see_pruning = true;
    bool counter_moves = true;
    bool staged_tt = true;
    bool cont_history = true;
    bool iir = false;
    bool singular = true;
    bool corr_history = true;
    bool nonpawn_corr = true;
    bool capt_history = false;
    bool node_tm = true;
    bool cut_node = true;
    bool probcut = true;
    bool singular_v2 = true;
    bool tt_pv = true;
    bool lmr_terms = false;
    bool iir_cut = false;
    bool unc_prune = false;
    bool unc_lmr = false;
    bool unc_prune2 = false;
    bool unc_lmr2 = false;
    bool lmr_history = false;
    bool cuckoo = false;
    bool cont_corr = false;
    bool pawn_history = false;
    bool threat_history = true;
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
    // Clears a previous stop. Call it from the thread that issues stops, before starting the search, so a stop
    // that arrives before the search thread runs is never lost.
    void clear_stop() { stop_.store(false, std::memory_order_relaxed); }
    int move_overhead = 50;

private:
    struct Frame {
        Move killers[2];
        Move current;
        Piece moved;
        Value static_eval;
        Move excluded;
        int double_ext;  // double extensions already made on the path to this node
        int pawn_idx;    // pawn-history slot of this node's pawn structure
        U64 threats;     // squares the opponent attacks at this node
    };

    Value negamax(Position& pos, Value alpha, Value beta, int depth, int ply, bool cut);
    Value qsearch(Position& pos, Value alpha, Value beta, int ply);
    void score_moves(const Position& pos, const Move* moves, int n, int* scores, Move tt_move, int ply);
    void update_quiet_stats(const Position& pos, Move best, const Move* quiets, int nq, int depth, int ply);
    bool out_of_time();
    int* cont_entry(int ply, Piece p, int to);
    int quiet_score(Color us, Move m, Piece p, int ply);
    int& capt_entry(const Position& pos, Move m);
    void update_capture_stats(const Position& pos, Move best, const Move* captures, int nc, int depth);
    int& corr_entry(const Position& pos);
    int& nonpawn_corr_entry(const Position& pos, Color side);
    int* cont_corr_entry(const Position& pos, int ply);
    Value corrected(const Position& pos, Value raw, int ply);
    void setup_time(const Position& pos, const SearchLimits& limits);
    std::int64_t elapsed() const;

    std::atomic<bool> stop_{false};
    std::uint64_t nodes_ = 0;
    std::uint64_t node_limit_ = 0;
    std::int64_t start_ = 0, soft_ = -1, hard_ = -1;
    int seldepth_ = 0;
    Move root_best_ = NO_MOVE;
    // Nodes spent under each root move this search, for time management.
    Move root_moves_[256] = {};
    std::uint64_t root_nodes_[256] = {};
    int root_count_ = 0;
    std::uint64_t root_nodes_of(Move m) const;

    int history_[COLOR_NB][SQUARE_NB][SQUARE_NB] = {};
    Move counter_[PIECE_NB][SQUARE_NB] = {};
    // Capture history: [moving piece][to][captured piece type].
    int capt_[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB] = {};
    // Continuation history: [previous piece][previous to][piece][to], indexed by cont_entry().
    // Correction history: running average of (search result - static eval) per pawn structure, x256.
    static constexpr int CORR_SIZE = 16384;
    int corr_[COLOR_NB][CORR_SIZE] = {};
    // The same, keyed by one side's non-pawn pieces: [side to move][side whose pieces][key].
    int np_corr_[COLOR_NB][COLOR_NB][CORR_SIZE] = {};
    // The same, keyed by the last two moves: [side to move][piece, to two plies up][piece, to one ply up].
    std::vector<int> cont_corr_ = std::vector<int>(int(COLOR_NB) * PIECE_NB * SQUARE_NB * PIECE_NB * SQUARE_NB);
    std::vector<int> cont_ = std::vector<int>(int(PIECE_NB) * SQUARE_NB * PIECE_NB * SQUARE_NB);
    // Pawn history: [pawn-structure slot][piece][to], for quiet moves.
    static constexpr int PAWN_HIST_SIZE = 512;
    std::vector<int> pawn_hist_ = std::vector<int>(PAWN_HIST_SIZE * PIECE_NB * SQUARE_NB);
    // Threat history: [side][from attacked][to attacked][from][to], replaces the main history for quiet moves.
    std::vector<int> threat_hist_ = std::vector<int>(int(COLOR_NB) * 4 * SQUARE_NB * SQUARE_NB);
    Frame stack_[MAX_PLY + 4] = {};
    Move pv_[MAX_PLY + 1][MAX_PLY + 1] = {};
    int pv_len_[MAX_PLY + 1] = {};
};

bool see_ge(const Position& pos, Move m, int threshold);
std::uint64_t bench(Searcher& searcher, int depth);

}  // namespace tsurugi
