#include "search.h"

#include "eval.h"
#include "movegen.h"
#include "tt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

namespace tsurugi {

SearchFeatures Features;

namespace {

constexpr int SEE_VALUE[PIECE_TYPE_NB] = {100, 300, 300, 500, 900, 20000};
constexpr int HISTORY_MAX = 16384;

int lmr_table[MAX_PLY][64];

struct LmrInit {
    LmrInit() {
        for (int d = 1; d < MAX_PLY; ++d)
            for (int m = 1; m < 64; ++m)
                lmr_table[d][m] = int(0.7 + std::log(d) * std::log(m) / 2.3);
    }
} lmr_init;

std::int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

PieceType least_valuable(const Position& pos, U64 attackers, Color side, int& sq) {
    for (int pt = PAWN; pt <= KING; ++pt) {
        U64 b = attackers & pos.pieces(side, PieceType(pt));
        if (b) {
            sq = lsb(b);
            return PieceType(pt);
        }
    }
    return PIECE_TYPE_NB;
}

// Swap-list static exchange evaluation on the destination square.
int see(const Position& pos, Move m) {
    int from = from_sq(m), to = to_sq(m);
    int gain[40];
    int d = 0;
    Piece target = pos.piece_on(to);
    gain[0] = move_flag(m) == EP_CAPTURE ? SEE_VALUE[PAWN] : target == NO_PIECE ? 0 : SEE_VALUE[type_of(target)];
    int on_square = SEE_VALUE[type_of(pos.piece_on(from))];
    if (is_promotion(m)) {
        gain[0] += SEE_VALUE[promo_type(m)] - SEE_VALUE[PAWN];
        on_square = SEE_VALUE[promo_type(m)];
    }
    U64 occ = pos.pieces() ^ bb(from);
    if (move_flag(m) == EP_CAPTURE) occ ^= bb(to + (pos.side_to_move() == WHITE ? -8 : 8));
    Color side = pos.side_to_move();

    for (;;) {
        side = ~side;
        U64 attackers = pos.attackers_to(to, occ) & occ & pos.pieces(side);
        if (!attackers) break;
        int sq = 0;
        PieceType pt = least_valuable(pos, attackers, side, sq);
        if (pt == KING && (pos.attackers_to(to, occ) & occ & pos.pieces(~side))) break;
        ++d;
        gain[d] = on_square - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;
        on_square = SEE_VALUE[pt];
        occ ^= bb(sq);
        if (d >= 38) break;
    }
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}

void history_update(int& entry, int bonus) {
    bonus = std::clamp(bonus, -HISTORY_MAX, HISTORY_MAX);
    entry += bonus - entry * std::abs(bonus) / HISTORY_MAX;
}

std::string score_to_uci(Value v) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return "mate " + std::to_string((VALUE_MATE - v + 1) / 2);
    if (v <= -VALUE_MATE_IN_MAX_PLY) return "mate " + std::to_string(-(VALUE_MATE + v) / 2);
    return "cp " + std::to_string(v);
}

}  // namespace

bool see_ge(const Position& pos, Move m, int threshold) {
    if (is_castle(m)) return threshold <= 0;
    return see(pos, m) >= threshold;
}

void Searcher::new_game() {
    std::memset(history_, 0, sizeof(history_));
    std::memset(counter_, 0, sizeof(counter_));
    std::memset(stack_, 0, sizeof(stack_));
    std::fill(cont_.begin(), cont_.end(), 0);
    std::memset(corr_, 0, sizeof(corr_));
    std::memset(capt_, 0, sizeof(capt_));
}

int& Searcher::capt_entry(const Position& pos, Move m) {
    PieceType victim = move_flag(m) == EP_CAPTURE ? PAWN : type_of(pos.piece_on(to_sq(m)));
    return capt_[pos.piece_on(from_sq(m))][to_sq(m)][victim];
}

int& Searcher::corr_entry(const Position& pos) {
    U64 h = (pos.pieces(WHITE, PAWN) * 0x9E3779B97F4A7C15ULL) ^ (pos.pieces(BLACK, PAWN) * 0xC2B2AE3D27D4EB4FULL);
    return corr_[pos.side_to_move()][h >> 50];
}

Value Searcher::corrected(const Position& pos, Value raw) {
    if (!Features.corr_history || raw == VALUE_NONE) return raw;
    return std::clamp(raw + corr_entry(pos) / 256, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

// The continuation-history entry for piece p moving to `to`, following the move made `ply` plies up the stack.
int* Searcher::cont_entry(int ply, Piece p, int to) {
    if (ply < 0 || stack_[ply].current == NO_MOVE) return nullptr;
    const Frame& f = stack_[ply];
    return &cont_[((int(f.moved) * SQUARE_NB + to_sq(f.current)) * PIECE_NB + p) * SQUARE_NB + to];
}

int Searcher::quiet_score(Color us, Move m, Piece p, int ply) {
    int s = history_[us][from_sq(m)][to_sq(m)];
    if (!Features.cont_history) return s;
    for (int back = 1; back <= 2; ++back)
        if (int* e = cont_entry(ply - back, p, to_sq(m))) s += *e;
    return s;
}

std::int64_t Searcher::elapsed() const { return now_ms() - start_; }

void Searcher::setup_time(const Position& pos, const SearchLimits& limits) {
    soft_ = hard_ = -1;
    if (limits.infinite) return;
    if (limits.movetime >= 0) {
        soft_ = hard_ = std::max<std::int64_t>(1, limits.movetime - move_overhead);
        return;
    }
    Color us = pos.side_to_move();
    if (limits.time[us] < 0) return;
    std::int64_t avail = std::max<std::int64_t>(1, limits.time[us] - move_overhead);
    int mtg = limits.movestogo > 0 ? std::min(limits.movestogo, 40) : 55;
    std::int64_t soft = avail / mtg + limits.inc[us] * 3 / 5;
    hard_ = std::max<std::int64_t>(1, std::min(soft * 2, avail / 10));
    soft_ = std::max<std::int64_t>(1, std::min(soft, hard_));
}

bool Searcher::out_of_time() {
    if (stop_.load(std::memory_order_relaxed)) return true;
    if ((nodes_ & 1023) != 0) return false;
    if ((node_limit_ && nodes_ >= node_limit_) || (hard_ >= 0 && elapsed() >= hard_)) {
        stop_.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void Searcher::score_moves(const Position& pos, const Move* moves, int n, int* scores, Move tt_move, int ply) {
    Color us = pos.side_to_move();
    const Frame& f = stack_[ply];
    Move counter = NO_MOVE;
    if (Features.counter_moves && ply > 0 && stack_[ply - 1].current != NO_MOVE)
        counter = counter_[stack_[ply - 1].moved][to_sq(stack_[ply - 1].current)];

    for (int i = 0; i < n; ++i) {
        Move m = moves[i];
        int from = from_sq(m), to = to_sq(m);
        if (m == tt_move) {
            scores[i] = 30'000'000;
        } else if (is_capture(m) || move_flag(m) == PROMO_Q) {
            Piece victim = pos.piece_on(to);
            int vv = move_flag(m) == EP_CAPTURE ? PIECE_VALUE[PAWN] : victim == NO_PIECE ? 0 : PIECE_VALUE[type_of(victim)];
            if (is_promotion(m)) vv += PIECE_VALUE[promo_type(m)];
            int s = vv * 16 - type_of(pos.piece_on(from));
            if (Features.capt_history && is_capture(m)) s += capt_entry(pos, m) / 8;
            bool good = !Features.see_pruning || see_ge(pos, m, 0);
            scores[i] = (good ? 20'000'000 : -20'000'000) + s;
        } else if (is_promotion(m)) {
            scores[i] = -25'000'000 + promo_type(m);
        } else if (m == f.killers[0]) {
            scores[i] = 10'000'000;
        } else if (m == f.killers[1]) {
            scores[i] = 9'900'000;
        } else if (m == counter) {
            scores[i] = 9'800'000;
        } else {
            scores[i] = quiet_score(us, m, pos.piece_on(from), ply);
        }
    }
}

void Searcher::update_capture_stats(const Position& pos, Move best, const Move* captures, int nc, int depth) {
    int bonus = std::min(depth * depth * 16, 1600);
    if (best != NO_MOVE) history_update(capt_entry(pos, best), bonus);
    for (int i = 0; i < nc; ++i) history_update(capt_entry(pos, captures[i]), -bonus);
}

void Searcher::update_quiet_stats(const Position& pos, Move best, const Move* quiets, int nq, int depth, int ply) {
    Color us = pos.side_to_move();
    int bonus = std::min(depth * depth * 16, 1600);
    history_update(history_[us][from_sq(best)][to_sq(best)], bonus);
    for (int i = 0; i < nq; ++i)
        if (quiets[i] != best) history_update(history_[us][from_sq(quiets[i])][to_sq(quiets[i])], -bonus);
    if (Features.cont_history) {
        for (int back = 1; back <= 2; ++back) {
            if (int* e = cont_entry(ply - back, pos.piece_on(from_sq(best)), to_sq(best))) history_update(*e, bonus);
            for (int i = 0; i < nq; ++i)
                if (quiets[i] != best)
                    if (int* e = cont_entry(ply - back, pos.piece_on(from_sq(quiets[i])), to_sq(quiets[i])))
                        history_update(*e, -bonus);
        }
    }

    Frame& f = stack_[ply];
    if (f.killers[0] != best) {
        f.killers[1] = f.killers[0];
        f.killers[0] = best;
    }
    if (ply > 0 && stack_[ply - 1].current != NO_MOVE)
        counter_[stack_[ply - 1].moved][to_sq(stack_[ply - 1].current)] = best;
}

Value Searcher::qsearch(Position& pos, Value alpha, Value beta, int ply) {
    ++nodes_;
    if (out_of_time()) return 0;
    seldepth_ = std::max(seldepth_, ply);
    if (ply >= MAX_PLY - 1) return evaluate(pos);
    if (pos.is_draw()) return VALUE_DRAW;

    bool pv = beta - alpha > 1;
    TTEntry tte;
    Move tt_move = NO_MOVE;
    if (TT.probe(pos.key(), tte)) {
        tt_move = tte.move;
        Value tv = score_from_tt(tte.score, ply);
        if (!pv && (tte.bound() == BOUND_EXACT || (tte.bound() == BOUND_LOWER && tv >= beta) ||
                    (tte.bound() == BOUND_UPPER && tv <= alpha)))
            return tv;
    }

    bool in_check = pos.in_check();
    Value best, stand = VALUE_NONE, raw_stand = VALUE_NONE;
    if (in_check) {
        best = -VALUE_MATE + ply;
    } else {
        raw_stand = evaluate(pos);
        stand = corrected(pos, raw_stand);
        if (stand >= beta) return stand;
        alpha = std::max(alpha, stand);
        best = stand;
    }

    MoveList list;
    if (in_check) generate<GEN_ALL>(pos, list);
    else generate<GEN_NOISY>(pos, list);
    int scores[256];
    score_moves(pos, list.moves, list.size, scores, tt_move, ply);

    Value alpha0 = alpha;
    Move best_move = NO_MOVE;
    int legal = 0;
    for (int i = 0; i < list.size; ++i) {
        int bi = i;
        for (int j = i + 1; j < list.size; ++j)
            if (scores[j] > scores[bi]) bi = j;
        std::swap(list.moves[i], list.moves[bi]);
        std::swap(scores[i], scores[bi]);
        Move m = list.moves[i];
        if (!pos.legal(m)) continue;
        ++legal;
        if (!in_check && Features.see_pruning && !see_ge(pos, m, 0)) continue;

        stack_[ply].current = m;
        stack_[ply].moved = pos.piece_on(from_sq(m));
        pos.make_move(m);
        TT.prefetch(pos.key());
        Value v = -qsearch(pos, -beta, -alpha, ply + 1);
        pos.unmake_move(m);
        if (stop_.load(std::memory_order_relaxed)) return 0;

        if (v > best) {
            best = v;
            if (v > alpha) {
                alpha = v;
                best_move = m;
                if (v >= beta) break;
            }
        }
    }
    if (in_check && legal == 0) return -VALUE_MATE + ply;

    Bound b = best >= beta ? BOUND_LOWER : best > alpha0 ? BOUND_EXACT : BOUND_UPPER;
    TT.store(pos.key(), best_move, score_to_tt(best, ply), raw_stand == VALUE_NONE ? 0 : raw_stand, 0, b);
    return best;
}

Value Searcher::negamax(Position& pos, Value alpha, Value beta, int depth, int ply) {
    bool pv = beta - alpha > 1;
    bool root = ply == 0;
    pv_len_[ply] = ply;

    if (depth <= 0) return qsearch(pos, alpha, beta, ply);

    ++nodes_;
    if (out_of_time()) return 0;
    seldepth_ = std::max(seldepth_, ply);

    if (!root) {
        if (pos.is_draw()) return VALUE_DRAW;
        if (ply >= MAX_PLY - 1) return evaluate(pos);
        alpha = std::max(alpha, -VALUE_MATE + ply);
        beta = std::min(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    Move excluded = stack_[ply].excluded;
    TTEntry tte;
    Move tt_move = NO_MOVE;
    bool tt_hit = TT.probe(pos.key(), tte);
    Value tt_value = tt_hit ? score_from_tt(tte.score, ply) : VALUE_NONE;
    if (tt_hit) {
        tt_move = tte.move;
        Value tv = tt_value;
        if (!pv && excluded == NO_MOVE && tte.depth >= depth &&
            (tte.bound() == BOUND_EXACT || (tte.bound() == BOUND_LOWER && tv >= beta) ||
             (tte.bound() == BOUND_UPPER && tv <= alpha)))
            return tv;
    }

    // No TT move at a deep node: search it shallower; the result fills the TT for the next visit.
    if (Features.iir && depth >= 4 && tt_move == NO_MOVE) --depth;

    bool in_check = pos.in_check();
    Value raw_eval = in_check ? VALUE_NONE : tt_hit ? Value(tte.eval) : evaluate(pos);
    Value eval = corrected(pos, raw_eval);
    stack_[ply].static_eval = eval;
    bool improving = !in_check && ply >= 2 && stack_[ply - 2].static_eval != VALUE_NONE &&
                     eval > stack_[ply - 2].static_eval;
    stack_[ply + 1].killers[0] = stack_[ply + 1].killers[1] = NO_MOVE;

    if (!pv && !in_check && excluded == NO_MOVE) {
        if (Features.rfp && depth <= 8 && eval - 75 * (depth - improving) >= beta && eval < VALUE_MATE_IN_MAX_PLY)
            return eval;

        if (Features.razoring && depth <= 3 && eval + 200 + 150 * depth <= alpha) {
            Value v = qsearch(pos, alpha, alpha + 1, ply);
            if (v <= alpha) return v;
        }

        if (Features.null_move && depth >= 3 && eval >= beta && pos.has_non_pawn_material(pos.side_to_move()) &&
            ply > 0 && stack_[ply - 1].current != NO_MOVE) {
            int r = 3 + depth / 4 + std::min((eval - beta) / 200, 3);
            stack_[ply].current = NO_MOVE;
            stack_[ply].moved = NO_PIECE;
            pos.make_null();
            Value v = -negamax(pos, -beta, -beta + 1, depth - 1 - r, ply + 1);
            pos.unmake_null();
            if (stop_.load(std::memory_order_relaxed)) return 0;
            if (v >= beta) return v >= VALUE_MATE_IN_MAX_PLY ? beta : v;
        }
    }

    // The TT move is tried before any generation; most cut-nodes then never generate moves at all.
    MoveList list;
    int scores[256];
    bool tt_first = Features.staged_tt && pos.pseudo_legal(tt_move), tt_done = false, generated = false;
    int next = 0;

    Value best = -VALUE_INFINITE, alpha0 = alpha;
    Move best_move = NO_MOVE;
    Move quiets[256], captures[64];
    int nq = 0, nc = 0, legal = 0;

    for (;;) {
        Move m;
        if (tt_first && !tt_done) {
            m = tt_move;
            tt_done = true;
        } else {
            if (!generated) {
                generate<GEN_ALL>(pos, list);
                score_moves(pos, list.moves, list.size, scores, tt_move, ply);
                generated = true;
                if (tt_first) {
                    for (int k = 0; k < list.size; ++k)
                        if (list.moves[k] == tt_move) {
                            std::swap(list.moves[0], list.moves[k]);
                            std::swap(scores[0], scores[k]);
                            next = 1;
                            break;
                        }
                }
            }
            if (next >= list.size) break;
            int bi = next;
            for (int j = next + 1; j < list.size; ++j)
                if (scores[j] > scores[bi]) bi = j;
            std::swap(list.moves[next], list.moves[bi]);
            std::swap(scores[next], scores[bi]);
            m = list.moves[next++];
        }
        if (m == excluded || !pos.legal(m)) continue;

        bool quiet = !is_capture(m) && !is_promotion(m);

        // Singular extension: if no other move comes close to the TT move's stored score, the TT move is
        // forced and gets searched deeper; if another move also beats beta, the node is cut (multi-cut).
        int extension = 0;
        if (Features.singular && !root && depth >= 8 && m == tt_move && excluded == NO_MOVE &&
            tte.depth >= depth - 3 && tte.bound() != BOUND_UPPER && std::abs(tt_value) < VALUE_MATE_IN_MAX_PLY) {
            Value sbeta = tt_value - 2 * depth;
            stack_[ply].excluded = m;
            Value sv = negamax(pos, sbeta - 1, sbeta, (depth - 1) / 2, ply);
            stack_[ply].excluded = NO_MOVE;
            if (stop_.load(std::memory_order_relaxed)) return 0;
            if (sv < sbeta) extension = 1;
            else if (sbeta >= beta) return sbeta;
            else if (tt_value >= beta) extension = -1;
        }

        if (!root && best > -VALUE_MATE_IN_MAX_PLY && pos.has_non_pawn_material(pos.side_to_move())) {
            if (quiet && Features.lmp && depth <= 6 && nq >= (3 + depth * depth) / (improving ? 1 : 2)) continue;
            if (quiet && Features.futility && !in_check && depth <= 6 && eval + 100 + 90 * depth <= alpha) continue;
            if (Features.see_pruning && depth <= 6 &&
                !see_ge(pos, m, quiet ? -60 * depth : -25 * depth * depth))
                continue;
        }

        ++legal;
        std::uint64_t nodes_before = nodes_;
        stack_[ply].current = m;
        stack_[ply].moved = pos.piece_on(from_sq(m));
        pos.make_move(m);
        TT.prefetch(pos.key());
        bool gives_check = pos.in_check();
        int new_depth = depth - 1 + extension + (Features.check_ext && gives_check && extension <= 0 ? 1 : 0);

        Value v;
        if (legal == 1) {
            v = -negamax(pos, -beta, -alpha, new_depth, ply + 1);
        } else {
            int r = 0;
            if (Features.lmr && depth >= 3 && quiet && legal > (pv ? 3 : 2)) {
                r = lmr_table[std::min(depth, MAX_PLY - 1)][std::min(legal, 63)];
                r += !pv;
                r -= improving;
                r -= gives_check;
                if (Features.cont_history)
                    r -= quiet_score(~pos.side_to_move(), m, stack_[ply].moved, ply) / 16384;
                else
                    r -= history_[~pos.side_to_move()][from_sq(m)][to_sq(m)] / 8192;
                r = std::clamp(r, 0, std::max(new_depth - 1, 0));
            }
            v = -negamax(pos, -alpha - 1, -alpha, new_depth - r, ply + 1);
            if (v > alpha && r > 0) v = -negamax(pos, -alpha - 1, -alpha, new_depth, ply + 1);
            if (v > alpha && v < beta) v = -negamax(pos, -beta, -alpha, new_depth, ply + 1);
        }
        pos.unmake_move(m);
        if (root) {
            int k = 0;
            while (k < root_count_ && root_moves_[k] != m) ++k;
            if (k == root_count_ && root_count_ < 256) {
                root_moves_[root_count_++] = m;
                root_nodes_[k] = 0;
            }
            if (k < 256) root_nodes_[k] += nodes_ - nodes_before;
        }
        if (stop_.load(std::memory_order_relaxed)) return 0;

        if (v > best) {
            best = v;
            if (v > alpha) {
                alpha = v;
                best_move = m;
                if (root) root_best_ = m;
                pv_[ply][ply] = m;
                for (int k = ply + 1; k < pv_len_[ply + 1]; ++k) pv_[ply][k] = pv_[ply + 1][k];
                pv_len_[ply] = std::max(pv_len_[ply + 1], ply + 1);
                if (v >= beta) {
                    if (quiet) update_quiet_stats(pos, m, quiets, nq, depth, ply);
                    if (Features.capt_history) update_capture_stats(pos, is_capture(m) ? m : NO_MOVE, captures, nc, depth);
                    break;
                }
            }
        }
        if (quiet) quiets[nq++] = m;
        else if (is_capture(m) && nc < 64) captures[nc++] = m;
    }

    if (legal == 0) {
        if (excluded != NO_MOVE) return alpha;
        bool any = false;
        for (Move m : list)
            if (pos.legal(m)) { any = true; break; }
        if (!any) return in_check ? -VALUE_MATE + ply : VALUE_DRAW;
        return alpha;
    }

    if (excluded != NO_MOVE) return best;
    Bound b = best >= beta ? BOUND_LOWER : best > alpha0 ? BOUND_EXACT : BOUND_UPPER;
    // Learn how far the static eval missed for this pawn structure, when the result says something about it.
    if (Features.corr_history && !in_check && std::abs(best) < VALUE_MATE_IN_MAX_PLY &&
        (best_move == NO_MOVE || (!is_capture(best_move) && !is_promotion(best_move))) &&
        !(b == BOUND_LOWER && best <= eval) && !(b == BOUND_UPPER && best >= eval)) {
        int& e = corr_entry(pos);
        int w = std::min(depth + 1, 16);
        e = std::clamp((e * (256 - w) + (best - eval) * 256 * w) / 256, -96 * 256, 96 * 256);
    }
    TT.store(pos.key(), best_move, score_to_tt(best, ply), raw_eval == VALUE_NONE ? 0 : raw_eval, depth, b);
    return best;
}

std::uint64_t Searcher::root_nodes_of(Move m) const {
    for (int k = 0; k < root_count_; ++k)
        if (root_moves_[k] == m) return root_nodes_[k];
    return 0;
}

SearchResult Searcher::search(Position& pos, const SearchLimits& limits, bool verbose) {
    start_ = now_ms();
    stop_.store(false);
    nodes_ = 0;
    node_limit_ = limits.nodes;
    setup_time(pos, limits);
    TT.new_search();
    std::memset(stack_, 0, sizeof(stack_));
    for (auto& f : stack_) f.static_eval = VALUE_NONE;
    root_count_ = 0;
    int stability = 0;

    SearchResult result;
    MoveList legal_moves;
    generate_legal(pos, legal_moves);
    if (legal_moves.size == 0) return result;
    result.best = legal_moves.moves[0];

    Value prev = 0;
    for (int depth = 1; depth <= limits.depth; ++depth) {
        seldepth_ = 0;
        root_best_ = NO_MOVE;
        Value v;
        if (Features.aspiration && depth >= 4) {
            int delta = 20;
            Value a = std::max(prev - delta, -VALUE_INFINITE), b = std::min(prev + delta, VALUE_INFINITE);
            for (;;) {
                v = negamax(pos, a, b, depth, 0);
                if (stop_.load()) break;
                if (v <= a) { b = (a + b) / 2; a = std::max(v - delta, -VALUE_INFINITE); }
                else if (v >= b) { b = std::min(v + delta, VALUE_INFINITE); }
                else break;
                delta += delta;
                if (delta > 800) { a = -VALUE_INFINITE; b = VALUE_INFINITE; }
            }
        } else {
            v = negamax(pos, -VALUE_INFINITE, VALUE_INFINITE, depth, 0);
        }

        if (stop_.load()) {
            if (root_best_ != NO_MOVE && depth > 1) result.best = root_best_;
            break;
        }
        Value prev_score = prev;
        Move prev_best = result.best;
        prev = v;
        result.best = pv_len_[0] > 0 ? pv_[0][0] : result.best;
        stability = result.best == prev_best ? std::min(stability + 1, 8) : 0;
        result.score = v;
        result.depth = depth;

        if (verbose) {
            std::int64_t t = elapsed();
            std::ostringstream ss;
            ss << "info depth " << depth << " seldepth " << seldepth_ << " score " << score_to_uci(v) << " nodes "
               << nodes_ << " nps " << (t ? nodes_ * 1000 / t : nodes_) << " hashfull " << TT.hashfull() << " time "
               << t << " pv";
            for (int k = 0; k < pv_len_[0]; ++k) ss << ' ' << move_to_uci(pv_[0][k]);
            std::cout << ss.str() << std::endl;
        }
        // The next iteration typically costs several times this one, so don't start what can't finish.
        // With NodeTM the budget shrinks when the best move holds most of the nodes and stays put, and grows
        // when the score drops.
        double scale = 1.0;
        if (Features.node_tm && depth >= 4 && nodes_ > 0) {
            double frac = double(root_nodes_of(result.best)) / double(nodes_);
            double nodes_f = (1.5 - frac) * 1.35;
            double stab_f = 1.25 - 0.05 * stability;
            double score_f = std::clamp(1.0 + 0.01 * (prev_score - v), 0.85, 1.5);
            scale = nodes_f * stab_f * score_f;
        }
        if (soft_ >= 0 && elapsed() >= soft_ * 2 / 5 * scale) break;
        if (node_limit_ && nodes_ >= node_limit_) break;
    }
    result.nodes = nodes_;
    return result;
}

std::uint64_t bench(Searcher& searcher, int depth) {
    static const char* FENS[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r2q1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9",
        "2r3k1/5ppp/p3p3/1p1nP3/3P4/P4N2/1P3PPP/2R3K1 w - - 0 25",
        "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
        "6k1/5p2/6p1/8/7p/8/6PP/6K1 b - - 0 1",
        "r1b1k2r/ppppnppp/2n2q2/2b5/3NP3/2P1B3/PP3PPP/RN1QKB1R w KQkq - 0 1",
    };
    std::uint64_t total = 0;
    std::int64_t t0 = now_ms();
    for (const char* fen : FENS) {
        Position pos;
        pos.set_fen(fen);
        TT.clear();
        searcher.new_game();
        SearchLimits lim;
        lim.depth = depth;
        total += searcher.search(pos, lim, false).nodes;
    }
    std::int64_t ms = std::max<std::int64_t>(1, now_ms() - t0);
    std::cout << total << " nodes " << total * 1000 / ms << " nps" << std::endl;
    return total;
}

}  // namespace tsurugi
