#include "search.h"

#include "eval.h"
#include "movegen.h"
#include "nnue.h"
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
    std::memset(np_corr_, 0, sizeof(np_corr_));
    std::fill(cont_corr_.begin(), cont_corr_.end(), 0);
    std::memset(capt_, 0, sizeof(capt_));
    std::fill(pawn_hist_.begin(), pawn_hist_.end(), 0);
    std::fill(threat_hist_.begin(), threat_hist_.end(), 0);
}

// Every square the side not to move attacks.
static U64 threatened(const Position& pos) {
    Color them = ~pos.side_to_move();
    U64 occ = pos.pieces(), t = 0;
    for (U64 b = pos.pieces(them, PAWN); b;) t |= attacks::pawn[them][pop_lsb(b)];
    for (int pt = KNIGHT; pt <= KING; ++pt)
        for (U64 b = pos.pieces(them, PieceType(pt)); b;) t |= attacks::piece(PieceType(pt), pop_lsb(b), occ);
    return t;
}

static int threat_idx(Color us, U64 threats, Move m) {
    int from = from_sq(m), to = to_sq(m);
    int k = int(us) * 4 + int((threats >> from) & 1) * 2 + int((threats >> to) & 1);
    return (k * SQUARE_NB + from) * SQUARE_NB + to;
}

static int pawn_slot(const Position& pos) {
    U64 h = (pos.pieces(WHITE, PAWN) * 0x9E3779B97F4A7C15ULL) ^ (pos.pieces(BLACK, PAWN) * 0xC2B2AE3D27D4EB4FULL);
    return int(h >> 55);
}

int& Searcher::capt_entry(const Position& pos, Move m) {
    PieceType victim = move_flag(m) == EP_CAPTURE ? PAWN : type_of(pos.piece_on(to_sq(m)));
    return capt_[pos.piece_on(from_sq(m))][to_sq(m)][victim];
}

int& Searcher::corr_entry(const Position& pos) {
    U64 h = (pos.pieces(WHITE, PAWN) * 0x9E3779B97F4A7C15ULL) ^ (pos.pieces(BLACK, PAWN) * 0xC2B2AE3D27D4EB4FULL);
    return corr_[pos.side_to_move()][h >> 50];
}

int& Searcher::nonpawn_corr_entry(const Position& pos, Color side) {
    static constexpr U64 MUL[PIECE_TYPE_NB] = {0, 0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
                                               0xD6E8FEB86659FD93ULL, 0xFF51AFD7ED558CCDULL};
    U64 h = 0;
    for (int pt = KNIGHT; pt <= KING; ++pt) h ^= pos.pieces(side, PieceType(pt)) * MUL[pt];
    return np_corr_[pos.side_to_move()][side][h >> 50];
}

int* Searcher::cont_corr_entry(const Position& pos, int ply) {
    if (ply < 2 || stack_[ply - 1].current == NO_MOVE || stack_[ply - 2].current == NO_MOVE) return nullptr;
    const Frame &a = stack_[ply - 2], &b = stack_[ply - 1];
    int i = ((int(pos.side_to_move()) * PIECE_NB + a.moved) * SQUARE_NB + to_sq(a.current)) * PIECE_NB + b.moved;
    return &cont_corr_[i * SQUARE_NB + to_sq(b.current)];
}

Value Searcher::corrected(const Position& pos, Value raw, int ply) {
    if (!Features.corr_history || raw == VALUE_NONE) return raw;
    int c = corr_entry(pos);
    if (Features.nonpawn_corr) c += (nonpawn_corr_entry(pos, WHITE) + nonpawn_corr_entry(pos, BLACK)) / 2;
    if (Features.cont_corr)
        if (int* e = cont_corr_entry(pos, ply)) c += *e;
    return std::clamp(raw + c / 256, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

// The continuation-history entry for piece p moving to `to`, following the move made `ply` plies up the stack.
int* Searcher::cont_entry(int ply, Piece p, int to) {
    if (ply < 0 || stack_[ply].current == NO_MOVE) return nullptr;
    const Frame& f = stack_[ply];
    return &cont_[((int(f.moved) * SQUARE_NB + to_sq(f.current)) * PIECE_NB + p) * SQUARE_NB + to];
}

int Searcher::quiet_score(Color us, Move m, Piece p, int ply) {
    int s = Features.threat_history ? threat_hist_[threat_idx(us, stack_[ply].threats, m)]
                                    : history_[us][from_sq(m)][to_sq(m)];
    if (Features.pawn_history) s += pawn_hist_[(stack_[ply].pawn_idx * PIECE_NB + p) * SQUARE_NB + to_sq(m)];
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
    if (Features.pawn_history) stack_[ply].pawn_idx = pawn_slot(pos);
    if (Features.threat_history) stack_[ply].threats = threatened(pos);
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
            int ch = (Features.capt_history || Features.capt_hist_see) && is_capture(m) ? capt_entry(pos, m) : 0;
            s += Features.capt_hist_see ? ch / 4 : ch / 8;
            // Captures that usually work out may lose a little material by SEE and still count as good.
            bool good = !Features.see_pruning || see_ge(pos, m, Features.capt_hist_see ? -ch / 48 : 0);
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
    if (Features.threat_history) {
        U64 t = threatened(pos);
        history_update(threat_hist_[threat_idx(us, t, best)], bonus);
        for (int i = 0; i < nq; ++i)
            if (quiets[i] != best) history_update(threat_hist_[threat_idx(us, t, quiets[i])], -bonus);
    }
    if (Features.pawn_history) {
        int* ph = &pawn_hist_[pawn_slot(pos) * PIECE_NB * SQUARE_NB];
        history_update(ph[int(pos.piece_on(from_sq(best))) * SQUARE_NB + to_sq(best)], bonus);
        for (int i = 0; i < nq; ++i)
            if (quiets[i] != best)
                history_update(ph[int(pos.piece_on(from_sq(quiets[i]))) * SQUARE_NB + to_sq(quiets[i])], -bonus);
    }
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
        stand = corrected(pos, raw_stand, ply);
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
        // Skip captures that cannot lift a static eval plus margin above alpha.
        if (!in_check && Features.qs_futility && !is_promotion(m)) {
            Value fbase = stand + 200;
            Piece cap = move_flag(m) == EP_CAPTURE ? make_piece(~pos.side_to_move(), PAWN) : pos.piece_on(to_sq(m));
            Value fv = fbase + SEE_VALUE[type_of(cap)];
            if (fv <= alpha) { best = std::max(best, fv); continue; }
            if (fbase <= alpha && !see_ge(pos, m, 1)) { best = std::max(best, fbase); continue; }
        }
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

// `cut` marks a node expected to fail high: the reduced null-window child of its parent.
Value Searcher::negamax(Position& pos, Value alpha, Value beta, int depth, int ply, bool cut) {
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
        if (Features.cuckoo && alpha < VALUE_DRAW && pos.upcoming_repetition(ply)) {
            alpha = VALUE_DRAW;
            if (alpha >= beta) return alpha;
        }
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
    // Nodes that were once on the principal variation get searched a little wider.
    bool tt_pv = Features.tt_pv && (pv || (tt_hit && tte.is_pv()));

    // No TT move at a deep node: search it shallower; the result fills the TT for the next visit.
    if ((Features.iir || (Features.iir_cut && (pv || cut))) && depth >= 4 && tt_move == NO_MOVE) --depth;

    bool in_check = pos.in_check();
    Value raw_eval = in_check ? VALUE_NONE : tt_hit ? Value(tte.eval) : evaluate(pos);
    Value eval = corrected(pos, raw_eval, ply);
    // The network's own estimate of how wrong this static eval is (cp); -1 when unused or unavailable.
    int unc = -1;
    bool use_unc = Features.unc_prune || Features.unc_lmr || Features.unc_prune2 || Features.unc_lmr2;
    if (use_unc && !in_check && Position::nnue_enabled && nnue::has_uncertainty())
        unc = nnue::uncertainty(pos.accumulator(), pos.side_to_move(), popcount(pos.pieces()));
    int unc_margin = Features.unc_prune && unc > 0 ? unc / 2 : 0;
    // Centred version: about 80 cp is the network's typical error, so margins tighten below it and widen above it.
    if (Features.unc_prune2 && unc >= 0) unc_margin = std::clamp(unc - 80, -60, 120) / 4;
    stack_[ply].static_eval = eval;
    // A quiet move that left the opponent with a worse static eval than ours before it was probably good.
    if (Features.eval_drop_hist && !in_check) {
        Move pm = ply > 0 && excluded == NO_MOVE ? stack_[ply - 1].current : NO_MOVE;
        const Frame& prev = stack_[ply > 0 ? ply - 1 : 0];
        if (pm != NO_MOVE && prev.static_eval != VALUE_NONE && !is_capture(pm) && !is_promotion(pm)) {
            int bonus = std::clamp(-2 * (prev.static_eval + eval), -320, 300) + 120;
            Color them = ~pos.side_to_move();
            if (Features.threat_history) history_update(threat_hist_[threat_idx(them, prev.threats, pm)], bonus);
            else history_update(history_[them][from_sq(pm)][to_sq(pm)], bonus);
        }
        if (Features.threat_history) stack_[ply].threats = threatened(pos);
    }
    bool improving = !in_check && ply >= 2 && stack_[ply - 2].static_eval != VALUE_NONE &&
                     eval > stack_[ply - 2].static_eval;
    stack_[ply + 1].killers[0] = stack_[ply + 1].killers[1] = NO_MOVE;

    if (!pv && !in_check && excluded == NO_MOVE) {
        if (Features.rfp && depth <= 8 && eval - 75 * (depth - improving) - unc_margin >= beta && eval < VALUE_MATE_IN_MAX_PLY)
            return eval;

        if (Features.razoring && depth <= 3 && eval + 200 + 150 * depth + unc_margin <= alpha) {
            Value v = qsearch(pos, alpha, alpha + 1, ply);
            if (v <= alpha) return v;
        }

        if (Features.null_move && depth >= 3 && eval >= beta && pos.has_non_pawn_material(pos.side_to_move()) &&
            ply > 0 && stack_[ply - 1].current != NO_MOVE && (!Features.cut_node || cut)) {
            int r = 3 + depth / 4 + std::min((eval - beta) / 200, 3);
            stack_[ply].current = NO_MOVE;
            stack_[ply].moved = NO_PIECE;
            stack_[ply + 1].double_ext = stack_[ply].double_ext;
            pos.make_null();
            Value v = -negamax(pos, -beta, -beta + 1, depth - 1 - r, ply + 1, !cut);
            pos.unmake_null();
            if (stop_.load(std::memory_order_relaxed)) return 0;
            if (v >= beta) return v >= VALUE_MATE_IN_MAX_PLY ? beta : v;
        }

        // ProbCut: a capture that clears beta by a wide margin in a shallow search almost always would in a full one.
        Value pc_beta = beta + 200;
        if (Features.probcut && depth >= 5 && std::abs(beta) < VALUE_MATE_IN_MAX_PLY &&
            !(tt_hit && tte.depth >= depth - 3 && tt_value != VALUE_NONE && tt_value < pc_beta)) {
            MoveList noisy;
            generate<GEN_NOISY>(pos, noisy);
            int pscores[256];
            score_moves(pos, noisy.moves, noisy.size, pscores, tt_move, ply);
            for (int i = 0; i < noisy.size; ++i) {
                int bi = i;
                for (int j = i + 1; j < noisy.size; ++j)
                    if (pscores[j] > pscores[bi]) bi = j;
                std::swap(noisy.moves[i], noisy.moves[bi]);
                std::swap(pscores[i], pscores[bi]);
                Move m = noisy.moves[i];
                if (!pos.legal(m) || !see_ge(pos, m, pc_beta - eval)) continue;
                stack_[ply].current = m;
                stack_[ply].moved = pos.piece_on(from_sq(m));
                pos.make_move(m);
                Value v = -qsearch(pos, -pc_beta, -pc_beta + 1, ply + 1);
                if (v >= pc_beta) v = -negamax(pos, -pc_beta, -pc_beta + 1, depth - 4, ply + 1, !cut);
                pos.unmake_move(m);
                if (stop_.load(std::memory_order_relaxed)) return 0;
                if (v >= pc_beta) {
                    TT.store(pos.key(), m, score_to_tt(v, ply), raw_eval, depth - 3, BOUND_LOWER, tt_pv);
                    return v;
                }
            }
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
            Value sv = negamax(pos, sbeta - 1, sbeta, (depth - 1) / 2, ply, cut);
            stack_[ply].excluded = NO_MOVE;
            if (stop_.load(std::memory_order_relaxed)) return 0;
            if (sv < sbeta)
                extension = Features.singular_v2 && !pv && sv < sbeta - 20 && stack_[ply].double_ext < 8 ? 2 : 1;
            else if (sbeta >= beta) return sbeta;
            else if (tt_value >= beta) extension = Features.singular_v2 ? -2 : -1;
            else if (Features.singular_v2 && cut) extension = -1;
        }

        if (!root && best > -VALUE_MATE_IN_MAX_PLY && pos.has_non_pawn_material(pos.side_to_move())) {
            if (quiet && Features.lmp && depth <= 6 && nq >= (3 + depth * depth) / (improving ? 1 : 2)) continue;
            if (quiet && Features.futility && !in_check && depth <= 6 && eval + 100 + 90 * depth + unc_margin <= alpha) continue;
            // Quiet moves that have kept failing in similar positions are not worth a shallow search.
            if (quiet && Features.hist_prune && depth <= 4 && legal > 0 &&
                quiet_score(pos.side_to_move(), m, pos.piece_on(from_sq(m)), ply) < -2500 * depth)
                continue;
            if (Features.see_pruning && depth <= 6 &&
                !see_ge(pos, m, quiet ? -60 * depth : -25 * depth * depth))
                continue;
        }

        ++legal;
        std::uint64_t nodes_before = nodes_;
        stack_[ply + 1].double_ext = stack_[ply].double_ext + (extension >= 2);
        stack_[ply].current = m;
        stack_[ply].moved = pos.piece_on(from_sq(m));
        pos.make_move(m);
        TT.prefetch(pos.key());
        bool gives_check = pos.in_check();
        int new_depth = depth - 1 + extension + (Features.check_ext && gives_check && extension <= 0 ? 1 : 0);

        Value v;
        if (legal == 1) {
            v = -negamax(pos, -beta, -alpha, new_depth, ply + 1, !pv && !cut);
        } else {
            int r = 0;
            if (Features.lmr && depth >= 3 && quiet && legal > (pv ? 3 : 2)) {
                r = lmr_table[std::min(depth, MAX_PLY - 1)][std::min(legal, 63)];
                r += !pv;
                r -= improving;
                r -= gives_check;
                if (Features.cut_node) r += 2 * cut;
                r -= tt_pv && !pv;
                if (Features.unc_lmr && unc >= 0) r += (unc < 30) - (unc > 120);
                // Only the extreme quarters: confident below ~35 cp, unsure above ~160 cp.
                if (Features.unc_lmr2 && unc >= 0 && !pv) r += (unc < 35) - (unc > 160);
                if (Features.lmr_terms) {
                    // Trust a deep TT entry, distrust nodes whose TT move is a capture, and reduce less where
                    // the static eval needed a big correction (an uncertain position).
                    r -= tt_hit && tte.depth >= depth;
                    r += tt_move != NO_MOVE && (is_capture(tt_move) || is_promotion(tt_move));
                    r -= raw_eval != VALUE_NONE && std::abs(eval - raw_eval) > 60;
                }
                if (Features.cont_history)
                    r -= quiet_score(~pos.side_to_move(), m, stack_[ply].moved, ply) / 16384;
                else
                    r -= history_[~pos.side_to_move()][from_sq(m)][to_sq(m)] / 8192;
                r = std::clamp(r, 0, std::max(new_depth - 1, 0));
            }
            v = -negamax(pos, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
            if (v > alpha && r > 0) {
                v = -negamax(pos, -alpha - 1, -alpha, new_depth, ply + 1, !cut);
                // The reduced search was wrong about this move: tell continuation history how it really did.
                if (Features.lmr_history && Features.cont_history && quiet) {
                    int bonus = std::min(new_depth * new_depth * 16, 1600);
                    int b = v <= alpha ? -bonus : v >= beta ? bonus : 0;
                    if (b)
                        for (int back = 1; back <= 2; ++back)
                            if (int* e = cont_entry(ply - back, stack_[ply].moved, to_sq(m))) history_update(*e, b);
                }
            }
            if (v > alpha && v < beta) v = -negamax(pos, -beta, -alpha, new_depth, ply + 1, false);
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
                    if (Features.capt_history || Features.capt_hist_see) update_capture_stats(pos, is_capture(m) ? m : NO_MOVE, captures, nc, depth);
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
        int w = std::min(depth + 1, 16);
        auto update = [&](int& e) {
            e = std::clamp((e * (256 - w) + (best - eval) * 256 * w) / 256, -96 * 256, 96 * 256);
        };
        update(corr_entry(pos));
        if (Features.nonpawn_corr) {
            update(nonpawn_corr_entry(pos, WHITE));
            update(nonpawn_corr_entry(pos, BLACK));
        }
        if (Features.cont_corr)
            if (int* e = cont_corr_entry(pos, ply)) update(*e);
    }
    TT.store(pos.key(), best_move, score_to_tt(best, ply), raw_eval == VALUE_NONE ? 0 : raw_eval, depth, b, tt_pv);
    return best;
}

std::uint64_t Searcher::root_nodes_of(Move m) const {
    for (int k = 0; k < root_count_; ++k)
        if (root_moves_[k] == m) return root_nodes_[k];
    return 0;
}

SearchResult Searcher::search(Position& pos, const SearchLimits& limits, bool verbose) {
    start_ = now_ms();
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
                v = negamax(pos, a, b, depth, 0, false);
                if (stop_.load()) break;
                if (v <= a) { b = (a + b) / 2; a = std::max(v - delta, -VALUE_INFINITE); }
                else if (v >= b) { b = std::min(v + delta, VALUE_INFINITE); }
                else break;
                delta += delta;
                if (delta > 800) { a = -VALUE_INFINITE; b = VALUE_INFINITE; }
            }
        } else {
            v = negamax(pos, -VALUE_INFINITE, VALUE_INFINITE, depth, 0, false);
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
        searcher.clear_stop();
        total += searcher.search(pos, lim, false).nodes;
    }
    std::int64_t ms = std::max<std::int64_t>(1, now_ms() - t0);
    std::cout << total << " nodes " << total * 1000 / ms << " nps" << std::endl;
    return total;
}

}  // namespace tsurugi
