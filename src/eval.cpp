#include "eval.h"

#include "psqt.h"

#include <algorithm>
#include <cstdlib>

namespace tsurugi {

namespace {

// Tapered score: midgame and endgame halves, blended by game phase.
struct Score {
    int mg = 0, eg = 0;
};

constexpr int MG_VALUE[PIECE_TYPE_NB] = {90, 320, 335, 460, 940, 0};
constexpr int EG_VALUE[PIECE_TYPE_NB] = {120, 290, 310, 520, 960, 0};
constexpr int TOTAL_PHASE = 24;
constexpr int TEMPO = 12;
constexpr Score BISHOP_PAIR = {25, 45};

Score pst[PIECE_TYPE_NB][SQUARE_NB];
}  // namespace

int PSQ_MG[PIECE_NB][SQUARE_NB];
int PSQ_EG[PIECE_NB][SQUARE_NB];

namespace {

int center_distance(int s) {
    int f = file_of(s), r = rank_of(s);
    return (std::abs(2 * f - 7) + std::abs(2 * r - 7)) / 2;  // 1 (center) .. 7 (corner)
}

void build_psts() {
    static constexpr int PAWN_MG_RANK[8] = {0, 0, 2, 6, 14, 28, 50, 0};
    static constexpr int PAWN_EG_RANK[8] = {0, 0, 8, 18, 34, 60, 100, 0};
    static constexpr int KING_MG_FILE[8] = {20, 30, 10, -5, -5, 10, 30, 20};

    for (int s = 0; s < 64; ++s) {
        int f = file_of(s), r = rank_of(s), dc = center_distance(s);
        bool central_file = f == 3 || f == 4;

        pst[PAWN][s] = {PAWN_MG_RANK[r] + (central_file && (r == 3 || r == 4) ? 12 : 0) - (central_file && r == 1 ? 6 : 0),
                        PAWN_EG_RANK[r]};
        pst[KNIGHT][s] = {20 - 7 * dc, 15 - 6 * dc};
        pst[BISHOP][s] = {10 - 3 * dc + (r == 0 ? -8 : 0), 10 - 3 * dc};
        pst[ROOK][s] = {(r == 6 ? 15 : 0) + (central_file ? 5 : 0), r == 6 ? 10 : 0};
        pst[QUEEN][s] = {4 - dc, 12 - 3 * dc};
        int king_mg = r == 0 ? KING_MG_FILE[f] : r == 1 ? KING_MG_FILE[f] / 2 - 15 : -30 - 12 * r;
        pst[KING][s] = {king_mg, 30 - 9 * dc};
    }
}

}  // namespace

namespace {
void init_masks();
}

void init_eval() {
    build_psts();
    init_masks();
    for (int pt = PAWN; pt <= KING; ++pt) {
        for (int s = 0; s < 64; ++s) {
            Score w = {MG_VALUE[pt] + pst[pt][s].mg, EG_VALUE[pt] + pst[pt][s].eg};
            Piece wp = make_piece(WHITE, PieceType(pt)), bp = make_piece(BLACK, PieceType(pt));
            PSQ_MG[wp][s] = w.mg;
            PSQ_EG[wp][s] = w.eg;
            PSQ_MG[bp][s ^ 56] = -w.mg;
            PSQ_EG[bp][s ^ 56] = -w.eg;
        }
    }
}

namespace {

U64 adjacent_files[8];
U64 passed_mask[COLOR_NB][SQUARE_NB];

constexpr Score MOBILITY_WEIGHT[PIECE_TYPE_NB] = {{0, 0}, {4, 4}, {5, 5}, {2, 4}, {1, 2}, {0, 0}};
constexpr int MOBILITY_CENTER[PIECE_TYPE_NB] = {0, 4, 6, 6, 12, 0};
constexpr int KING_ATTACK_WEIGHT[PIECE_TYPE_NB] = {0, 2, 2, 3, 5, 0};
constexpr int PASSED2_MG[8] = {0, 2, 5, 7, 12, 20, 30, 0};
constexpr int PASSED2_EG[8] = {0, 5, 7, 12, 22, 37, 55, 0};

int square_distance(int a, int b) {
    return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b)));
}

constexpr Score ROOK_OPEN = {25, 10};
constexpr Score ROOK_SEMI_OPEN = {12, 6};

void add(Score& s, Score t, int n = 1) {
    s.mg += t.mg * n;
    s.eg += t.eg * n;
}

void init_masks() {
    for (int f = 0; f < 8; ++f) {
        adjacent_files[f] = 0;
        if (f > 0) adjacent_files[f] |= FILE_A_BB << (f - 1);
        if (f < 7) adjacent_files[f] |= FILE_A_BB << (f + 1);
    }
    for (int s = 0; s < 64; ++s) {
        U64 files = adjacent_files[file_of(s)] | (FILE_A_BB << file_of(s));
        U64 ahead_w = 0, ahead_b = 0;
        for (int r = rank_of(s) + 1; r < 8; ++r) ahead_w |= RANK_1_BB << (8 * r);
        for (int r = rank_of(s) - 1; r >= 0; --r) ahead_b |= RANK_1_BB << (8 * r);
        passed_mask[WHITE][s] = files & ahead_w;
        passed_mask[BLACK][s] = files & ahead_b;
    }
}

U64 pawn_attacks(const Position& pos, Color c) {
    U64 p = pos.pieces(c, PAWN);
    U64 up = c == WHITE ? p << 8 : p >> 8;
    return shift_east(up) | shift_west(up);
}

Score eval_side(const Position& pos, Color c) {
    Score s;
    Color them = ~c;
    U64 occ = pos.pieces();
    U64 own_pawns = pos.pieces(c, PAWN), their_pawns = pos.pieces(them, PAWN);

    if (EvalTerms.mobility || EvalTerms.king_safety) {
        U64 area = ~pos.pieces(c) & ~pawn_attacks(pos, them);
        int ksq = pos.king_sq(them);
        U64 zone = attacks::king[ksq] | bb(ksq);
        int units = 0, attackers = 0;
        for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
            U64 b = pos.pieces(c, PieceType(pt));
            while (b) {
                U64 a = attacks::piece(PieceType(pt), pop_lsb(b), occ);
                if (EvalTerms.mobility) add(s, MOBILITY_WEIGHT[pt], popcount(a & area) - MOBILITY_CENTER[pt]);
                if (a & zone) {
                    ++attackers;
                    units += KING_ATTACK_WEIGHT[pt] * popcount(a & zone);
                }
            }
        }
        if (EvalTerms.king_safety) {
            if (attackers >= 2) s.mg += std::min(units * units * 3 / 2, 400);
            int own_k = pos.king_sq(c);
            int kf = file_of(own_k), dir = c == WHITE ? 8 : -8;
            for (int f = std::max(kf - 1, 0); f <= std::min(kf + 1, 7); ++f) {
                int s1 = own_k + (f - kf) + dir, s2 = s1 + dir;
                if (s1 >= 0 && s1 < 64 && (own_pawns & bb(s1))) s.mg += 12;
                else if (s2 >= 0 && s2 < 64 && (own_pawns & bb(s2))) s.mg += 6;
                else s.mg -= 10;
            }
        }
    }

    if (EvalTerms.passed_pawns2) {
        U64 b = own_pawns;
        int own_k = pos.king_sq(c), their_k = pos.king_sq(them);
        while (b) {
            int sq = pop_lsb(b);
            if (!(passed_mask[c][sq] & their_pawns)) {
                int r = relative_rank(c, sq);
                int stop = sq + (c == WHITE ? 8 : -8);
                s.mg += PASSED2_MG[r];
                s.eg += PASSED2_EG[r] + (r - 1) * (5 * square_distance(their_k, stop) - 2 * square_distance(own_k, stop));
            }
        }
    }

    if (EvalTerms.rook_files) {
        U64 b = pos.pieces(c, ROOK);
        while (b) {
            U64 file = file_bb(pop_lsb(b));
            if (!(file & own_pawns)) add(s, (file & their_pawns) ? ROOK_SEMI_OPEN : ROOK_OPEN);
        }
    }
    return s;
}

}  // namespace

EvalFeatures EvalTerms;

Value evaluate(const Position& pos) {
    if (Position::nnue_enabled) return nnue::evaluate(pos.accumulator(), pos.side_to_move(), popcount(pos.pieces()));
    Score total;
    if (EvalTerms.mobility || EvalTerms.king_safety || EvalTerms.passed_pawns2 || EvalTerms.rook_files) {
        Score w = eval_side(pos, WHITE), b = eval_side(pos, BLACK);
        total.mg = w.mg - b.mg;
        total.eg = w.eg - b.eg;
    }
    int phase = pos.phase();
    total.mg += pos.psq_mg();
    total.eg += pos.psq_eg();
    if (more_than_one(pos.pieces(WHITE, BISHOP))) { total.mg += BISHOP_PAIR.mg; total.eg += BISHOP_PAIR.eg; }
    if (more_than_one(pos.pieces(BLACK, BISHOP))) { total.mg -= BISHOP_PAIR.mg; total.eg -= BISHOP_PAIR.eg; }

    phase = std::min(phase, TOTAL_PHASE);
    int v = (total.mg * phase + total.eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    v = pos.side_to_move() == WHITE ? v : -v;
    return v + TEMPO;
}

}  // namespace tsurugi
