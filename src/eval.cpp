#include "eval.h"

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
constexpr int PHASE_WEIGHT[PIECE_TYPE_NB] = {0, 1, 1, 2, 4, 0};
constexpr int TOTAL_PHASE = 24;
constexpr int TEMPO = 12;
constexpr Score BISHOP_PAIR = {25, 45};

Score pst[PIECE_TYPE_NB][SQUARE_NB];
Score table[PIECE_NB][SQUARE_NB];

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

void init_eval() {
    build_psts();
    for (int pt = PAWN; pt <= KING; ++pt) {
        for (int s = 0; s < 64; ++s) {
            Score w = {MG_VALUE[pt] + pst[pt][s].mg, EG_VALUE[pt] + pst[pt][s].eg};
            table[make_piece(WHITE, PieceType(pt))][s] = w;
            table[make_piece(BLACK, PieceType(pt))][s ^ 56] = {-w.mg, -w.eg};
        }
    }
}

Value evaluate(const Position& pos) {
    Score total;
    int phase = 0;
    for (int p = W_PAWN; p <= B_KING; ++p) {
        U64 b = pos.pieces(Piece(p));
        phase += PHASE_WEIGHT[type_of(Piece(p))] * popcount(b);
        while (b) {
            int s = pop_lsb(b);
            total.mg += table[p][s].mg;
            total.eg += table[p][s].eg;
        }
    }
    if (more_than_one(pos.pieces(WHITE, BISHOP))) { total.mg += BISHOP_PAIR.mg; total.eg += BISHOP_PAIR.eg; }
    if (more_than_one(pos.pieces(BLACK, BISHOP))) { total.mg -= BISHOP_PAIR.mg; total.eg -= BISHOP_PAIR.eg; }

    phase = std::min(phase, TOTAL_PHASE);
    int v = (total.mg * phase + total.eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    v = pos.side_to_move() == WHITE ? v : -v;
    return v + TEMPO;
}

}  // namespace tsurugi
