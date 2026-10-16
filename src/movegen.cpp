#include "movegen.h"

#include "position.h"

namespace tsurugi {

namespace {

template <GenType T>
void add_promotions(MoveList& list, int from, int to, bool capture) {
    int base = capture ? PROMO_CAP_N : PROMO_N;
    if (T != GEN_QUIET) list.add(make_move(from, to, base + 3));
    if (T != GEN_NOISY)
        for (int p = 0; p < 3; ++p) list.add(make_move(from, to, base + p));
}

template <GenType T>
void gen_pawns(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    int up = us == WHITE ? 8 : -8;
    U64 empty = ~pos.pieces();
    U64 them = pos.pieces(~us);
    U64 promo_from = us == WHITE ? RANK_7_BB : RANK_2_BB;
    U64 double_rank = us == WHITE ? RANK_2_BB : RANK_7_BB;
    int ep = pos.ep_square();

    U64 pawns = pos.pieces(us, PAWN);
    while (pawns) {
        int from = pop_lsb(pawns);
        int to = from + up;
        U64 caps = attacks::pawn[us][from] & them;

        if (bb(from) & promo_from) {
            if (bb(to) & empty) add_promotions<T>(list, from, to, false);
            while (caps) add_promotions<T>(list, from, pop_lsb(caps), true);
            continue;
        }
        if (T != GEN_QUIET) {
            while (caps) list.add(make_move(from, pop_lsb(caps), CAPTURE));
            if (ep != NO_SQ && (attacks::pawn[us][from] & bb(ep))) list.add(make_move(from, ep, EP_CAPTURE));
        }
        if (T != GEN_NOISY && (bb(to) & empty)) {
            list.add(make_move(from, to));
            if ((bb(from) & double_rank) && (bb(to + up) & empty)) list.add(make_move(from, to + up, DOUBLE_PUSH));
        }
    }
}

template <GenType T>
void gen_pieces(const Position& pos, MoveList& list, PieceType pt) {
    Color us = pos.side_to_move();
    U64 them = pos.pieces(~us), empty = ~pos.pieces(), occ = pos.pieces();
    U64 b = pos.pieces(us, pt);
    while (b) {
        int from = pop_lsb(b);
        U64 a = attacks::piece(pt, from, occ);
        if (T != GEN_QUIET)
            for (U64 c = a & them; c;) list.add(make_move(from, pop_lsb(c), CAPTURE));
        if (T != GEN_NOISY)
            for (U64 q = a & empty; q;) list.add(make_move(from, pop_lsb(q)));
    }
}

void gen_castling(const Position& pos, MoveList& list) {
    Color us = pos.side_to_move();
    int cr = pos.castling_rights();
    U64 occ = pos.pieces();
    Color them = ~us;
    if (us == WHITE) {
        if ((cr & WHITE_OO) && !(occ & (bb(F1) | bb(G1))) && !pos.attacked_by(F1, them) && !pos.attacked_by(G1, them))
            list.add(make_move(E1, G1, KING_CASTLE));
        if ((cr & WHITE_OOO) && !(occ & (bb(B1) | bb(C1) | bb(D1))) && !pos.attacked_by(D1, them) && !pos.attacked_by(C1, them))
            list.add(make_move(E1, C1, QUEEN_CASTLE));
    } else {
        if ((cr & BLACK_OO) && !(occ & (bb(F8) | bb(G8))) && !pos.attacked_by(F8, them) && !pos.attacked_by(G8, them))
            list.add(make_move(E8, G8, KING_CASTLE));
        if ((cr & BLACK_OOO) && !(occ & (bb(B8) | bb(C8) | bb(D8))) && !pos.attacked_by(D8, them) && !pos.attacked_by(C8, them))
            list.add(make_move(E8, C8, QUEEN_CASTLE));
    }
}

}  // namespace

template <GenType T>
void generate(const Position& pos, MoveList& list) {
    if (!more_than_one(pos.checkers())) {
        gen_pawns<T>(pos, list);
        for (PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN}) gen_pieces<T>(pos, list, pt);
    }
    gen_pieces<T>(pos, list, KING);
    if (T != GEN_NOISY && !pos.in_check()) gen_castling(pos, list);
}

template void generate<GEN_NOISY>(const Position&, MoveList&);
template void generate<GEN_QUIET>(const Position&, MoveList&);
template void generate<GEN_ALL>(const Position&, MoveList&);

void generate_legal(const Position& pos, MoveList& list) {
    MoveList pseudo;
    generate<GEN_ALL>(pos, pseudo);
    for (Move m : pseudo)
        if (pos.legal(m)) list.add(m);
}

}  // namespace tsurugi
