#pragma once

#include "types.h"

#include <bit>

namespace tsurugi {

constexpr U64 FILE_A_BB = 0x0101010101010101ULL;
constexpr U64 FILE_H_BB = FILE_A_BB << 7;
constexpr U64 RANK_1_BB = 0xFFULL;
constexpr U64 RANK_2_BB = RANK_1_BB << 8;
constexpr U64 RANK_3_BB = RANK_1_BB << 16;
constexpr U64 RANK_6_BB = RANK_1_BB << 40;
constexpr U64 RANK_7_BB = RANK_1_BB << 48;
constexpr U64 RANK_8_BB = RANK_1_BB << 56;

constexpr U64 bb(int s) { return 1ULL << s; }
inline int popcount(U64 b) { return std::popcount(b); }
inline int lsb(U64 b) { return std::countr_zero(b); }
inline int msb(U64 b) { return 63 - std::countl_zero(b); }
inline int pop_lsb(U64& b) { int s = lsb(b); b &= b - 1; return s; }
constexpr bool more_than_one(U64 b) { return b & (b - 1); }

constexpr U64 file_bb(int s) { return FILE_A_BB << file_of(s); }
constexpr U64 rank_bb(int s) { return RANK_1_BB << (8 * rank_of(s)); }

template <Color C>
constexpr U64 shift_up(U64 b) { return C == WHITE ? b << 8 : b >> 8; }
constexpr U64 shift_east(U64 b) { return (b & ~FILE_H_BB) << 1; }
constexpr U64 shift_west(U64 b) { return (b & ~FILE_A_BB) >> 1; }

namespace attacks {

extern U64 pawn[COLOR_NB][SQUARE_NB];
extern U64 knight[SQUARE_NB];
extern U64 king[SQUARE_NB];
extern U64 between[SQUARE_NB][SQUARE_NB];  // exclusive of both ends
extern U64 line[SQUARE_NB][SQUARE_NB];     // full line through both squares, or 0

struct Magic {
    U64 mask;
    U64 magic;
    U64* table;
    int shift;
    unsigned index(U64 occ) const { return unsigned(((occ & mask) * magic) >> shift); }
};

extern Magic rook_magics[SQUARE_NB];
extern Magic bishop_magics[SQUARE_NB];

void init();

inline U64 rook(int s, U64 occ) { const Magic& m = rook_magics[s]; return m.table[m.index(occ)]; }
inline U64 bishop(int s, U64 occ) { const Magic& m = bishop_magics[s]; return m.table[m.index(occ)]; }
inline U64 queen(int s, U64 occ) { return rook(s, occ) | bishop(s, occ); }

inline U64 piece(PieceType pt, int s, U64 occ) {
    switch (pt) {
        case KNIGHT: return knight[s];
        case BISHOP: return bishop(s, occ);
        case ROOK:   return rook(s, occ);
        case QUEEN:  return queen(s, occ);
        case KING:   return king[s];
        default:     return 0;
    }
}

}  // namespace attacks

}  // namespace tsurugi
