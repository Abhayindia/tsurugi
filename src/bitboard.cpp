#include "bitboard.h"

#include <vector>

namespace tsurugi {

namespace attacks {

U64 pawn[COLOR_NB][SQUARE_NB];
U64 knight[SQUARE_NB];
U64 king[SQUARE_NB];
U64 between[SQUARE_NB][SQUARE_NB];
U64 line[SQUARE_NB][SQUARE_NB];
Magic rook_magics[SQUARE_NB];
Magic bishop_magics[SQUARE_NB];

namespace {

U64 rook_table[0x19000];
U64 bishop_table[0x1480];

constexpr int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
constexpr int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

U64 slide(int s, U64 occ, const int dirs[4][2]) {
    U64 result = 0;
    for (int d = 0; d < 4; ++d) {
        int f = file_of(s) + dirs[d][0], r = rank_of(s) + dirs[d][1];
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int t = make_square(f, r);
            result |= bb(t);
            if (occ & bb(t)) break;
            f += dirs[d][0];
            r += dirs[d][1];
        }
    }
    return result;
}

U64 step(int s, const int (*deltas)[2], int n) {
    U64 result = 0;
    for (int i = 0; i < n; ++i) {
        int f = file_of(s) + deltas[i][0], r = rank_of(s) + deltas[i][1];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) result |= bb(make_square(f, r));
    }
    return result;
}

struct Rng {
    U64 s;
    U64 next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL; }
    U64 sparse() { return next() & next() & next(); }
};

void init_magics(Magic* magics, U64* table, const int dirs[4][2]) {
    Rng rng{0x7375727567693131ULL};
    std::vector<U64> occs(4096), refs(4096);
    std::vector<int> epoch(4096, 0);
    int attempt = 0;
    U64* next_table = table;

    for (int s = 0; s < 64; ++s) {
        U64 edges = ((RANK_1_BB | RANK_8_BB) & ~rank_bb(s)) | ((FILE_A_BB | FILE_H_BB) & ~file_bb(s));
        Magic& m = magics[s];
        m.mask = slide(s, 0, dirs) & ~edges;
        m.shift = 64 - popcount(m.mask);
        m.table = next_table;
        int size = 0;

        U64 sub = 0;
        do {
            occs[size] = sub;
            refs[size] = slide(s, sub, dirs);
            ++size;
            sub = (sub - m.mask) & m.mask;
        } while (sub);

#if defined(TSURUGI_PEXT)
        for (int i = 0; i < size; ++i) m.table[m.index(occs[i])] = refs[i];
        (void)rng;
        (void)attempt;
        next_table += size;
        continue;
#endif
        for (;;) {
            m.magic = rng.sparse();
            if (popcount((m.mask * m.magic) >> 56) < 6) continue;
            ++attempt;
            bool ok = true;
            for (int i = 0; i < size && ok; ++i) {
                unsigned idx = m.index(occs[i]);
                if (epoch[idx] < attempt) {
                    epoch[idx] = attempt;
                    m.table[idx] = refs[i];
                } else if (m.table[idx] != refs[i]) {
                    ok = false;
                }
            }
            if (ok) break;
        }
        next_table += size;
    }
}

}  // namespace

void init() {
    static constexpr int KNIGHT_D[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    static constexpr int KING_D[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    static constexpr int WPAWN_D[2][2] = {{-1, 1}, {1, 1}};
    static constexpr int BPAWN_D[2][2] = {{-1, -1}, {1, -1}};

    for (int s = 0; s < 64; ++s) {
        knight[s] = step(s, KNIGHT_D, 8);
        king[s] = step(s, KING_D, 8);
        pawn[WHITE][s] = step(s, WPAWN_D, 2);
        pawn[BLACK][s] = step(s, BPAWN_D, 2);
    }

    init_magics(rook_magics, rook_table, ROOK_DIRS);
    init_magics(bishop_magics, bishop_table, BISHOP_DIRS);

    for (int a = 0; a < 64; ++a) {
        for (int b = 0; b < 64; ++b) {
            between[a][b] = line[a][b] = 0;
            if (a == b) continue;
            for (const auto* dirs : {ROOK_DIRS, BISHOP_DIRS}) {
                if (slide(a, 0, dirs) & bb(b)) {
                    line[a][b] = (slide(a, 0, dirs) & slide(b, 0, dirs)) | bb(a) | bb(b);
                    between[a][b] = slide(a, bb(b), dirs) & slide(b, bb(a), dirs);
                }
            }
        }
    }
}

}  // namespace attacks

std::string square_to_string(int s) {
    return {char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string move_to_uci(Move m) {
    if (m == NO_MOVE) return "0000";
    std::string s = square_to_string(from_sq(m)) + square_to_string(to_sq(m));
    if (is_promotion(m)) s += "nbrq"[promo_type(m) - KNIGHT];
    return s;
}

}  // namespace tsurugi
