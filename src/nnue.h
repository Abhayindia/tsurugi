#pragma once

#include "types.h"

#include <cstdint>
#include <string>

namespace tsurugi::nnue {

// Network file (little-endian), written by scripts/nnue/train.py:
//   version 1: "TSRG" | u32 1 | u32 hidden=H | i16 ft_w[768][H] | i16 ft_b[H] | i16 out_w[2H] | i32 out_b
//   version 2: "TSRG" | u32 2 | u32 hidden=H | u32 buckets=B | i16 ft_w[768][H] | i16 ft_b[H]
//              | i16 out_w[B][2H] | i32 out_b[B]
//   version 3: version 2 followed by an uncertainty head i16 aux_w[B][2H] | i32 aux_b[B] | i32 aux_q
//              (scripts/nnue/fit_uncertainty.py), predicting the evaluation's error in centipawns.
//   version 4: "TSRG" | u32 4 | u32 hidden=H | u32 buckets=B | u32 king_layout | i16 ft_w[K*768][H] | i16 ft_b[H]
//              | i16 out_w[B][2H] | i32 out_b[B], where K is the number of king buckets of the layout.
//   version 5: version 4 with a u32 activation after king_layout: 0 = clipped ReLU, 1 = SCReLU (clamp(acc, 0, QA)^2,
//              output scaled by SCALE/(QA*QA*QB) and the bias by an extra QA).
//   version 6: version 5 with a u32 aux flag after the activation; if it is 1 the file ends with version 3's
//              uncertainty head (computed on clamp(acc, 0, QA) whatever the activation).
// King layouts: 0 = none (K = 1); 1 = each side sees its king on files a-d (the board is mirrored otherwise) and the
// king square picks one of K = 4 buckets: a1/b1, c1/d1, the second rank, the third rank and beyond.
// Inputs: 768 piece-square features per perspective; hidden = clamp(acc, 0, QA); output scaled by SCALE/(QA*QB).
// With B output buckets the output layer is chosen by the number of pieces on the board.
#ifndef TSURUGI_NNUE_HIDDEN
#define TSURUGI_NNUE_HIDDEN 1536
#endif

// H is fixed at build time (-DTSURUGI_NNUE_HIDDEN=...); a file with another size is rejected.
constexpr int HIDDEN = TSURUGI_NNUE_HIDDEN;
constexpr int QA = 255;
constexpr int QB = 64;
constexpr int SCALE = 400;
constexpr int MAX_BUCKETS = 8;
constexpr int MAX_KING_BUCKETS = 4;

// The loaded network's king layout (see above); 0 for networks without king buckets.
extern int king_layout;

struct alignas(64) Accumulator {
    std::int16_t v[COLOR_NB][HIDDEN];  // [perspective][neuron]
};

bool load(const std::string& path);
bool loaded();
const std::string& description();

// Perspective c's own king square, seen from c (ranks flipped for Black) and mirrored onto files a-d.
inline int king_view(Color c, int ksq) {
    int k = c == WHITE ? ksq : ksq ^ 56;
    return (k & 7) >= 4 ? k ^ 7 : k;
}

inline int king_bucket(int kv) { return kv < 8 ? ((kv & 7) < 2 ? 0 : 1) : kv < 16 ? 2 : 3; }

// Which input weights perspective c uses: a perspective's accumulator is rebuilt when this changes.
inline int king_key(Color c, int ksq) {
    if (king_layout == 0) return 0;
    int k = c == WHITE ? ksq : ksq ^ 56;
    return king_bucket(king_view(c, ksq)) * 2 + ((k & 7) >= 4);
}

// The feature row for piece p on square s, seen from perspective c whose own king stands on ksq.
inline int feature(Color c, Piece p, int s, int ksq) {
    int pc = c == WHITE ? int(p) : (int(p) + 6) % 12;
    int sq = c == WHITE ? s : s ^ 56;
    if (king_layout == 0) return pc * 64 + sq;
    int k = c == WHITE ? ksq : ksq ^ 56;
    if ((k & 7) >= 4) sq ^= 7;
    return king_bucket(king_view(c, ksq)) * 768 + pc * 64 + sq;
}

// Refresh cache ("Finny table"): for each perspective and king key, the accumulator of the board last seen with that
// key. A refresh then applies only the squares that differ from that board instead of rebuilding from every piece.
struct FinnyTable {
    struct alignas(64) Entry {
        std::int16_t v[HIDDEN];
        Piece board[SQUARE_NB];
    };
    Entry e[COLOR_NB][2 * MAX_KING_BUCKETS];
    void reset();  // every entry back to the empty board (call after loading a network)
};
void refresh(Accumulator& acc, const Piece board[SQUARE_NB], FinnyTable& cache);

// A move's effect on the inputs: at most two pieces appear and two disappear.
struct Delta {
    Piece add_p[2], sub_p[2];
    int add_s[2], sub_s[2];
    int adds = 0, subs = 0;
    int ksq[COLOR_NB];  // king squares after the move (they select the input weights)
    void add(Piece p, int s) { add_p[adds] = p; add_s[adds++] = s; }
    void sub(Piece p, int s) { sub_p[subs] = p; sub_s[subs++] = s; }
};
// dst = src + the delta, in one pass over the accumulator.
void update(const Accumulator& src, Accumulator& dst, const Delta& d);
// `pieces` is the number of pieces on the board, kings included.
Value evaluate(const Accumulator& acc, Color stm, int pieces);
// The network's predicted error of its own evaluation, in centipawns; -1 when the network has no uncertainty head.
int uncertainty(const Accumulator& acc, Color stm, int pieces);
bool has_uncertainty();

}  // namespace tsurugi::nnue
