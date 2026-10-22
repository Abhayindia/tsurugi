#pragma once

#include "types.h"

#include <cstdint>
#include <string>

namespace tsurugi::nnue {

// Network file (little-endian), written by scripts/nnue/train.py:
//   version 1: "TSRG" | u32 1 | u32 hidden=H | i16 ft_w[768][H] | i16 ft_b[H] | i16 out_w[2H] | i32 out_b
//   version 2: "TSRG" | u32 2 | u32 hidden=H | u32 buckets=B | i16 ft_w[768][H] | i16 ft_b[H]
//              | i16 out_w[B][2H] | i32 out_b[B]
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

struct alignas(64) Accumulator {
    std::int16_t v[COLOR_NB][HIDDEN];  // [perspective][neuron]
};

bool load(const std::string& path);
bool loaded();
const std::string& description();

// The feature row for piece p on square s, seen from perspective c.
inline int feature(Color c, Piece p, int s) {
    return c == WHITE ? int(p) * 64 + s : ((int(p) + 6) % 12) * 64 + (s ^ 56);
}

void refresh(Accumulator& acc, const Piece board[SQUARE_NB]);
void add(Accumulator& acc, Piece p, int s);
void sub(Accumulator& acc, Piece p, int s);
void move(Accumulator& acc, Piece p, int from, int to);

// A move's effect on the inputs: at most two pieces appear and two disappear.
struct Delta {
    Piece add_p[2], sub_p[2];
    int add_s[2], sub_s[2];
    int adds = 0, subs = 0;
    void add(Piece p, int s) { add_p[adds] = p; add_s[adds++] = s; }
    void sub(Piece p, int s) { sub_p[subs] = p; sub_s[subs++] = s; }
};
// dst = src + the delta, in one pass over the accumulator.
void update(const Accumulator& src, Accumulator& dst, const Delta& d);
// `pieces` is the number of pieces on the board, kings included.
Value evaluate(const Accumulator& acc, Color stm, int pieces);

}  // namespace tsurugi::nnue
