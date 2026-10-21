#pragma once

#include "types.h"

#include <cstdint>
#include <string>

namespace tsurugi::nnue {

// Network file (little-endian), written by scripts/nnue/train.py:
//   "TSRG" | u32 version=1 | u32 hidden=H | i16 ft_w[768][H] | i16 ft_b[H] | i16 out_w[2H] | i32 out_b
// Inputs: 768 piece-square features per perspective; hidden = clamp(acc, 0, QA); output scaled by SCALE/(QA*QB).
#ifndef TSURUGI_NNUE_HIDDEN
#define TSURUGI_NNUE_HIDDEN 1024
#endif

// H is fixed at build time (-DTSURUGI_NNUE_HIDDEN=...); a file with another size is rejected.
constexpr int HIDDEN = TSURUGI_NNUE_HIDDEN;
constexpr int QA = 255;
constexpr int QB = 64;
constexpr int SCALE = 400;

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
Value evaluate(const Accumulator& acc, Color stm);

}  // namespace tsurugi::nnue
