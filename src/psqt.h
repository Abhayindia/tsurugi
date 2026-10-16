#pragma once

#include "types.h"

namespace tsurugi {

// Material + piece-square values per piece and square (Black already negated), filled by init_eval().
extern int PSQ_MG[PIECE_NB][SQUARE_NB];
extern int PSQ_EG[PIECE_NB][SQUARE_NB];

constexpr int PHASE_WEIGHT[PIECE_TYPE_NB] = {0, 1, 1, 2, 4, 0};

}  // namespace tsurugi
