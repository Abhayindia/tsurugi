#pragma once

#include "position.h"

namespace tsurugi {

void init_eval();
Value evaluate(const Position& pos);

constexpr Value PIECE_VALUE[PIECE_TYPE_NB] = {100, 310, 330, 500, 950, 0};

}  // namespace tsurugi
