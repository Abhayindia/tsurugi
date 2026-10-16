#pragma once

#include "position.h"

namespace tsurugi {

// Evaluation terms beyond material + PST, switchable for SPRT tests and ablation.
struct EvalFeatures {
    bool mobility = false;
    bool pawn_structure = false;
    bool passed_pawns = false;
    bool king_safety = false;
    bool rook_files = false;
};

extern EvalFeatures EvalTerms;

void init_eval();
Value evaluate(const Position& pos);

constexpr Value PIECE_VALUE[PIECE_TYPE_NB] = {100, 310, 330, 500, 950, 0};

}  // namespace tsurugi
