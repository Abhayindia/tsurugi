#pragma once

#include "bitboard.h"
#include "nnue.h"

#include <string>
#include <vector>

namespace tsurugi {

namespace zobrist {
extern U64 psq[PIECE_NB][SQUARE_NB];
extern U64 castling[16];
extern U64 ep[8];
extern U64 side;
}  // namespace zobrist

struct StateInfo {
    U64 key;
    U64 checkers;
    U64 pinned;
    int castling;
    int ep;
    int rule50;
    int plies_from_null;
    Piece captured;
};

class Position {
public:
    static void init();

    bool set_fen(const std::string& fen);
    std::string fen() const;
    Move parse_move(const std::string& uci) const;

    Color side_to_move() const { return stm_; }
    Piece piece_on(int s) const { return board_[s]; }
    U64 pieces() const { return occ_; }
    U64 pieces(Color c) const { return by_color_[c]; }
    U64 pieces(Piece p) const { return by_piece_[p]; }
    U64 pieces(Color c, PieceType pt) const { return by_piece_[make_piece(c, pt)]; }
    U64 pieces(PieceType pt) const { return by_piece_[pt] | by_piece_[pt + 6]; }
    int king_sq(Color c) const { return lsb(pieces(c, KING)); }

    const StateInfo& state() const { return states_.back(); }
    U64 key() const { return state().key; }
    U64 checkers() const { return state().checkers; }
    U64 pinned() const { return state().pinned; }
    bool in_check() const { return state().checkers != 0; }
    int ep_square() const { return state().ep; }
    int castling_rights() const { return state().castling; }
    int rule50() const { return state().rule50; }
    int game_ply() const { return start_ply_ + int(states_.size()) - 1; }

    U64 attackers_to(int s, U64 occ) const;
    bool attacked_by(int s, Color c) const { return attackers_to(s, occ_) & by_color_[c]; }
    bool legal(Move m) const;
    bool pseudo_legal(Move m) const;  // true iff generate<GEN_ALL> would produce m here

    void make_move(Move m);
    void unmake_move(Move m);
    void make_null();
    void unmake_null();

    bool is_draw() const;
    // True if a single reversible move reaches a position seen within the search since the last irreversible move.
    bool upcoming_repetition(int ply) const;
    U64 compute_key() const;
    // True while a network is loaded and UseNNUE is on; accumulators are only maintained then.
    static inline bool nnue_enabled = false;
    const nnue::Accumulator& accumulator() const { return acc_.back(); }

    int psq_mg() const { return psq_mg_; }
    int psq_eg() const { return psq_eg_; }
    int phase() const { return phase_; }

    bool has_non_pawn_material(Color c) const {
        return pieces(c) & ~(pieces(c, PAWN) | pieces(c, KING));
    }

private:
    void put(Piece p, int s);
    void remove(int s);
    void relocate(int from, int to);
    void update_check_info();

    U64 by_piece_[PIECE_NB];
    U64 by_color_[COLOR_NB];
    U64 occ_;
    Piece board_[SQUARE_NB];
    Color stm_;
    int start_ply_;
    int psq_mg_ = 0, psq_eg_ = 0, phase_ = 0;
    std::vector<StateInfo> states_;
    std::vector<nnue::Accumulator> acc_;
};

}  // namespace tsurugi
