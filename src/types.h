#pragma once

#include <cstdint>
#include <string>

namespace tsurugi {

using U64 = std::uint64_t;

enum Color : int { WHITE, BLACK, COLOR_NB };

enum PieceType : int { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NB };

enum Piece : int {
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    NO_PIECE, PIECE_NB = 12
};

enum Square : int {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NO_SQ, SQUARE_NB = 64
};

enum CastlingRight : int {
    WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8, ALL_CASTLING = 15
};

constexpr Color operator~(Color c) { return Color(c ^ 1); }
constexpr Piece make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }
constexpr Color color_of(Piece p) { return Color(p / 6); }

constexpr int file_of(int s) { return s & 7; }
constexpr int rank_of(int s) { return s >> 3; }
constexpr int make_square(int f, int r) { return r * 8 + f; }
constexpr int relative_rank(Color c, int s) { return c == WHITE ? rank_of(s) : 7 - rank_of(s); }

// 16-bit move: bits 0-5 from, 6-11 to, 12-15 flag.
using Move = std::uint16_t;
constexpr Move NO_MOVE = 0;

enum MoveFlag : int {
    QUIET = 0, DOUBLE_PUSH = 1, KING_CASTLE = 2, QUEEN_CASTLE = 3,
    CAPTURE = 4, EP_CAPTURE = 5,
    PROMO_N = 8, PROMO_B = 9, PROMO_R = 10, PROMO_Q = 11,
    PROMO_CAP_N = 12, PROMO_CAP_B = 13, PROMO_CAP_R = 14, PROMO_CAP_Q = 15
};

constexpr Move make_move(int from, int to, int flag = QUIET) {
    return Move(from | (to << 6) | (flag << 12));
}
constexpr int from_sq(Move m) { return m & 63; }
constexpr int to_sq(Move m) { return (m >> 6) & 63; }
constexpr int move_flag(Move m) { return m >> 12; }
constexpr bool is_capture(Move m) { return move_flag(m) & CAPTURE; }
constexpr bool is_promotion(Move m) { return move_flag(m) & 8; }
constexpr bool is_castle(Move m) { return move_flag(m) == KING_CASTLE || move_flag(m) == QUEEN_CASTLE; }
constexpr PieceType promo_type(Move m) { return PieceType(KNIGHT + (move_flag(m) & 3)); }

using Value = int;
constexpr Value VALUE_DRAW = 0;
constexpr Value VALUE_MATE = 32000;
constexpr Value VALUE_INFINITE = 32001;
constexpr Value VALUE_NONE = 32002;
constexpr int MAX_PLY = 128;
constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;

std::string square_to_string(int s);
std::string move_to_uci(Move m);

}  // namespace tsurugi
