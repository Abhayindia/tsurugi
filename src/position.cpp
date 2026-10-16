#include "position.h"

#include "movegen.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace tsurugi {

namespace zobrist {
U64 psq[PIECE_NB][SQUARE_NB];
U64 castling[16];
U64 ep[8];
U64 side;
}  // namespace zobrist

namespace {

int castle_mask[SQUARE_NB];

constexpr char PIECE_CHARS[] = "PNBRQKpnbrqk";

int up(Color c) { return c == WHITE ? 8 : -8; }

}  // namespace

void Position::init() {
    attacks::init();

    U64 s = 0x9E3779B97F4A7C15ULL;
    auto rnd = [&s] {
        s += 0x9E3779B97F4A7C15ULL;
        U64 z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (auto& p : zobrist::psq)
        for (auto& k : p) k = rnd();
    U64 rights[4] = {rnd(), rnd(), rnd(), rnd()};
    for (int i = 0; i < 16; ++i) {
        zobrist::castling[i] = 0;
        for (int b = 0; b < 4; ++b)
            if (i & (1 << b)) zobrist::castling[i] ^= rights[b];
    }
    for (auto& k : zobrist::ep) k = rnd();
    zobrist::side = rnd();

    for (int& m : castle_mask) m = ALL_CASTLING;
    castle_mask[E1] &= ~(WHITE_OO | WHITE_OOO);
    castle_mask[H1] &= ~WHITE_OO;
    castle_mask[A1] &= ~WHITE_OOO;
    castle_mask[E8] &= ~(BLACK_OO | BLACK_OOO);
    castle_mask[H8] &= ~BLACK_OO;
    castle_mask[A8] &= ~BLACK_OOO;
}

void Position::put(Piece p, int s) {
    board_[s] = p;
    by_piece_[p] |= bb(s);
    by_color_[color_of(p)] |= bb(s);
    occ_ |= bb(s);
}

void Position::remove(int s) {
    Piece p = board_[s];
    by_piece_[p] ^= bb(s);
    by_color_[color_of(p)] ^= bb(s);
    occ_ ^= bb(s);
    board_[s] = NO_PIECE;
}

void Position::relocate(int from, int to) {
    Piece p = board_[from];
    U64 m = bb(from) | bb(to);
    by_piece_[p] ^= m;
    by_color_[color_of(p)] ^= m;
    occ_ ^= m;
    board_[from] = NO_PIECE;
    board_[to] = p;
}

U64 Position::attackers_to(int s, U64 occ) const {
    return (attacks::pawn[BLACK][s] & by_piece_[W_PAWN])
         | (attacks::pawn[WHITE][s] & by_piece_[B_PAWN])
         | (attacks::knight[s] & pieces(KNIGHT))
         | (attacks::bishop(s, occ) & (pieces(BISHOP) | pieces(QUEEN)))
         | (attacks::rook(s, occ) & (pieces(ROOK) | pieces(QUEEN)))
         | (attacks::king[s] & pieces(KING));
}

void Position::update_check_info() {
    StateInfo& st = states_.back();
    Color us = stm_, them = ~stm_;
    int ksq = king_sq(us);
    st.checkers = attackers_to(ksq, occ_) & by_color_[them];
    st.pinned = 0;
    U64 snipers = (attacks::rook(ksq, 0) & (pieces(them, ROOK) | pieces(them, QUEEN)))
                | (attacks::bishop(ksq, 0) & (pieces(them, BISHOP) | pieces(them, QUEEN)));
    while (snipers) {
        int s = pop_lsb(snipers);
        U64 blockers = attacks::between[ksq][s] & occ_;
        if (blockers && !more_than_one(blockers) && (blockers & by_color_[us])) st.pinned |= blockers;
    }
}

U64 Position::compute_key() const {
    U64 k = 0;
    for (int s = 0; s < 64; ++s)
        if (board_[s] != NO_PIECE) k ^= zobrist::psq[board_[s]][s];
    k ^= zobrist::castling[state().castling];
    if (state().ep != NO_SQ) k ^= zobrist::ep[file_of(state().ep)];
    if (stm_ == BLACK) k ^= zobrist::side;
    return k;
}

bool Position::set_fen(const std::string& fen) {
    for (auto& b : by_piece_) b = 0;
    by_color_[WHITE] = by_color_[BLACK] = occ_ = 0;
    for (auto& p : board_) p = NO_PIECE;
    states_.clear();
    states_.reserve(1024);

    std::istringstream ss(fen);
    std::string placement, side, castling = "-", ep = "-";
    int rule50 = 0, fullmove = 1;
    ss >> placement >> side >> castling >> ep >> rule50 >> fullmove;
    if (placement.empty()) return false;

    int f = 0, r = 7;
    for (char c : placement) {
        if (c == '/') { --r; f = 0; continue; }
        if (std::isdigit(static_cast<unsigned char>(c))) { f += c - '0'; continue; }
        const char* p = std::char_traits<char>::find(PIECE_CHARS, 12, c);
        if (!p || f > 7 || r < 0) return false;
        put(Piece(p - PIECE_CHARS), make_square(f++, r));
    }
    if (popcount(pieces(WHITE, KING)) != 1 || popcount(pieces(BLACK, KING)) != 1) return false;

    stm_ = side == "b" ? BLACK : WHITE;

    StateInfo st{};
    st.captured = NO_PIECE;
    st.ep = NO_SQ;
    for (char c : castling) {
        if (c == 'K') st.castling |= WHITE_OO;
        if (c == 'Q') st.castling |= WHITE_OOO;
        if (c == 'k') st.castling |= BLACK_OO;
        if (c == 'q') st.castling |= BLACK_OOO;
    }
    if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && (ep[1] == '3' || ep[1] == '6')) {
        int s = make_square(ep[0] - 'a', ep[1] - '1');
        if (attacks::pawn[~stm_][s] & pieces(stm_, PAWN)) st.ep = s;
    }
    st.rule50 = rule50;
    start_ply_ = std::max(2 * (fullmove - 1), 0) + (stm_ == BLACK);
    states_.push_back(st);
    states_.back().key = compute_key();
    update_check_info();
    return true;
}

std::string Position::fen() const {
    std::string s;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board_[make_square(f, r)];
            if (p == NO_PIECE) { ++empty; continue; }
            if (empty) { s += char('0' + empty); empty = 0; }
            s += PIECE_CHARS[p];
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    s += stm_ == WHITE ? " w " : " b ";
    int cr = castling_rights();
    if (cr & WHITE_OO) s += 'K';
    if (cr & WHITE_OOO) s += 'Q';
    if (cr & BLACK_OO) s += 'k';
    if (cr & BLACK_OOO) s += 'q';
    if (!cr) s += '-';
    s += ' ';
    s += ep_square() == NO_SQ ? "-" : square_to_string(ep_square());
    s += ' ' + std::to_string(rule50()) + ' ' + std::to_string(game_ply() / 2 + 1);
    return s;
}

bool Position::legal(Move m) const {
    Color us = stm_, them = ~stm_;
    int from = from_sq(m), to = to_sq(m), ksq = king_sq(us);

    if (move_flag(m) == EP_CAPTURE) {
        int cap = to - up(us);
        U64 occ = (occ_ ^ bb(from) ^ bb(cap)) | bb(to);
        return !(attackers_to(ksq, occ) & by_color_[them] & ~bb(cap));
    }
    if (from == ksq) {
        if (is_castle(m)) return true;
        return !(attackers_to(to, occ_ ^ bb(ksq)) & by_color_[them]);
    }
    U64 chk = checkers();
    if (chk) {
        if (more_than_one(chk)) return false;
        if (!(bb(to) & (attacks::between[ksq][lsb(chk)] | chk))) return false;
    }
    return !(pinned() & bb(from)) || (attacks::line[ksq][from] & bb(to));
}

void Position::make_move(Move m) {
    StateInfo ns = states_.back();
    Color us = stm_, them = ~stm_;
    int from = from_sq(m), to = to_sq(m), flag = move_flag(m);
    Piece pc = board_[from];
    U64 k = ns.key;

    ns.rule50++;
    ns.plies_from_null++;
    ns.captured = NO_PIECE;
    if (ns.ep != NO_SQ) {
        k ^= zobrist::ep[file_of(ns.ep)];
        ns.ep = NO_SQ;
    }

    if (flag == EP_CAPTURE) {
        int cap = to - up(us);
        ns.captured = board_[cap];
        k ^= zobrist::psq[ns.captured][cap];
        remove(cap);
    } else if (flag & CAPTURE) {
        ns.captured = board_[to];
        k ^= zobrist::psq[ns.captured][to];
        remove(to);
    }
    if (ns.captured != NO_PIECE) ns.rule50 = 0;

    if (is_castle(m)) {
        int rf = flag == KING_CASTLE ? to + 1 : to - 2;
        int rt = flag == KING_CASTLE ? to - 1 : to + 1;
        Piece rook = board_[rf];
        k ^= zobrist::psq[rook][rf] ^ zobrist::psq[rook][rt];
        relocate(rf, rt);
    }

    k ^= zobrist::psq[pc][from] ^ zobrist::psq[pc][to];
    relocate(from, to);

    if (type_of(pc) == PAWN) {
        ns.rule50 = 0;
        if (flag == DOUBLE_PUSH) {
            int eps = from + up(us);
            if (attacks::pawn[us][eps] & pieces(them, PAWN)) {
                ns.ep = eps;
                k ^= zobrist::ep[file_of(eps)];
            }
        } else if (is_promotion(m)) {
            Piece promo = make_piece(us, promo_type(m));
            remove(to);
            put(promo, to);
            k ^= zobrist::psq[pc][to] ^ zobrist::psq[promo][to];
        }
    }

    k ^= zobrist::castling[ns.castling];
    ns.castling &= castle_mask[from] & castle_mask[to];
    k ^= zobrist::castling[ns.castling];

    k ^= zobrist::side;
    ns.key = k;
    stm_ = them;
    states_.push_back(ns);
    update_check_info();
}

void Position::unmake_move(Move m) {
    stm_ = ~stm_;
    Color us = stm_;
    int from = from_sq(m), to = to_sq(m), flag = move_flag(m);
    const StateInfo& st = states_.back();

    if (is_promotion(m)) {
        remove(to);
        put(make_piece(us, PAWN), to);
    }
    relocate(to, from);
    if (is_castle(m)) {
        int rf = flag == KING_CASTLE ? to + 1 : to - 2;
        int rt = flag == KING_CASTLE ? to - 1 : to + 1;
        relocate(rt, rf);
    }
    if (st.captured != NO_PIECE) put(st.captured, flag == EP_CAPTURE ? to - up(us) : to);
    states_.pop_back();
}

void Position::make_null() {
    StateInfo ns = states_.back();
    if (ns.ep != NO_SQ) {
        ns.key ^= zobrist::ep[file_of(ns.ep)];
        ns.ep = NO_SQ;
    }
    ns.key ^= zobrist::side;
    ns.rule50++;
    ns.plies_from_null = 0;
    ns.captured = NO_PIECE;
    stm_ = ~stm_;
    states_.push_back(ns);
    update_check_info();
}

void Position::unmake_null() {
    stm_ = ~stm_;
    states_.pop_back();
}

bool Position::is_draw() const {
    const StateInfo& st = states_.back();
    if (st.rule50 >= 100) {
        if (!st.checkers) return true;
        MoveList list;
        generate_legal(*this, list);
        return list.size > 0;
    }
    int end = std::min(st.rule50, st.plies_from_null);
    int n = int(states_.size()) - 1;
    for (int i = 4; i <= end && n - i >= 0; i += 2)
        if (states_[n - i].key == st.key) return true;

    U64 minors = pieces(KNIGHT) | pieces(BISHOP);
    return !(pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) && popcount(minors) <= 1;
}

Move Position::parse_move(const std::string& uci) const {
    MoveList list;
    generate_legal(*this, list);
    for (Move m : list)
        if (move_to_uci(m) == uci) return m;
    return NO_MOVE;
}

}  // namespace tsurugi
