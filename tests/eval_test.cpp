#include "eval.h"
#include "movegen.h"
#include "position.h"
#include "search.h"

#include <cctype>
#include <iostream>
#include <sstream>
#include <string>

using namespace tsurugi;

namespace {

char swap_case(char c) {
    unsigned char u = static_cast<unsigned char>(c);
    return std::isupper(u) ? char(std::tolower(u)) : std::islower(u) ? char(std::toupper(u)) : c;
}

// Flips the board vertically and swaps colours; the evaluation must be identical.
std::string mirror_fen(const std::string& fen) {
    std::istringstream ss(fen);
    std::string board, side, castling, ep, rest;
    ss >> board >> side >> castling >> ep;
    std::getline(ss, rest);

    std::string ranks[8], cur;
    int r = 0;
    for (char c : board) {
        if (c == '/') { ranks[r++] = cur; cur.clear(); }
        else cur += c;
    }
    ranks[r] = cur;
    std::string out;
    for (int i = 7; i >= 0; --i) {
        for (char c : ranks[i]) out += swap_case(c);
        if (i) out += '/';
    }
    std::string sorted;
    for (char c : std::string("KQkq"))
        if (castling.find(swap_case(c)) != std::string::npos) sorted += c;
    if (sorted.empty()) sorted = "-";
    if (ep != "-") ep[1] = ep[1] == '3' ? '6' : '3';
    return out + (side == "w" ? " b " : " w ") + sorted + " " + ep + rest;
}

const char* FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "2r3k1/5ppp/p3p3/1p1nP3/3P4/P4N2/1P3PPP/2R3K1 w - - 0 25",
    "6k1/5p2/6p1/8/7p/8/6PP/6K1 b - - 0 1",
    "r1b1k2r/ppppnppp/2n2q2/2b5/3NP3/2P1B3/PP3PPP/RN1QKB1R w KQkq - 0 1",
    "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1",
};

}  // namespace

int main() {
    Position::init();
    init_eval();
    EvalTerms = {true, true, true, true, true};
    int failures = 0;
    for (const char* fen : FENS) {
        Position a, b;
        std::string m = mirror_fen(fen);
        if (!a.set_fen(fen) || !b.set_fen(m)) {
            std::cout << "bad fen " << fen << " / " << m << '\n';
            ++failures;
            continue;
        }
        Value ea = evaluate(a), eb = evaluate(b);
        bool ok = ea == eb;
        if (!ok) ++failures;
        std::cout << (ok ? "ok   " : "FAIL ") << ea << " vs " << eb << "  " << fen << '\n';

        MoveList la, lb;
        generate_legal(a, la);
        generate_legal(b, lb);
        if (la.size != lb.size) {
            std::cout << "FAIL mirrored move count " << la.size << " vs " << lb.size << '\n';
            ++failures;
        }
    }

    Position p;
    p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (!see_ge(p, p.parse_move("e2e4"), 0)) { std::cout << "FAIL see quiet\n"; ++failures; }
    p.set_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
    if (!see_ge(p, p.parse_move("e4d5"), 100)) { std::cout << "FAIL see free pawn\n"; ++failures; }
    p.set_fen("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1");
    if (see_ge(p, p.parse_move("d2d5"), 0)) { std::cout << "FAIL see QxP defended by pawn\n"; ++failures; }
    p.set_fen("4k3/8/2p5/3p4/8/8/3R4/3RK3 w - - 0 1");
    if (see_ge(p, p.parse_move("d2d5"), 0)) { std::cout << "FAIL see RxP defended by pawn\n"; ++failures; }
    p.set_fen("4k3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1");
    if (!see_ge(p, p.parse_move("d2d5"), 100)) { std::cout << "FAIL see RxP, RxR, RxR\n"; ++failures; }

    std::cout << (failures ? "FAILED" : "ALL PASSED") << " (" << failures << " failures)\n";
    return failures ? 1 : 0;
}
