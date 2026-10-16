#include "eval.h"
#include "movegen.h"
#include "perft.h"
#include "position.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace tsurugi;

namespace {

struct Case {
    const char* name;
    const char* fen;
    std::vector<std::uint64_t> counts;  // counts[d-1] = perft(d)
    int full_depth;
    int quick_depth;
};

const std::vector<Case> CASES = {
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     {20, 400, 8902, 197281, 4865609, 119060324, 3195901860ULL}, 6, 5},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     {48, 2039, 97862, 4085603, 193690690, 8031647685ULL}, 6, 4},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     {14, 191, 2812, 43238, 674624, 11030083, 178633661}, 7, 6},
    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     {6, 264, 9467, 422333, 15833292, 706045033}, 6, 5},
    {"pos4_mirrored", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
     {6, 264, 9467, 422333, 15833292, 706045033}, 6, 4},
    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     {44, 1486, 62379, 2103487, 89941194}, 5, 4},
    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     {46, 2079, 89890, 3894594, 164075551, 6923051137ULL}, 6, 4},
};

// Walks the tree checking incremental key == recomputed key and that unmake restores the FEN.
bool check_consistency(Position& pos, int depth) {
    if (pos.key() != pos.compute_key()) {
        std::cout << "  key mismatch at " << pos.fen() << '\n';
        return false;
    }
    Position fresh;
    fresh.set_fen(pos.fen());
    if (fresh.psq_mg() != pos.psq_mg() || fresh.psq_eg() != pos.psq_eg() || fresh.phase() != pos.phase()) {
        std::cout << "  incremental eval mismatch at " << pos.fen() << '\n';
        return false;
    }
    if (depth == 0) return true;
    MoveList list;
    generate_legal(pos, list);
    std::string before = pos.fen();
    for (Move m : list) {
        pos.make_move(m);
        bool ok = check_consistency(pos, depth - 1);
        pos.unmake_move(m);
        if (!ok) return false;
        if (pos.fen() != before) {
            std::cout << "  unmake of " << move_to_uci(m) << " broke " << before << '\n';
            return false;
        }
    }
    return true;
}

// pseudo_legal(m) must be true exactly for the moves generate<GEN_ALL> produces, over all 2^16 encodings.
bool check_pseudo_legal(Position& pos, int depth) {
    MoveList list;
    generate<GEN_ALL>(pos, list);
    std::vector<bool> generated(65536, false);
    for (Move m : list) generated[m] = true;
    for (int i = 1; i < 65536; ++i) {
        if (pos.pseudo_legal(Move(i)) != generated[i]) {
            std::cout << "  pseudo_legal mismatch for " << move_to_uci(Move(i)) << " flag " << (i >> 12)
                      << " (generated=" << generated[i] << ") at " << pos.fen() << '\n';
            return false;
        }
    }
    if (depth == 0) return true;
    MoveList legal;
    generate_legal(pos, legal);
    for (Move m : legal) {
        pos.make_move(m);
        bool ok = check_pseudo_legal(pos, depth - 1);
        pos.unmake_move(m);
        if (!ok) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    Position::init();
    init_eval();

    int failures = 0;
    std::uint64_t total_nodes = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (const Case& c : CASES) {
        Position pos;
        if (!pos.set_fen(c.fen)) {
            std::cout << c.name << ": bad FEN\n";
            ++failures;
            continue;
        }
        if (!check_consistency(pos, 3)) {
            std::cout << c.name << ": consistency FAILED\n";
            ++failures;
        }
        if (!check_pseudo_legal(pos, 2)) {
            std::cout << c.name << ": pseudo_legal FAILED\n";
            ++failures;
        }
        int maxd = quick ? c.quick_depth : c.full_depth;
        for (int d = 1; d <= maxd; ++d) {
            std::uint64_t n = perft(pos, d);
            total_nodes += n;
            bool ok = n == c.counts[d - 1];
            if (!ok) ++failures;
            std::cout << c.name << " d" << d << ": " << n << (ok ? "  ok" : "  FAIL expected " + std::to_string(c.counts[d - 1])) << '\n';
        }
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n" << (failures ? "FAILED" : "ALL PASSED") << " (" << failures << " failures), "
              << total_nodes << " nodes in " << ms << " ms";
    if (ms) std::cout << ", " << total_nodes / ms / 1000 << " Mnps";
    std::cout << '\n';
    return failures ? 1 : 0;
}
