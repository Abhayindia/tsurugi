#include "perft.h"

#include "movegen.h"

#include <chrono>
#include <iostream>

namespace tsurugi {

std::uint64_t perft(Position& pos, int depth) {
    MoveList list;
    generate_legal(pos, list);
    if (depth <= 1) return depth == 1 ? std::uint64_t(list.size) : 1;
    std::uint64_t nodes = 0;
    for (Move m : list) {
        pos.make_move(m);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(m);
    }
    return nodes;
}

void perft_divide(Position& pos, int depth) {
    auto t0 = std::chrono::steady_clock::now();
    MoveList list;
    generate_legal(pos, list);
    std::uint64_t total = 0;
    for (Move m : list) {
        pos.make_move(m);
        std::uint64_t n = depth > 1 ? perft(pos, depth - 1) : 1;
        pos.unmake_move(m);
        total += n;
        std::cout << move_to_uci(m) << ": " << n << '\n';
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\nNodes searched: " << total << "\nTime: " << ms << " ms\nNPS: "
              << (ms ? total * 1000 / ms : total) << '\n';
}

}  // namespace tsurugi
