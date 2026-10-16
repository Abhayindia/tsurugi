#include "perft.h"
#include "position.h"

#include <cstdlib>
#include <iostream>

using namespace tsurugi;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: divide \"<fen>\" <depth>\n";
        return 2;
    }
    Position::init();
    Position pos;
    if (!pos.set_fen(argv[1])) {
        std::cerr << "bad fen\n";
        return 2;
    }
    perft_divide(pos, std::atoi(argv[2]));
}
