#include "uci.h"

#include "datagen.h"
#include "eval.h"
#include "nnue.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "tt.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace tsurugi {

namespace {

constexpr const char* NAME = "Tsurugi 0.1-dev";
constexpr const char* AUTHOR = "Abhay Chaudhary";
constexpr const char* STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct Toggle {
    const char* name;
    bool* flag;
};

const Toggle TOGGLES[] = {
    {"NullMove", &Features.null_move},       {"LMR", &Features.lmr},
    {"RFP", &Features.rfp},                  {"Razoring", &Features.razoring},
    {"Futility", &Features.futility},        {"LMP", &Features.lmp},
    {"Aspiration", &Features.aspiration},    {"CheckExtension", &Features.check_ext},
    {"SEEPruning", &Features.see_pruning},   {"CounterMoves", &Features.counter_moves},
    {"StagedTTMove", &Features.staged_tt},   {"ContHistory", &Features.cont_history},
    {"IIR", &Features.iir},   {"Singular", &Features.singular},
    {"CorrHistory", &Features.corr_history}, {"CaptHistory", &Features.capt_history},
    {"NodeTM", &Features.node_tm},
    {"EvalMobility", &EvalTerms.mobility},   {"EvalPawnStructure", &EvalTerms.pawn_structure},
    {"EvalPassedPawns", &EvalTerms.passed_pawns}, {"EvalPassedPawns2", &EvalTerms.passed_pawns2},
    {"EvalKingSafety", &EvalTerms.king_safety},
    {"EvalRookFiles", &EvalTerms.rook_files},
};

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

class Engine {
public:
    Engine() {
        TT.resize(16);
        Position::nnue_enabled = use_nnue_ && nnue::loaded();
        pos_.set_fen(STARTPOS);
    }
    ~Engine() { stop(); }

    void loop() {
        std::string line;
        while (std::getline(std::cin, line))
            if (!command(line)) break;
        stop();
    }

    bool command(const std::string& line) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        if (cmd == "uci") {
            std::cout << "id name " << NAME << "\nid author " << AUTHOR << '\n'
                      << "option name UseNNUE type check default " << (use_nnue_ ? "true" : "false") << '\n'
                      << "option name EvalFile type string default " << nnue::description() << '\n'
                      << "option name Hash type spin default 16 min 1 max 4096\n"
                      << "option name Threads type spin default 1 min 1 max 1\n"
                      << "option name Move Overhead type spin default 50 min 0 max 5000\n";
            for (const Toggle& t : TOGGLES)
                std::cout << "option name " << t.name << " type check default " << (*t.flag ? "true" : "false") << '\n';
            std::cout << "uciok" << std::endl;
        } else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            stop();
            TT.clear();
            searcher_.new_game();
        } else if (cmd == "position") {
            stop();
            position(is);
        } else if (cmd == "go") {
            go(is);
        } else if (cmd == "stop") {
            stop();
        } else if (cmd == "quit") {
            return false;
        } else if (cmd == "setoption") {
            stop();
            setoption(is);
        } else if (cmd == "bench") {
            stop();
            int depth = 8;
            is >> depth;
            bench(searcher_, depth);
        } else if (cmd == "datagen") {
            stop();
            int games = 0;
            std::uint64_t nodes = 5000, seed = 1;
            std::string path;
            is >> games >> nodes >> seed >> path;
            if (games <= 0 || path.empty()) std::cout << "info string usage: datagen <games> <nodes> <seed> <outfile>" << std::endl;
            else datagen(games, nodes, seed, path);
        } else if (cmd == "perft") {
            stop();
            int depth = 1;
            is >> depth;
            perft_divide(pos_, depth);
        } else if (cmd == "d") {
            std::cout << pos_.fen() << std::endl;
        } else if (cmd == "eval") {
            std::cout << "eval " << evaluate(pos_) << " (side to move)" << std::endl;
        }
        return true;
    }

private:
    void stop() {
        searcher_.request_stop();
        if (thread_.joinable()) thread_.join();
    }

    void position(std::istringstream& is) {
        std::string token, fen;
        is >> token;
        if (token == "startpos") {
            fen = STARTPOS;
            is >> token;
        } else if (token == "fen") {
            while (is >> token && token != "moves") fen += token + ' ';
        } else {
            return;
        }
        if (!pos_.set_fen(fen)) {
            std::cout << "info string invalid fen" << std::endl;
            pos_.set_fen(STARTPOS);
            return;
        }
        while (is >> token) {
            Move m = pos_.parse_move(token);
            if (m == NO_MOVE) {
                std::cout << "info string illegal move " << token << std::endl;
                break;
            }
            pos_.make_move(m);
        }
    }

    void go(std::istringstream& is) {
        stop();
        SearchLimits lim;
        std::string token;
        while (is >> token) {
            if (token == "wtime") is >> lim.time[WHITE];
            else if (token == "btime") is >> lim.time[BLACK];
            else if (token == "winc") is >> lim.inc[WHITE];
            else if (token == "binc") is >> lim.inc[BLACK];
            else if (token == "movestogo") is >> lim.movestogo;
            else if (token == "depth") is >> lim.depth;
            else if (token == "nodes") is >> lim.nodes;
            else if (token == "movetime") is >> lim.movetime;
            else if (token == "infinite") lim.infinite = true;
            else if (token == "perft") {
                int d = 1;
                is >> d;
                perft_divide(pos_, d);
                return;
            }
        }
        lim.depth = std::clamp(lim.depth, 1, MAX_PLY - 1);
        search_pos_ = pos_;
        thread_ = std::thread([this, lim] {
            SearchResult r = searcher_.search(search_pos_, lim, true);
            std::cout << "bestmove " << move_to_uci(r.best) << std::endl;
        });
    }

    void setoption(std::istringstream& is) {
        std::string token, name, value;
        is >> token;
        while (is >> token && token != "value") name += (name.empty() ? "" : " ") + token;
        while (is >> token) value += (value.empty() ? "" : " ") + token;
        std::string n = lower(name);
        try {
            if (n == "hash") {
                TT.resize(std::clamp(std::stoi(value), 1, 4096));
                return;
            }
            if (n == "move overhead") {
                searcher_.move_overhead = std::clamp(std::stoi(value), 0, 5000);
                return;
            }
        } catch (const std::exception&) {
            std::cout << "info string bad value for " << name << std::endl;
            return;
        }
        if (n == "threads") return;
        if (n == "evalfile") {
            if (!nnue::load(value)) std::cout << "info string failed to load network " << value << std::endl;
            else std::cout << "info string loaded network " << value << std::endl;
            set_nnue(use_nnue_);
            return;
        }
        if (n == "usennue") {
            set_nnue(lower(value) == "true");
            return;
        }
        for (const Toggle& t : TOGGLES)
            if (n == lower(t.name)) {
                *t.flag = lower(value) == "true";
                return;
            }
        std::cout << "info string unknown option " << name << std::endl;
    }

    void set_nnue(bool on) {
        use_nnue_ = on;
        if (on && !nnue::loaded()) std::cout << "info string UseNNUE requested but no network is loaded" << std::endl;
        Position::nnue_enabled = on && nnue::loaded();
        pos_.set_fen(pos_.fen());  // rebuild accumulators (history is irrelevant: positions are always resent)
    }

    bool use_nnue_ = true;
    Position pos_, search_pos_;
    Searcher searcher_;
    std::thread thread_;
};

}  // namespace

int uci_main(int argc, char** argv) {
    Position::init();
    init_eval();
    std::string exe = argv[0];
    std::string dir = exe.find_last_of("/\\") == std::string::npos ? "." : exe.substr(0, exe.find_last_of("/\\"));
    nnue::load(dir + "/tsurugi.nnue");
    Engine engine;
    if (argc > 1) {
        std::string cmd;
        for (int i = 1; i < argc; ++i) cmd += std::string(i > 1 ? " " : "") + argv[i];
        engine.command(cmd);
        return 0;
    }
    engine.loop();
    return 0;
}

}  // namespace tsurugi
