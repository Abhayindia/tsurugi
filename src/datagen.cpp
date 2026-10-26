#include "datagen.h"

#include "movegen.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "tt.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace tsurugi {

namespace {

constexpr const char* STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int MAX_GAME_PLIES = 400;
constexpr Value MAX_RECORD_SCORE = 2000;
constexpr Value MAX_OPENING_SCORE = 400;
constexpr Value WIN_SCORE = 1500;
constexpr int WIN_PLIES = 4;
constexpr int DRAW_MIN_PLY = 80;
constexpr Value DRAW_SCORE = 10;
constexpr int DRAW_PLIES = 8;

struct Sample {
    std::string fen;
    Value score;  // white's point of view
};

// Plays 8 or 9 random legal plies; false if the game ended on the way.
bool random_opening(Position& pos, std::mt19937_64& rng) {
    pos.set_fen(STARTPOS);
    int plies = 8 + int(rng() % 2);
    for (int i = 0; i < plies; ++i) {
        MoveList list;
        generate_legal(pos, list);
        if (list.size == 0) return false;
        pos.make_move(list.moves[rng() % list.size]);
    }
    MoveList list;
    generate_legal(pos, list);
    return list.size > 0 && !pos.is_draw();
}

const char* result_string(double r) { return r == 1.0 ? "1.0" : r == 0.0 ? "0.0" : "0.5"; }

}  // namespace

void datagen(int games, std::uint64_t nodes, std::uint64_t seed, const std::string& path, bool guided) {
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cout << "info string cannot open " << path << std::endl;
        return;
    }
    std::mt19937_64 rng(seed);
    auto searcher = std::make_unique<Searcher>();
    SearchLimits limits;
    limits.nodes = nodes;

    std::uint64_t total_positions = 0, total_nodes = 0, total_moves = 0;
    // Running mean of the uncertainty and of the clamped raw budgets; rescaling by the latter keeps the average
    // budget exactly `nodes`, so guided and plain generation use the same compute.
    double unc_sum = 75.0, unc_count = 1.0, raw_sum = 0.0;
    guided = guided && Position::nnue_enabled && nnue::has_uncertainty();
    if (guided) std::cerr << "guided datagen: node budget scaled by network uncertainty" << std::endl;
    auto start = std::chrono::steady_clock::now();
    int played = 0;
    while (played < games) {
        Position pos;
        if (!random_opening(pos, rng)) continue;
        TT.clear();
        searcher->new_game();
        searcher->clear_stop();
        if (std::abs(searcher->search(pos, limits, false).score) > MAX_OPENING_SCORE) continue;

        std::vector<Sample> samples;
        double result = 0.5;
        int win_run = 0, draw_run = 0;
        for (int ply = 0;; ++ply) {
            MoveList list;
            generate_legal(pos, list);
            if (list.size == 0) {
                result = pos.in_check() ? (pos.side_to_move() == WHITE ? 0.0 : 1.0) : 0.5;
                break;
            }
            if (pos.is_draw() || ply >= MAX_GAME_PLIES) break;

            searcher->clear_stop();
            SearchLimits move_limits = limits;
            if (guided && !pos.in_check()) {
                int u = std::max(1, nnue::uncertainty(pos.accumulator(), pos.side_to_move(), popcount(pos.pieces())));
                unc_sum += u;
                unc_count += 1.0;
                double raw = std::clamp(double(nodes) * u / (unc_sum / unc_count), nodes / 3.0, nodes * 3.0);
                raw_sum += raw;
                move_limits.nodes = std::uint64_t(raw * double(nodes) * (unc_count - 1.0) / raw_sum);
            }
            SearchResult r = searcher->search(pos, move_limits, false);
            total_nodes += move_limits.nodes;
            ++total_moves;
            Value white = pos.side_to_move() == WHITE ? r.score : -r.score;

            if (!pos.in_check() && !is_capture(r.best) && !is_promotion(r.best) && std::abs(r.score) < MAX_RECORD_SCORE)
                samples.push_back({pos.fen(), white});

            win_run = std::abs(white) >= WIN_SCORE ? win_run + 1 : 0;
            if (win_run >= WIN_PLIES) {
                result = white > 0 ? 1.0 : 0.0;
                break;
            }
            draw_run = pos.game_ply() >= DRAW_MIN_PLY && std::abs(white) <= DRAW_SCORE ? draw_run + 1 : 0;
            if (draw_run >= DRAW_PLIES) break;

            pos.make_move(r.best);
        }

        for (const Sample& s : samples) out << s.fen << " | " << s.score << " | " << result_string(result) << '\n';
        out.flush();
        total_positions += samples.size();
        ++played;
        if (played % 10 == 0 || played == games) {
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cerr << "games " << played << " positions " << total_positions << " pos/s "
                      << int(total_positions / (secs > 0 ? secs : 1)) << " avg_nodes "
                      << (total_moves ? total_nodes / total_moves : 0) << std::endl;
        }
    }
}

}  // namespace tsurugi
