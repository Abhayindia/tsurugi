#include "nnue.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>

namespace tsurugi::nnue {

namespace {

struct Network {
    alignas(64) std::int16_t ft_w[768][HIDDEN];
    alignas(64) std::int16_t ft_b[HIDDEN];
    alignas(64) std::int16_t out_w[2 * HIDDEN];
    std::int32_t out_b;
};

Network net;
bool is_loaded = false;
std::string desc = "none";

}  // namespace

bool load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4];
    std::uint32_t version = 0, hidden = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&hidden), 4);
    if (!in || std::memcmp(magic, "TSRG", 4) != 0 || version != 1 || hidden != HIDDEN) return false;
    auto tmp = std::make_unique<Network>();
    in.read(reinterpret_cast<char*>(tmp->ft_w), sizeof(tmp->ft_w));
    in.read(reinterpret_cast<char*>(tmp->ft_b), sizeof(tmp->ft_b));
    in.read(reinterpret_cast<char*>(tmp->out_w), sizeof(tmp->out_w));
    in.read(reinterpret_cast<char*>(&tmp->out_b), sizeof(tmp->out_b));
    if (!in || in.peek() != std::char_traits<char>::eof()) return false;
    net = *tmp;
    is_loaded = true;
    desc = path;
    return true;
}

bool loaded() { return is_loaded; }
const std::string& description() { return desc; }

void refresh(Accumulator& acc, const Piece board[SQUARE_NB]) {
    for (int c = 0; c < COLOR_NB; ++c) std::memcpy(acc.v[c], net.ft_b, sizeof(net.ft_b));
    for (int s = 0; s < 64; ++s)
        if (board[s] != NO_PIECE) add(acc, board[s], s);
}

void add(Accumulator& acc, Piece p, int s) {
    for (int c = 0; c < COLOR_NB; ++c) {
        const std::int16_t* w = net.ft_w[feature(Color(c), p, s)];
        for (int i = 0; i < HIDDEN; ++i) acc.v[c][i] += w[i];
    }
}

void sub(Accumulator& acc, Piece p, int s) {
    for (int c = 0; c < COLOR_NB; ++c) {
        const std::int16_t* w = net.ft_w[feature(Color(c), p, s)];
        for (int i = 0; i < HIDDEN; ++i) acc.v[c][i] -= w[i];
    }
}

void move(Accumulator& acc, Piece p, int from, int to) {
    for (int c = 0; c < COLOR_NB; ++c) {
        const std::int16_t* wf = net.ft_w[feature(Color(c), p, from)];
        const std::int16_t* wt = net.ft_w[feature(Color(c), p, to)];
        for (int i = 0; i < HIDDEN; ++i) acc.v[c][i] += wt[i] - wf[i];
    }
}

Value evaluate(const Accumulator& acc, Color stm) {
    const std::int16_t* us = acc.v[stm];
    const std::int16_t* them = acc.v[~stm];
    std::int32_t sum = 0;
    for (int i = 0; i < HIDDEN; ++i) {
        sum += std::clamp<std::int32_t>(us[i], 0, QA) * net.out_w[i];
        sum += std::clamp<std::int32_t>(them[i], 0, QA) * net.out_w[HIDDEN + i];
    }
    std::int64_t v = (std::int64_t(sum) + net.out_b) * SCALE / (QA * QB);
    return Value(std::clamp<std::int64_t>(v, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1));
}

}  // namespace tsurugi::nnue
