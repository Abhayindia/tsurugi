#include "nnue.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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

void update(const Accumulator& src, Accumulator& dst, const Delta& d) {
    for (int c = 0; c < COLOR_NB; ++c) {
        const std::int16_t* __restrict s = src.v[c];
        std::int16_t* __restrict o = dst.v[c];
        const std::int16_t* a0 = net.ft_w[feature(Color(c), d.add_p[0], d.add_s[0])];
        const std::int16_t* s0 = net.ft_w[feature(Color(c), d.sub_p[0], d.sub_s[0])];
        if (d.adds == 1 && d.subs == 1) {
            for (int i = 0; i < HIDDEN; ++i) o[i] = s[i] + a0[i] - s0[i];
        } else if (d.adds == 1 && d.subs == 2) {
            const std::int16_t* s1 = net.ft_w[feature(Color(c), d.sub_p[1], d.sub_s[1])];
            for (int i = 0; i < HIDDEN; ++i) o[i] = s[i] + a0[i] - s0[i] - s1[i];
        } else {
            const std::int16_t* a1 = net.ft_w[feature(Color(c), d.add_p[1], d.add_s[1])];
            const std::int16_t* s1 = net.ft_w[feature(Color(c), d.sub_p[1], d.sub_s[1])];
            for (int i = 0; i < HIDDEN; ++i) o[i] = s[i] + a0[i] + a1[i] - s0[i] - s1[i];
        }
    }
}

Value evaluate(const Accumulator& acc, Color stm) {
    const std::int16_t* us = acc.v[stm];
    const std::int16_t* them = acc.v[~stm];
    std::int32_t sum = 0;
#if defined(__AVX2__)
    // Clipped ReLU on 16 neurons at a time, then multiply-add pairs into 8 int32 lanes (same integer result).
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
    __m256i acc32 = _mm256_setzero_si256();
    for (int i = 0; i < HIDDEN; i += 16) {
        __m256i u = _mm256_load_si256(reinterpret_cast<const __m256i*>(us + i));
        __m256i t = _mm256_load_si256(reinterpret_cast<const __m256i*>(them + i));
        u = _mm256_min_epi16(_mm256_max_epi16(u, zero), qa);
        t = _mm256_min_epi16(_mm256_max_epi16(t, zero), qa);
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(u, _mm256_load_si256(reinterpret_cast<const __m256i*>(net.out_w + i))));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(t, _mm256_load_si256(reinterpret_cast<const __m256i*>(net.out_w + HIDDEN + i))));
    }
    __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(acc32), _mm256_extracti128_si256(acc32, 1));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0xB1));
    sum = _mm_cvtsi128_si32(s128);
#else
    for (int i = 0; i < HIDDEN; ++i) {
        sum += std::clamp<std::int32_t>(us[i], 0, QA) * net.out_w[i];
        sum += std::clamp<std::int32_t>(them[i], 0, QA) * net.out_w[HIDDEN + i];
    }
#endif
    std::int64_t v = (std::int64_t(sum) + net.out_b) * SCALE / (QA * QB);
    return Value(std::clamp<std::int64_t>(v, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1));
}

}  // namespace tsurugi::nnue
