#include "nnue.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tsurugi::nnue {

namespace {

struct Network {
    alignas(64) std::int16_t ft_w[768 * MAX_KING_BUCKETS][HIDDEN];
    alignas(64) std::int16_t ft_b[HIDDEN];
    alignas(64) std::int16_t out_w[MAX_BUCKETS][2 * HIDDEN];
    std::int32_t out_b[MAX_BUCKETS];
    alignas(64) std::int16_t aux_w[MAX_BUCKETS][2 * HIDDEN];
    std::int32_t aux_b[MAX_BUCKETS];
    std::int32_t aux_q;
    int buckets;
    bool has_aux;
    bool screlu;
    bool screlu_fast;  // every bucket's worst-case SCReLU sum fits in int32
};

Network net;
bool is_loaded = false;
std::string desc = "none";

}  // namespace

int king_layout = 0;

bool load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4];
    std::uint32_t version = 0, hidden = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&hidden), 4);
    if (!in || std::memcmp(magic, "TSRG", 4) != 0 || (version < 1 || version > 6) || hidden != HIDDEN) return false;
    std::uint32_t buckets = 1, layout = 0, activation = 0, aux = version == 3;
    if (version >= 2) in.read(reinterpret_cast<char*>(&buckets), 4);
    if (version >= 4) in.read(reinterpret_cast<char*>(&layout), 4);
    if (version >= 5) in.read(reinterpret_cast<char*>(&activation), 4);
    if (version == 6) in.read(reinterpret_cast<char*>(&aux), 4);
    if (!in || buckets < 1 || buckets > MAX_BUCKETS || layout > 1 || activation > 1 || aux > 1) return false;
    int king_buckets = layout == 1 ? 4 : 1;
    auto tmp = std::make_unique<Network>();
    tmp->buckets = int(buckets);
    tmp->screlu = activation == 1;
    in.read(reinterpret_cast<char*>(tmp->ft_w), std::streamsize(sizeof(std::int16_t) * 768 * king_buckets * HIDDEN));
    in.read(reinterpret_cast<char*>(tmp->ft_b), sizeof(tmp->ft_b));
    for (std::uint32_t b = 0; b < buckets; ++b) in.read(reinterpret_cast<char*>(tmp->out_w[b]), sizeof(tmp->out_w[b]));
    in.read(reinterpret_cast<char*>(tmp->out_b), std::streamsize(sizeof(std::int32_t) * buckets));
    tmp->has_aux = aux == 1;
    if (tmp->has_aux) {
        for (std::uint32_t b = 0; b < buckets; ++b) in.read(reinterpret_cast<char*>(tmp->aux_w[b]), sizeof(tmp->aux_w[b]));
        in.read(reinterpret_cast<char*>(tmp->aux_b), std::streamsize(sizeof(std::int32_t) * buckets));
        in.read(reinterpret_cast<char*>(&tmp->aux_q), 4);
        if (tmp->aux_q <= 0) return false;
    }
    if (!in || in.peek() != std::char_traits<char>::eof()) return false;
    tmp->screlu_fast = true;
    for (std::uint32_t b = 0; b < buckets; ++b) {
        std::int64_t bound = 0;
        for (int i = 0; i < 2 * HIDDEN; ++i) bound += std::int64_t(QA) * QA * std::abs(int(tmp->out_w[b][i]));
        if (bound > INT32_MAX) tmp->screlu_fast = false;
    }
    net = *tmp;
    king_layout = int(layout);
    is_loaded = true;
    desc = path;
    return true;
}

bool loaded() { return is_loaded; }
const std::string& description() { return desc; }

void refresh(Accumulator& acc, const Piece board[SQUARE_NB]) {
    int ksq[COLOR_NB] = {0, 0};
    for (int s = 0; s < 64; ++s) {
        if (board[s] == W_KING) ksq[WHITE] = s;
        if (board[s] == B_KING) ksq[BLACK] = s;
    }
    for (int c = 0; c < COLOR_NB; ++c) {
        std::memcpy(acc.v[c], net.ft_b, sizeof(net.ft_b));
        for (int s = 0; s < 64; ++s) {
            if (board[s] == NO_PIECE) continue;
            const std::int16_t* w = net.ft_w[feature(Color(c), board[s], s, ksq[c])];
            for (int i = 0; i < HIDDEN; ++i) acc.v[c][i] += w[i];
        }
    }
}

void update(const Accumulator& src, Accumulator& dst, const Delta& d) {
    for (int c = 0; c < COLOR_NB; ++c) {
        const std::int16_t* __restrict s = src.v[c];
        std::int16_t* __restrict o = dst.v[c];
        const std::int16_t* a0 = net.ft_w[feature(Color(c), d.add_p[0], d.add_s[0], d.ksq[c])];
        const std::int16_t* s0 = net.ft_w[feature(Color(c), d.sub_p[0], d.sub_s[0], d.ksq[c])];
        if (d.adds == 1 && d.subs == 1) {
            for (int i = 0; i < HIDDEN; ++i) o[i] = s[i] + a0[i] - s0[i];
        } else if (d.adds == 1 && d.subs == 2) {
            const std::int16_t* s1 = net.ft_w[feature(Color(c), d.sub_p[1], d.sub_s[1], d.ksq[c])];
            for (int i = 0; i < HIDDEN; ++i) o[i] = s[i] + a0[i] - s0[i] - s1[i];
        } else {
            const std::int16_t* a1 = net.ft_w[feature(Color(c), d.add_p[1], d.add_s[1], d.ksq[c])];
            const std::int16_t* s1 = net.ft_w[feature(Color(c), d.sub_p[1], d.sub_s[1], d.ksq[c])];
            for (int i = 0; i < HIDDEN; ++i) o[i] = s[i] + a0[i] + a1[i] - s0[i] - s1[i];
        }
    }
}

namespace {

int bucket_of(int pieces) {
    return net.buckets == 1 ? 0 : std::clamp((pieces - 2) * net.buckets / 32, 0, net.buckets - 1);
}

// sum over neurons of clamp(us, 0, QA) * w[i] + clamp(them, 0, QA) * w[HIDDEN + i]
std::int32_t dot(const std::int16_t* us, const std::int16_t* them, const std::int16_t* w) {
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
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(u, _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i))));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(t, _mm256_load_si256(reinterpret_cast<const __m256i*>(w + HIDDEN + i))));
    }
    __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(acc32), _mm256_extracti128_si256(acc32, 1));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0xB1));
    sum = _mm_cvtsi128_si32(s128);
#else
    for (int i = 0; i < HIDDEN; ++i) {
        sum += std::clamp<std::int32_t>(us[i], 0, QA) * w[i];
        sum += std::clamp<std::int32_t>(them[i], 0, QA) * w[HIDDEN + i];
    }
#endif
    return sum;
}

// sum over neurons of clamp(x, 0, QA)^2 * w; int32 only when load() proved no partial sum can overflow.
std::int64_t dot_screlu(const std::int16_t* us, const std::int16_t* them, const std::int16_t* w) {
    if (!net.screlu_fast) {
        std::int64_t sum = 0;
        for (int i = 0; i < HIDDEN; ++i) {
            std::int64_t u = std::clamp<std::int32_t>(us[i], 0, QA), t = std::clamp<std::int32_t>(them[i], 0, QA);
            sum += u * u * w[i] + t * t * w[HIDDEN + i];
        }
        return sum;
    }
    std::int32_t sum = 0;
#if defined(__AVX2__)
    // x * w fits in int16 (255 * 127), so x * (x * w) is one multiply and one multiply-add per 16 neurons.
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
    __m256i acc32 = _mm256_setzero_si256();
    for (int i = 0; i < HIDDEN; i += 16) {
        __m256i u = _mm256_load_si256(reinterpret_cast<const __m256i*>(us + i));
        __m256i t = _mm256_load_si256(reinterpret_cast<const __m256i*>(them + i));
        u = _mm256_min_epi16(_mm256_max_epi16(u, zero), qa);
        t = _mm256_min_epi16(_mm256_max_epi16(t, zero), qa);
        __m256i uw = _mm256_mullo_epi16(u, _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i)));
        __m256i tw = _mm256_mullo_epi16(t, _mm256_load_si256(reinterpret_cast<const __m256i*>(w + HIDDEN + i)));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(u, uw));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(t, tw));
    }
    __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(acc32), _mm256_extracti128_si256(acc32, 1));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
    s128 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0xB1));
    sum = _mm_cvtsi128_si32(s128);
#else
    for (int i = 0; i < HIDDEN; ++i) {
        std::int32_t u = std::clamp<std::int32_t>(us[i], 0, QA), t = std::clamp<std::int32_t>(them[i], 0, QA);
        sum += u * u * w[i] + t * t * w[HIDDEN + i];
    }
#endif
    return sum;
}

}  // namespace

Value evaluate(const Accumulator& acc, Color stm, int pieces) {
    int bucket = bucket_of(pieces);
    std::int64_t v;
    if (net.screlu) {
        // The squared activation carries an extra factor QA.
        std::int64_t sum = dot_screlu(acc.v[stm], acc.v[~stm], net.out_w[bucket]);
        v = (sum + std::int64_t(net.out_b[bucket]) * QA) * SCALE / (std::int64_t(QA) * QA * QB);
    } else {
        std::int32_t sum = dot(acc.v[stm], acc.v[~stm], net.out_w[bucket]);
        v = (std::int64_t(sum) + net.out_b[bucket]) * SCALE / (QA * QB);
    }
    return Value(std::clamp<std::int64_t>(v, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1));
}

int uncertainty(const Accumulator& acc, Color stm, int pieces) {
    if (!net.has_aux) return -1;
    int bucket = bucket_of(pieces);
    std::int32_t sum = dot(acc.v[stm], acc.v[~stm], net.aux_w[bucket]);
    return std::max(0, int((std::int64_t(sum) + net.aux_b[bucket]) / net.aux_q));
}

bool has_uncertainty() { return is_loaded && net.has_aux; }

}  // namespace tsurugi::nnue
