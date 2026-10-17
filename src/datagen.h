#pragma once

#include <cstdint>
#include <string>

namespace tsurugi {

// Self-play data generation: plays `games` games from random openings at a fixed node budget per move and
// appends quiet positions to `path` as "fen | white-view score (cp) | white-view result (1.0/0.5/0.0)".
void datagen(int games, std::uint64_t nodes, std::uint64_t seed, const std::string& path);

}  // namespace tsurugi
