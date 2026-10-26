#pragma once

#include <cstdint>
#include <string>

namespace tsurugi {

// Self-play data generation: plays `games` games from random openings at a fixed node budget per move and
// appends quiet positions to `path` as "fen | white-view score (cp) | white-view result (1.0/0.5/0.0)".
// guided: each move's node budget scales with the network's uncertainty about the position (same average budget),
// so positions the network finds unclear get deeper, better-labelled searches.
void datagen(int games, std::uint64_t nodes, std::uint64_t seed, const std::string& path, bool guided = false);

}  // namespace tsurugi
