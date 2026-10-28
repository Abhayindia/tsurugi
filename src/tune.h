#pragma once

// Search constants. A build with -DTSURUGI_TUNE=ON exposes each one as a UCI spin option (for SPSA tuning);
// other builds compile them as constants.

#include <vector>

namespace tsurugi::tune {

struct Param {
    const char* name;
    int* value;
    int min, max;
};

#ifdef TSURUGI_TUNE
inline std::vector<Param>& params() {
    static std::vector<Param> list;
    return list;
}
struct Register {
    Register(const char* name, int* value, int min, int max) { params().push_back({name, value, min, max}); }
};
#define TUNABLE(name, def, lo, hi) \
    inline int name = def;         \
    inline Register name##_reg{#name, &name, lo, hi};
#else
#define TUNABLE(name, def, lo, hi) constexpr int name = def;
#endif

TUNABLE(rfp_depth, 8, 4, 12)
TUNABLE(rfp_margin, 75, 40, 120)
TUNABLE(razor_depth, 3, 1, 5)
TUNABLE(razor_base, 200, 100, 350)
TUNABLE(razor_mult, 150, 80, 250)
TUNABLE(nmp_base, 3, 2, 5)
TUNABLE(nmp_depth_div, 4, 3, 6)
TUNABLE(nmp_eval_div, 200, 100, 400)
TUNABLE(probcut_depth, 5, 3, 8)
TUNABLE(probcut_margin, 200, 120, 320)
TUNABLE(se_depth, 8, 6, 10)
TUNABLE(se_mult, 2, 1, 4)
TUNABLE(se_double, 20, 8, 40)
TUNABLE(lmp_depth, 6, 3, 9)
TUNABLE(lmp_base, 3, 1, 6)
TUNABLE(fut_depth, 6, 3, 9)
TUNABLE(fut_base, 100, 40, 200)
TUNABLE(fut_mult, 90, 50, 150)
TUNABLE(see_depth, 6, 3, 9)
TUNABLE(see_quiet, 60, 30, 100)
TUNABLE(see_noisy, 25, 10, 50)
TUNABLE(lmr_base, 70, 30, 110)    // x100
TUNABLE(lmr_div, 230, 150, 320)   // x100
TUNABLE(lmr_hist_div, 16384, 8192, 32768)
TUNABLE(asp_delta, 20, 10, 40)

#undef TUNABLE

}  // namespace tsurugi::tune
