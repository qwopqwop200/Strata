#include "ds4_cache_scan.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <vector>
using strata::ds4::CacheScan;
static std::vector<size_t> tick(CacheScan& scan, size_t count, size_t capacity, bool enabled, const std::vector<bool>& eligible) {
    std::vector<size_t> jobs;
    const size_t start = scan.begin(count, enabled);
    for (size_t i = 0; i < count; ++i) {
        const size_t il = CacheScan::layer(start, i, count);
        if (!eligible[il] || jobs.size() == capacity) continue;
        jobs.push_back(il);
        scan.queued(il, count, enabled);
    }
    return jobs;
}
int main() {
    // Fixed layer order starves the last 32 layers under repeated all-layer demand.
    CacheScan off, on;
    std::array<int, 40> counts{};
    const std::vector<bool> all(40, true);
    for (int t = 0; t < 5; ++t) {
        const auto old = tick(off, 40, 8, false, all);
        for (size_t i = 0; i < 8; ++i) assert(old[i] == i);
        for (size_t il : tick(on, 40, 8, true, all)) ++counts[il];
    }
    for (int c : counts) assert(c == 1);
    assert(on.next == 0 && off.next == 0);
    // A full in-flight queue must not move the cursor past waiting candidates.
    on.next = 19;
    assert(tick(on, 40, 0, true, all).empty() && on.next == 19);
    std::vector<bool> sparse(40, false);
    sparse[2] = sparse[25] = sparse[38] = true;
    assert((tick(on, 40, 2, true, sparse) == std::vector<size_t>{25, 38}));
    assert((tick(on, 40, 1, true, sparse) == std::vector<size_t>{2}));
    // OFF ignores any retained cursor. Empty vectors do not evaluate modulo zero.
    on.next = 999;
    assert(on.begin(0, true) == 0 && tick(on, 0, 8, true, {}).empty());
    assert(on.begin(40, true) == 39 && on.begin(40, false) == 0);
    assert((tick(on, 40, 2, false, sparse) == std::vector<size_t>{2, 25}));
    assert(on.next == 999);
    std::puts("PASS fixed-order parity, cap-blocked cursor retention, sparse/wrapped scheduling and 40-layer fairness");
}
