// Opt-in host-side cache scheduling; no changes to the expert selection rule.
#pragma once
#include <cstddef>
namespace strata::ds4 {
struct CacheScan {
    size_t next = 0;
    size_t begin(size_t count, bool enabled) const { return enabled && count ? next % count : 0; }
    static size_t layer(size_t begin, size_t offset, size_t count) { return (begin + offset) % count; }
    void queued(size_t layer, size_t count, bool enabled) { if (enabled) next = (layer + 1) % count; }
};
}
