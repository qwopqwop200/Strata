#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::ds4::context_capacity {
inline bool requested(const char* value) { return value && value[0] == '1' && value[1] == '\0'; }
inline uint64_t input_fingerprint(const int* ids, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t token = (uint32_t) ids[i];
        for (unsigned b = 0; b < 4; ++b) {
            hash ^= (uint8_t) (token >> (8 * b));
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}
struct Plan {
    uint64_t max_positions = 0, context = 0, source_rows = 0, largest_source_rows = 0;
    uint64_t source_cache_bytes = 0, decode_score_bytes = 0, prefill_score_bytes = 0;
    uint64_t decode_bitmap_bytes = 0, prefill_bitmap_bytes = 0;
    int index_slab_rows = 0;
};
inline bool multiply(uint64_t a, uint64_t b, uint64_t& value) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    value = a * b;
    return true;
}
// Mirrors current single-device source-cache and score/bitmap allocation formulas.
// This is a partial accounting, never a whole-model/RSS/UMA admission guarantee.
inline bool make(int64_t maximum, int64_t context, int nl, int hd, int index_hd,
                 int candidate_block, bool chunked, const std::vector<int>& ratios,
                 const std::vector<int>& sources, Plan& out, std::string& error) {
    out = {};
    if (maximum <= 0 || context <= 0 || context > maximum ||
        context > (int64_t) std::numeric_limits<int>::max() - 64) {
        error = "context must be positive, within actual max_position_embeddings and int-index capacity";
        return false;
    }
    if (nl <= 0 || (size_t) nl > ratios.size() || hd <= 0 || index_hd <= 0 ||
        hd % 32 || index_hd % 32 || candidate_block <= 0 || sources.empty()) {
        error = "incomplete context/cache geometry";
        return false;
    }
    const uint64_t span = (uint64_t) context + 64;
    const uint64_t kv_rb = ((uint64_t) hd / 2 + (uint64_t) hd / 16 + 15) / 16 * 16;
    const uint64_t ik_rb = ((uint64_t) index_hd / 2 + (uint64_t) index_hd / 32 + 15) / 16 * 16;
    uint64_t rows = 0, largest = 0;
    std::vector<bool> seen((size_t) nl, false);
    for (int source : sources) {
        if (source < 0 || source >= nl || seen[(size_t) source] || ratios[(size_t) source] <= 0) {
            error = "invalid or duplicate compressed KV source/ratio";
            return false;
        }
        seen[(size_t) source] = true;
        const uint64_t ratio = (uint64_t) ratios[(size_t) source];
        const uint64_t n = ((uint64_t) context + ratio - 1) / ratio + 1;
        if (n > (uint64_t) std::numeric_limits<int>::max() || rows > std::numeric_limits<uint64_t>::max() - n) {
            error = "compressed source row overflow";
            return false;
        }
        rows += n;
        if (n > largest) largest = n;
    }
    int slab = (int) (((uint64_t) 64 << 20) / (span * 4)) / 32 * 32;
    if (slab < 32) slab = 32;
    if (slab > 128) slab = 128;
    const uint64_t bitmap_stride = span / (uint64_t) candidate_block + 64;
    out.max_positions = (uint64_t) maximum;
    out.context = (uint64_t) context;
    out.source_rows = rows;
    out.largest_source_rows = largest;
    out.index_slab_rows = chunked ? slab : 0;
    if (!multiply(rows, kv_rb + ik_rb, out.source_cache_bytes) ||
        !multiply(span, 16 * sizeof(float), out.decode_score_bytes) ||
        !multiply(span, chunked ? (uint64_t) slab * sizeof(float) : 0, out.prefill_score_bytes) ||
        !multiply(bitmap_stride, 16 * sizeof(float), out.decode_bitmap_bytes) ||
        !multiply(bitmap_stride, chunked ? (uint64_t) slab * sizeof(float) : 0, out.prefill_bitmap_bytes)) {
        error = "context allocation byte overflow";
        return false;
    }
    return true;
}
} // namespace strata::ds4::context_capacity
