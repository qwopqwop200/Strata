#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace strata::ds4::vision_contract {
inline bool required(const char* value) { return value != nullptr && std::strcmp(value, "1") == 0; }
inline bool bias_layout(bool present, bool f32, uint64_t bytes, size_t rank, int64_t width, int experts) {
    return present && f32 && experts > 0 && rank == 1 && width == experts && bytes == uint64_t(experts) * 4;
}
} // namespace strata::ds4::vision_contract
