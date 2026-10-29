#include <cassert>
#include "../../src/dsv41/ds4_os.hpp"
int main() {
    using strata::ds4::os::warm_limit;
    constexpr uint64_t GiB = 1ull << 30;
    assert(warm_limit(70*GiB, 40*GiB, 8*GiB) == 32*GiB);
    assert(warm_limit(4*GiB, 40*GiB, 8*GiB) == 4*GiB);
    assert(warm_limit(70*GiB, 8*GiB, 8*GiB) == 0);
    assert(warm_limit(70*GiB, 0, 8*GiB) == 0);
    assert(warm_limit(70*GiB, ~uint64_t{0}, 8*GiB) == 70*GiB);
    assert(warm_limit(~uint64_t{0}, 40*GiB, 8*GiB) == 32*GiB);
}
