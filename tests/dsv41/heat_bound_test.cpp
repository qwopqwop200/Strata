#include <cassert>
#include <cstring>
#include <numeric>
#include "../../src/dsv41/ds4_heat.hpp"
int main() {
    using strata::ds4::bound_heat;
    std::vector<float> h{0, 131072, 65536, 16384}, original = h;
    assert(!bound_heat(h, 0) && std::memcmp(h.data(), original.data(), h.size()*4) == 0);
    assert(!bound_heat(h, INFINITY) && h == original);
    assert(bound_heat(h, 512));
    assert(h[0] == 0 && h[1] > h[2] && h[2] > h[3]);
    assert(std::abs(h[1]/h[2] - 2.f) < 1e-6f);
    assert(std::abs(h[2]/h[3] - 4.f) < 1e-6f);
    assert(std::abs(std::accumulate(h.begin(), h.end(), 0.0) - 512) < 1e-3);
    std::vector<float> small{0, 10, 20}; original = small;
    assert(!bound_heat(small, 512) && small == original);
    std::vector<float> zero(384, 0);
    assert(!bound_heat(zero, 512));
    std::vector<float> short_prompt{512, 618, 0}, saved = short_prompt;
    assert(!bound_heat(short_prompt, 512, 8) && short_prompt == saved);
    std::vector<float> boundary{2048, 2048}; saved = boundary;
    assert(!bound_heat(boundary, 512, 8) && boundary == saved);
    boundary[1] += 1;
    assert(bound_heat(boundary, 512, 8));
    assert(std::abs(std::accumulate(boundary.begin(), boundary.end(), 0.0) - 512) < 1e-3);
    h = {0, 131072, 65536, 16384}; saved = h;
    assert(bound_heat(h, 512, 8) && bound_heat(saved, 512, 1) && h == saved);
}
