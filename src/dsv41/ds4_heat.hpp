#pragma once
#include <cmath>
#include <vector>

namespace strata::ds4 {
// Keep cache priorities in order while limiting a prompt's inertia against decode hits.
// This changes placement hints only, never expert weights or routing selections.
inline bool bound_heat(std::vector<float>& heat, double cap, double trigger = 1.0) {
    if (!(cap > 0.0) || !std::isfinite(cap) || !(trigger >= 1.0) || !std::isfinite(trigger)) return false;
    double sum = 0;
    for (float h : heat) sum += h;
    if (!(sum > cap * trigger)) return false;
    const double scale = cap / sum;
    for (float& h : heat) h = (float) (h * scale);
    return true;
}
}
