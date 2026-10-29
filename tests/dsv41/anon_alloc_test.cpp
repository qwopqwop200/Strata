// Linux host arena advice: opt-out must preserve writable memory and default advice.
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include "../../src/dsv41/ds4_os.hpp"

#ifndef _WIN32
static bool check(const char* value, const char* flag) {
    if (value) setenv("DS4_HOST_THP", value, 1);
    else unsetenv("DS4_HOST_THP");
    constexpr size_t bytes = 4u << 20;
    void* p = strata::ds4::os::anon_alloc(bytes, -1, 1);
    if (!p) return false;
    std::ifstream f("/proc/self/smaps");
    std::string line;
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    bool found = false, selected = false;
    while (std::getline(f, line)) {
        unsigned long lo, hi;
        if (std::sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2)
            selected = addr >= lo && addr < hi;
        if (selected && line.rfind("VmFlags:", 0) == 0) {
            std::istringstream flags(line.substr(8));
            std::string v;
            while (flags >> v) found = found || v == flag;
            break;
        }
    }
    auto* data = static_cast<unsigned char*>(p);
    for (size_t i = 0; i < bytes; i += 4096) data[i] = static_cast<unsigned char>(i / 4096);
    for (size_t i = 0; i < bytes; i += 4096)
        if (data[i] != static_cast<unsigned char>(i / 4096)) found = false;
    strata::ds4::os::anon_free(p, bytes);
    std::cout << (value ? value : "unset") << ": " << flag << " " << (found ? "PASS" : "FAIL") << '\n';
    return found;
}
#endif
int main() {
#ifndef _WIN32
    const char* prior = std::getenv("DS4_HOST_THP");
    const std::string saved = prior ? prior : "";
    const bool had_prior = prior != nullptr;
    const bool ok = check(nullptr, "hg") && check("1", "hg") && check("0", "nh") && check("1", "hg");
    if (had_prior) setenv("DS4_HOST_THP", saved.c_str(), 1);
    else unsetenv("DS4_HOST_THP");
    return ok ? 0 : 1;
#else
    return 0;
#endif
}
