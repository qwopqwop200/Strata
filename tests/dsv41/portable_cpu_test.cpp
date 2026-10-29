// Model-free smoke test of the CPU symbols used by the DeepSeek engine.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "ds4_gguf_cpu.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#endif

namespace kc = strata::kernels::cpu;
namespace gg = strata::ds4::gguf;

#if defined(STRATA_NATIVE_EXPERTS)
static void iq_rows() {
    constexpr int n = 256, nt = 2, rows = 4;
    for (int type : {16, 18, 20}) {
        gg::IqCpu f;
        std::string error;
        assert(gg::gg_cpu_fmt(type, type, n, n, f, error));
        std::vector<uint8_t> w(f.d_row * rows);
        uint32_t state = 12345;
        for (auto& b : w) { state = state * 1664525u + 1013904223u; b = uint8_t(state >> 24); }
        const size_t block = ggml_type_size(ggml_type(type));
        const auto scale = ggml_fp32_to_fp16(0.003f);
        for (size_t i = 0; i < w.size(); i += block) std::memcpy(w.data() + i, &scale, 2);
        std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(f.h_bytes));
        const void* ap[nt];
        float expected[nt][rows], actual[nt][rows];
        float* out[nt] = {actual[0], actual[1]};
        const auto* trait = ggml_get_type_traits_cpu(ggml_type(type));
        for (int t = 0; t < nt; ++t) {
            float x[n];
            for (int i = 0; i < n; ++i) x[i] = float((i * 7 + t * 13) % 127 - 63) / 64;
            gg::gg_cpu_quant(f.d_act, x, n, act[t].data());
            ap[t] = act[t].data();
            for (int r = 0; r < rows; ++r)
                trait->vec_dot(n, &expected[t][r], 0, w.data() + r * f.d_row, 0, ap[t], 0, 1);
        }
        for (int mode : {0, 1, 2}) {
            gg::gg_cpu_set_iq256(mode);
            gg::gg_cpu_down(f, w.data(), ap, nt, out, 0, rows);
            assert(std::memcmp(actual, expected, sizeof actual) == 0);
        }
        std::printf("PASS CPU IQ type %d: ggml dot / forced / auto bit-identical\n", type);
    }
}

static void tq_rows() {
    constexpr int n = 256, rows = 4, type = 145;
    gg::IqCpu f;
    std::string error;
    assert(gg::tq_cpu_fmt(type, type, n, n, f, error));
    std::vector<uint8_t> weights(rows * f.d_row);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = uint8_t(i * 71 + 11);
    const auto dw = ggml_fp32_to_fp16(1.f / 256);
    for (size_t i = 0; i < weights.size(); i += kc::tq_block_bytes(type))
        std::memcpy(weights.data() + i, &dw, 2);
    // Same exactly representable scales and int8 values in the two activation layouts.
    std::vector<uint8_t> dsact(f.h_bytes), q8act(n / 32 * 34);
    kc::tq_detail::TqAct unpacked{};
    const auto da = ggml_fp32_to_fp16(1.f / 128);
    for (int b = 0; b < n / 32; ++b) {
        unpacked.d[b] = 1.f / 128;
        std::memcpy(dsact.data() + b * 4, &unpacked.d[b], 4);
        std::memcpy(q8act.data() + b * 34, &da, 2);
        for (int i = 0; i < 32; ++i) {
            const int8_t q = int8_t((b * 32 + i) % 127 - 63);
            dsact[n / 32 * 4 + b * 32 + i] = uint8_t(q);
            q8act[b * 34 + 2 + i] = uint8_t(q);
            unpacked.q[b * 32 + i] = q;
        }
    }
    float expected[rows], actual[rows], public_rows[rows];
    float* rp[1] = {expected};
    const kc::tq_detail::TqAct* up[1] = {&unpacked};
    kc::tq_detail::tq_rows_scalar(kc::tq_kbits(type), kc::tq_state_table_for(type),
                                weights.data(), f.d_row, n, up, 1, rp, 0, rows);
    const void* ap[1] = {dsact.data()};
    float* op[1] = {actual};
    gg::tq_cpu_down(f, weights.data(), ap, 1, op, 0, rows);
    assert(std::memcmp(expected, actual, sizeof actual) == 0);
    const void* qp[1] = {q8act.data()};
    float* pp[1] = {public_rows};
    kc::tq_rows(kc::tq_isa(), type, weights.data(), f.d_row, n, qp, 1, pp, 0, rows);
    assert(std::memcmp(expected, public_rows, sizeof public_rows) == 0);
    for (float y : actual) assert(std::isfinite(y));
#if !defined(__x86_64__) && !defined(_M_X64)
    assert(kc::tq_isa() == kc::TqIsa::scalar);
#endif
    std::printf("PASS CPU TQ: DeepSeek / public %s dispatch / scalar bit-identical\n", kc::tq_isa_name(kc::tq_isa()));
}
#endif

int main() {
#if !defined(__x86_64__) && !defined(_M_X64)
    assert(!kc::cpu_avx1_ok() && !kc::cpu_avx2_ok() && !kc::cpu_avx512_ok());
#endif
    // These two functions also have useful portable implementations without native experts.
    if (kc::cpu_avx2_ok()
#if !defined(__x86_64__) && !defined(_M_X64)
        || true
#endif
    ) {
        uint16_t w[16];
        float x[16];
        for (int i = 0; i < 16; ++i) { w[i] = 0x3f80; x[i] = float(i); }
        float out;
        kc::bf16_rows_dot_multi(w, 1, 16, x, 1, &out);
        assert(out == 120.f);
    }
    float x[32];
    for (int i = 0; i < 32; ++i) x[i] = float(i - 16) / 16;
    kc::ActQ act{};
    kc::act_quant_q8_1(x, 32, act);
    assert(act.nchunks == 1 && act.q[0] == -127 && act.scale[0] > 0);
#if defined(STRATA_NATIVE_EXPERTS)
    iq_rows();
    tq_rows();
    std::puts("PASS portable CPU smoke native=on");
#else
    gg::IqCpu f;
    std::string error;
    assert(!gg::gg_cpu_fmt(16, 16, 256, 256, f, error) && !error.empty());
    error.clear();
    assert(!gg::tq_cpu_fmt(145, 145, 256, 256, f, error) && !error.empty());
    std::puts("PASS portable CPU smoke native=off");
#endif
}
