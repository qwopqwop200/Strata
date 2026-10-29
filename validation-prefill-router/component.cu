#include "ds4_launch.h"
#include <cstring>
#include <mutex>
#include <cstdint>
namespace strata::ds4 {
constexpr int kThreads=256;
static int cdiv(int a,int b){return (a+b-1)/b;}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__global__ void __launch_bounds__(kThreads) gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                            int n, int k, float* __restrict__ y, int ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const float4* w = (const float4*) (W + (size_t) row * k);
    const float4* xv = (const float4*) (x + (size_t) t * k);
    float acc = 0.0f;
    for (int c = lane; c < k / 4; c += 32) {
        const float4 a = w[c], b = xv[c];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}

// PREFILL_ROUTER_REUSE_BEGIN
#if defined(__CUDACC__) && !defined(STRATA_USE_HIP) && !defined(STRATA_METAL)
bool prefill_router_reuse_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("DS4_PREFILL_ROUTER_REUSE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}
// Two independent tokens share each float4 weight load. Each accumulator keeps
// the original four fmaf operations and warp XOR reduction order.
__global__ void __launch_bounds__(kThreads) gemv_f32_prefill_pair_kernel(
    const float* __restrict__ W, const float* __restrict__ x, int n, int k, int T,
    float* __restrict__ y, int ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = 2 * blockIdx.y;
    if (row >= n) return;
    const float4* w = (const float4*) (W + (size_t) row * k);
    const float4* x0 = (const float4*) (x + (size_t) t * k);
    const bool second = t + 1 < T;
    const float4* x1 = second ? (const float4*) (x + (size_t) (t + 1) * k) : x0;
    float a0 = 0.0f, a1 = 0.0f;
    for (int c = lane; c < k / 4; c += 32) {
        const float4 a = w[c], b0 = x0[c];
        a0 = fmaf(a.x, b0.x, a0);
        a0 = fmaf(a.y, b0.y, a0);
        a0 = fmaf(a.z, b0.z, a0);
        a0 = fmaf(a.w, b0.w, a0);
        if (second) {
            const float4 b1 = x1[c];
            a1 = fmaf(a.x, b1.x, a1);
            a1 = fmaf(a.y, b1.y, a1);
            a1 = fmaf(a.z, b1.z, a1);
            a1 = fmaf(a.w, b1.w, a1);
        }
    }
    a0 = warp_sum(a0);
    if (second) a1 = warp_sum(a1);
    if (lane == 0) {
        y[(size_t) t * ldy + row] = a0;
        if (second) y[(size_t) (t + 1) * ldy + row] = a1;
    }
}
#endif
// PREFILL_ROUTER_REUSE_END

void gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy, cudaStream_t s) {
// PREFILL_ROUTER_REUSE_DISPATCH_BEGIN
#if defined(__CUDACC__) && !defined(STRATA_USE_HIP) && !defined(STRATA_METAL)
    // Only the current pack's large prompt router. Window/decode and static
    // router batching keep their original dispatch, buffers and launch order.
    if (prefill_router_reuse_enabled() && T > 16 && T <= 8192 && n == 384 && k == 5120 && ldy == 1024) {
        strata_ds4_launch(gemv_f32_prefill_pair_kernel,
                         dim3(cdiv(n, kThreads / 32), cdiv(T, 2)), kThreads, 0, s,
                         W, x, n, k, T, y, ldy);
        static std::once_flag reported;
        std::call_once(reported, [=] {
            std::fprintf(stderr, "ds4 prefill router reuse: requested=1 dispatch_selected=1 tokens_per_warp=2 first_T=%d n=384 k=5120 ldy=1024 scope=process_first_dispatch kernel_completion_certified=0\n", T);
        });
        return;
    }
#endif
// PREFILL_ROUTER_REUSE_DISPATCH_END
#if defined(STRATA_METAL)
    if (T > 0 && ::strata::dsv41::metal_host::gemv_f32(W, x, n, k, T, y, ldy)) return;
#endif
    if (T > 0) strata_ds4_launch(gemv_f32_kernel, dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s, W, x, n, k, y, ldy);
}


}
#include "main.inc"
