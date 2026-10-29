// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in llama.cpp's order.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  llama.cpp builds its chain by walking `params.samplers`, whose
// default is { PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE } (`common/common.h`
// at 3cf03257) - ONE penalties stage, first, and TEMPERATURE AFTER THE TRUNCATION FILTERS.  (Issue #53: this file
// used to apply the penalties a second time after the temperature, and min_p before top_p - both taken from the
// order of the `case` labels in `common/sampling.cpp`, which is not the order the chain runs.)  Every order
// produces a valid token, so only a comparison at the distribution level can tell them apart; the parity test
// does that against an independently computed distribution.
//
// Both kernels put ONE BLOCK per token over the vocabulary: `sampler_greedy_kernel` is the plain argmax,
// `sampler_kernel` runs the sampled chain as `top_k` block-argmax rounds followed by the top_p / temperature /
// draw chain (its header says why the selection must be parallel and why the tie rule keeps the semantics).
#include "../ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "strata/dsv41/full_sampler.hpp"
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif

#include <cuda_runtime.h>
#if defined(STRATA_METAL)
#include <strata/gpu/runtime.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <atomic>

namespace strata::dsv41k {
namespace {

// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
__device__ __forceinline__ uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                     uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__device__ __forceinline__ int history_count(const int* __restrict__ h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

__device__ __forceinline__ float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

/// **THE GREEDY ARGMAX, ONE BLOCK PER TOKEN, COVERING THE VOCABULARY.**
///
/// **WHY THIS IS A SEPARATE KERNEL AND NOT A BRANCH.**  `sampler_kernel` is launched as a grid over TOKENS
/// with 64 threads and a `if (t >= n_tokens) return;` at the top.  The decode path has `n_tokens == 1`, so
/// that launch was `<<<1, 64>>>`, 63 threads exited on the first line, and ONE THREAD walked all 248,320
/// logits in a dependent loop on one SM of 48.  Measured in isolation (`bench/micro/sampler_cost.cu`):
/// **3.11 ms per token**, 5.7% of an ~54 ms token, and the whole of round 309's `sample` phase - the two
/// synchronisations around it are 0.03 ms each.
///
/// The obvious repair is to parallelise the scan inside `sampler_kernel`, and it is WRONG: with one thread
/// per token, a block reduction over the vocabulary has nothing to reduce, and the threads that returned
/// early are not there for `__syncthreads` or `__shfl_down_sync`.  The first attempt did exactly that and
/// produced the token `5120` thirty-two times.  The grid has to be over tokens with the BLOCK over the
/// vocabulary, which is a different launch configuration and therefore a different kernel.
///
/// **THE TIE RULE IS UNCHANGED AND THAT IS THE WHOLE CORRECTNESS ARGUMENT.**  The serial scan walked `v`
/// ascending with `if (s > bv)`, so the LOWEST index wins a tie.  Each thread keeps that rule over its own
/// strided subset and the reduction resolves two candidates by taking the larger value and, on equality, the
/// SMALLER index - the same total order, so `sampler_parity` and C1 see no change.
__global__ void sampler_greedy_kernel(const float* __restrict__ logits, int n_vocab,
                                      const int* __restrict__ history, int history_len, const SamplerParams p,
                                      int pmin, int plen, int* __restrict__ out) {
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    (void) pmin;
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // PENALTY MEMBERSHIP AS A BITMAP.  The history touches at most `hlen` tokens of a quarter-million
    // vocabulary, but the naive `history_count` per candidate per argmax round costs O(k x n_vocab x hlen)
    // integer compares (~318 M per token at k=20, hlen=64 - measured 45 -> 31 tok/s on a real workload).
    // A shared bitmap gives an O(1) membership test, and only the (at most hlen) hits pay the count scan;
    // the counts - and therefore every sampled value - are exactly what the per-candidate scan produced.
    extern __shared__ unsigned int penal_bits[];
    const int bits_words = (int) ((n_vocab + 31) / 32);
    // The gate needs a NON-EMPTY WINDOW (`hlen > 0`): the launch sizes the shared bitmap only when penalties
    // are on, so a caller handing over a history buffer with `penalty_last_n == 0` must not touch it.
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = threadIdx.x; w < bits_words; w += blockDim.x) penal_bits[w] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < hlen; i += blockDim.x)
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                atomicOr(&penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        __syncthreads();
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = __int_as_float(0xff800000);   // -inf
    int best = n_vocab;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        const float s = apply_penalties(p.logit_bias ? l[v] + p.logit_bias[v] : l[v], hit_count(v), p);
        if (s > bv) { bv = s; best = v; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xFFFFFFFFu, bv, off);
        const int oi = __shfl_down_sync(0xFFFFFFFFu, best, off);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (int) ((blockDim.x + 31) >> 5);
        float wv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
        int wi = lane < nw ? si[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, wv, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, wi, off);
            if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
        }
        // A tie between two `-inf` candidates leaves `wi == n_vocab`, and the serial version answered 0.
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}


// The positive-temperature chain is native C++ over complete rows. CUDA/HIP/Metal stage only
// actual-vocabulary storage. This portable fallback deliberately trades a host transfer for exact
// unrestricted probabilities; the existing unconstrained greedy GPU kernel stays intact.
constexpr uint64_t kDraftKey=0x5bd1e9955bd1e995ull,kAcceptKey=0x9e3779b97f4a7c15ull,kResidKey=0xc2b2ae3d27d4eb4full;
void checked(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> std::vector<T> read_device(const T*p,size_t n,void*stream) {
    std::vector<T> x(n);if(n){checked(cudaMemcpyAsync(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,(cudaStream_t)stream));checked(cudaStreamSynchronize((cudaStream_t)stream));}return x;
}
template<class T> void write_device(T*p,const T*x,size_t n,void*stream) {
    if(n){checked(cudaMemcpyAsync(p,x,n*sizeof(T),cudaMemcpyHostToDevice,(cudaStream_t)stream));checked(cudaStreamSynchronize((cudaStream_t)stream));}
}
std::vector<double> distribution(const float*logits,int V,const int*history,int H,const SamplerParams&p,void*s) {
    if(V<=0)throw std::invalid_argument("empty sampler vocabulary");
    auto l=read_device(logits,size_t(V),s);
    if(p.logit_bias){auto b=read_device(p.logit_bias,size_t(V),s);for(int v=0;v<V;++v)l[v]+=b[v];}
    auto h=history&&H>0?read_device(history,size_t(H),s):std::vector<int>{};
    return full_distribution(l,p,h);
}
void audit_distribution(const std::vector<double>&d,const SamplerParams&p,bool verify) {
    static std::atomic<unsigned> seen{0};
    const unsigned bit=1u<<((p.host_allowed?1:0)+(verify?2:0));
    if(!(seen.fetch_or(bit)&bit))
        std::fprintf(stderr,"ds4 native distribution: stage=%s actual_vocab=%zu positive=%zu sum=%.17g mask_applied=%d top_k=%d temperature=%g\n",
                     verify?"verify":"draw",d.size(),size_t(std::count_if(d.begin(),d.end(),[](double x){return x>0;})),
                     std::accumulate(d.begin(),d.end(),0.0),p.host_allowed!=nullptr,p.top_k,p.temperature);
}
} // namespace

void sample_tokens(const float*logits,int T,int V,const int*history,int H,const SamplerParams&p,int*out,void*stream) {
    if(T<=0)return;
    if(V<=0)throw std::invalid_argument("empty sampler vocabulary");
    if(p.penalty_last_n>0 && (!history || H<=0))throw std::invalid_argument("penalties need history");
    if((p.greedy || p.temperature<=0) && !p.host_allowed) {
        unsigned shmem=(history&&H>0&&p.penalty_last_n>0)?unsigned((size_t(V)+31)/32*sizeof(unsigned)):0;
        strata_ds4_launch(sampler_greedy_kernel,(unsigned)T,1024,shmem,(cudaStream_t)stream,
                          logits,V,history,H,p,p.penalty_last_n,p.penalty_last_n,out);
        checked(cudaGetLastError());if(!stream)checked(cudaDeviceSynchronize());return;
    }
    std::vector<int> picks(size_t(T),0);
    for(int t=0;t<T;++t){auto d=distribution(logits+size_t(t)*V,V,history?history+size_t(t)*H:nullptr,H,p,stream);
        audit_distribution(d,p,false);picks[t]=draw_distribution(d,full_uniform(p.seed,p.counter+uint64_t(t)));}
    write_device(out,picks.data(),picks.size(),stream);
}
void sample_draft(const float*logits,int T,int V,const SamplerParams&p,SpecList*q,int*out,void*stream) {
    if(T<=0)return;
    auto meta=read_device(q,size_t(T),stream);std::vector<int> picks(size_t(T),0);
    for(int t=0;t<T;++t){auto d=distribution(logits+size_t(t)*V,V,nullptr,0,p,stream);
        picks[t]=draw_distribution(d,full_uniform(p.seed^kDraftKey,p.counter+uint64_t(t)));
        int count=int(std::count_if(d.begin(),d.end(),[](double x){return x>0;}));
        if(count<=64){meta[t].n=count;int i=0;for(int v=0;v<V;++v)if(d[v]>0){meta[t].ids[i]=v;meta[t].p[i++]=d[v];}}
        else {if(!meta[t].full || meta[t].capacity<V)throw std::runtime_error("draft full-vocabulary buffer capacity mismatch");
            write_device(meta[t].full,d.data(),d.size(),stream);meta[t].n=-V;}
    }
    write_device(q,meta.data(),meta.size(),stream);write_device(out,picks.data(),picks.size(),stream);
}
void sample_verify(const float*logits,int T,int V,const SamplerParams&p,const SpecList*q,const int*draft,int*out,void*stream) {
    if(T<=0)return;
    auto meta=read_device(q,size_t(T),stream);auto drafts=read_device(draft,size_t(T),stream);std::vector<int> picks(size_t(T),0);
    for(int t=0;t<T;++t){auto d=distribution(logits+size_t(t)*V,V,nullptr,0,p,stream);audit_distribution(d,p,true);std::vector<double> dq(size_t(V),0);
        if(meta[t].n<0){if(meta[t].n!=-V || !meta[t].full || meta[t].capacity<V)throw std::runtime_error("invalid dense draft distribution");dq=read_device(meta[t].full,size_t(V),stream);}
        else {if(meta[t].n>64)throw std::runtime_error("invalid compact draft distribution");for(int i=0;i<meta[t].n;++i){int v=meta[t].ids[i];if(v<0 || v>=V)throw std::runtime_error("invalid draft token id");dq[v]=meta[t].p[i];}}
        auto c=p.counter+uint64_t(t);
        picks[t]=verify_distribution(d,dq,drafts[t],full_uniform(p.seed^kAcceptKey,c),full_uniform(p.seed^kResidKey,c));
    }
    write_device(out,picks.data(),picks.size(),stream);
}
} // namespace strata::dsv41k
