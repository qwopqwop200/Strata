#include "strata/dsv41/full_sampler.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
using namespace strata::dsv41k;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(){
    constexpr int V=129280,T=8;
    float *l=nullptr;int*out=nullptr;SpecList*q=nullptr;double*full=nullptr;
    ck(cudaMalloc(&l,size_t(V)*T*4));ck(cudaMemset(l,0,size_t(V)*T*4));
    ck(cudaMalloc(&out,T*4));ck(cudaMalloc(&q,T*sizeof(SpecList)));ck(cudaMalloc(&full,size_t(V)*T*8));
    std::vector<SpecList> h(T);for(int i=0;i<T;++i){h[i].full=full+size_t(i)*V;h[i].capacity=V;}
    ck(cudaMemcpy(q,h.data(),T*sizeof(SpecList),cudaMemcpyHostToDevice));
    SamplerParams p;p.top_k=0;p.top_p=1;p.temperature=1;p.seed=123;
    sample_tokens(l,T,V,nullptr,0,p,out,nullptr);
    std::vector<int> picks(T);ck(cudaMemcpy(picks.data(),out,T*4,cudaMemcpyDeviceToHost));
    assert(std::any_of(picks.begin(),picks.end(),[](int v){return v>64;}));
    sample_draft(l,T,V,p,q,out,nullptr);
    ck(cudaMemcpy(h.data(),q,T*sizeof(SpecList),cudaMemcpyDeviceToHost));assert(h[0].n==-V);
    std::vector<double> prob(V);ck(cudaMemcpy(prob.data(),full,V*8,cudaMemcpyDeviceToHost));
    assert(std::abs(std::accumulate(prob.begin(),prob.end(),0.0)-1)<1e-10);
    int*drafts=nullptr;ck(cudaMalloc(&drafts,T*4));ck(cudaMemcpy(drafts,out,T*4,cudaMemcpyDeviceToDevice));
    sample_verify(l,T,V,p,q,drafts,out,nullptr);std::vector<int>a(T),b(T);
    ck(cudaMemcpy(a.data(),out,T*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),drafts,T*4,cudaMemcpyDeviceToHost));assert(a==b);
    // Missing dense capacity is a recoverable error; compact distributions need no dense allocation.
    h[0].full=nullptr;h[0].capacity=0;ck(cudaMemcpy(q,h.data(),sizeof(SpecList),cudaMemcpyHostToDevice));
    bool rejected=false;try{sample_draft(l,1,V,p,q,out,nullptr);}catch(const std::exception&){rejected=true;}assert(rejected);
    p.top_k=128;sample_tokens(l,1,V,nullptr,0,p,out,nullptr);
    int v=-1;ck(cudaMemcpy(&v,out,4,cudaMemcpyDeviceToHost));assert(v>=0&&v<128);
    p.temperature=0;p.top_k=0;sample_tokens(l,1,V,nullptr,0,p,out,nullptr);ck(cudaMemcpy(&v,out,4,cudaMemcpyDeviceToHost));assert(v==0);
    for(void*x:{(void*)l,(void*)out,(void*)q,(void*)full,(void*)drafts})ck(cudaFree(x));
    std::cout<<"PASS CUDA native full-vocabulary draw/draft/verify, 129280 probabilities, rank>64, capacity failure, top-k128, greedy\n";
}
