#include "strata/dsv41/full_sampler.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
using namespace strata::dsv41k;
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch(const std::exception&) {caught=true;} assert(caught); }
int main() {
    SamplerParams p; p.top_k=0; p.top_p=1; p.temperature=1;
    std::vector<float> l(129280,0);
    auto d=full_distribution(l,p);
    assert(d.size()==l.size());
    assert(std::abs(std::accumulate(d.begin(),d.end(),0.0)-1)<1e-10);
    assert(draw_distribution(d,.99)>64);
    for(double x:d) assert(x>0);
    p.top_k=128; d=full_distribution(l,p);
    assert(std::count_if(d.begin(),d.end(),[](double x){return x>0;})==128);
    assert(draw_distribution(d,.99)>64);
    p.top_k=0; p.top_p=.5; d=full_distribution({0,0,0,0},p);
    assert(d[0]==.5 && d[1]==.5 && d[2]==0 && d[3]==0);
    p.top_p=1; p.temperature=2; d=full_distribution({0,2},p);
    assert(std::abs(d[1]-1/(1+std::exp(-1.0)))<1e-12);
    p.temperature=0; d=full_distribution({1,3,3},p); assert(d[1]==1 && draw_distribution(d,.9)==1);
    p.temperature=1; p.min_p=.5; d=full_distribution({0,-1,-2},p); assert(d[0]==1);
    p.min_p=0; std::vector<uint8_t> mask={0,1,0}; p.host_allowed=&mask;
    d=full_distribution({9,0,8},p); assert(d[1]==1); p.host_allowed=nullptr;
    rejects([&]{full_distribution({},p);});
    rejects([&]{full_distribution({-INFINITY,-INFINITY},p);});
    p.top_k=-1; rejects([&]{full_distribution({0},p);});p.top_k=0;
    p.temperature=NAN; rejects([&]{full_distribution({0},p);});p.temperature=1;
    p.top_p=0;rejects([&]{full_distribution({0},p);});p.top_p=1;
    // Both sampled-q and delta-q acceptance/residual paths, with residual mass beyond rank 64.
    std::vector<double> a(257,1.0/257),q(257,0);q[0]=1;
    assert(verify_distribution(a,q,0,0,.7)==0);
    assert(verify_distribution(a,q,0,.9,.99)>64);
    q=a; assert(verify_distribution(a,q,143,.999,.4)==143);
    a={.2,.3,.5};q={.6,.3,.1};
    std::vector<int> counts(3);constexpr int N=100000;
    for(int i=0;i<N;++i) {
        int draft=draw_distribution(q,full_uniform(13,i));
        int pick=verify_distribution(a,q,draft,full_uniform(17,i),full_uniform(19,i));
        ++counts[pick];
    }
    for(int i=0;i<3;++i) assert(std::abs(double(counts[i])/N-a[i])<.012);
    std::cout<<"PASS full vocabulary 129280, rank>64, top-k128, normalization, filters, masks, invalid input, greedy, sampled/delta q residual statistics\n";
}
