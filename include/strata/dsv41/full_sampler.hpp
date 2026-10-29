// Complete-vocabulary reference and native host fallback. All probability paths share this chain.
#pragma once
#include "sampler.hpp"
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strata::dsv41k {
inline double full_uniform(uint64_t seed,uint64_t counter) {
    uint32_t a=uint32_t(counter),b=uint32_t(counter>>32),c=uint32_t(seed),d=uint32_t(seed>>32);
    for(uint32_t i=0;i<10;++i) {
        uint64_t x=uint64_t(0x9E3779B9u)*a,y=uint64_t(0xBB67AE85u)*c;
        a=uint32_t(y>>32)^b^i;b=uint32_t(y);c=uint32_t(x>>32)^d;d=uint32_t(x);
    }
    return double(a>>8)/16777216.0;
}
inline std::vector<double> full_distribution(const std::vector<float>& logits,const SamplerParams&p,
                                             const std::vector<int>&history={}) {
    const size_t V=logits.size();
    if(!V || V>size_t(INT32_MAX) || p.top_k<0 || !std::isfinite(p.temperature) ||
       !std::isfinite(p.top_p) || p.top_p<=0 || p.top_p>1 || !std::isfinite(p.min_p) || p.min_p<0 || p.min_p>1 ||
       !std::isfinite(p.penalty_repeat) || p.penalty_repeat<=0 || !std::isfinite(p.penalty_freq) ||
       !std::isfinite(p.penalty_present) || (p.host_allowed && p.host_allowed->size()!=V))
        throw std::invalid_argument("invalid full-vocabulary sampler parameters");
    std::vector<int> counts(V,0),ids;ids.reserve(V);
    if(p.penalty_last_n>0) {
        if(history.empty())throw std::invalid_argument("penalty window needs history");
        for(size_t i=history.size()-std::min(history.size(),size_t(p.penalty_last_n));i<history.size();++i)
            if(history[i]>=0 && size_t(history[i])<V)++counts[size_t(history[i])];
    }
    std::vector<double> score(V),prob(V,0);
    for(size_t v=0;v<V;++v) {
        double x=logits[v];
        if(counts[v]) {x=x<=0?x*p.penalty_repeat:x/p.penalty_repeat;x-=counts[v]*p.penalty_freq+p.penalty_present;}
        score[v]=x;
        if((!p.host_allowed || (*p.host_allowed)[v]) && !std::isnan(x) && x!=-INFINITY)ids.push_back(int(v));
    }
    if(ids.empty())throw std::runtime_error("no legal finite sampling candidates");
    auto before=[&](int a,int b){return score[a]!=score[b]?score[a]>score[b]:a<b;};
    if(p.greedy || p.temperature<=0) {
        int best=ids[0];for(int v:ids)if(before(v,best))best=v;prob[best]=1;return prob;
    }
    if(p.top_k>0 && size_t(p.top_k)<ids.size()) {
        std::nth_element(ids.begin(),ids.begin()+p.top_k,ids.end(),before);ids.resize(size_t(p.top_k));
    }
    // Sorting is only needed for truncation. The unlimited chain draws in token-id order, without a shortlist.
    if(p.top_p<1 || p.min_p>0)std::sort(ids.begin(),ids.end(),before);
    double mx=-INFINITY;for(int v:ids)mx=std::max(mx,score[v]);
    auto weight=[&](int v,double temp){return mx==INFINITY?double(score[v]==INFINITY):std::exp((score[v]-mx)/temp);};
    if(p.top_p<1) {
        double z=0;for(int v:ids)z+=weight(v,1);
        double cum=0;size_t keep=0;
        for(;keep<ids.size();++keep) {cum+=weight(ids[keep],1)/z;if(cum>=p.top_p){++keep;break;}}
        ids.resize(std::min(ids.size(),std::max(keep,size_t(std::max(1,p.min_keep)))));
    }
    if(p.min_p>0) {
        double threshold=mx+std::log(double(p.min_p));
        ids.erase(std::remove_if(ids.begin(),ids.end(),[&](int v){return score[v]<threshold;}),ids.end());
    }
    double z=0;for(int v:ids){prob[v]=weight(v,p.temperature);z+=prob[v];}
    if(!(z>0) || !std::isfinite(z))throw std::runtime_error("invalid probability normalization");
    for(double&x:prob)x/=z;
    return prob;
}
inline int draw_distribution(const std::vector<double>&p,double u) {
    if(p.empty() || !std::isfinite(u) || u<0 || u>=1)throw std::invalid_argument("invalid categorical draw");
    double z=std::accumulate(p.begin(),p.end(),0.0),cum=0;int last=-1;
    if(!(z>0) || !std::isfinite(z))throw std::invalid_argument("empty categorical distribution");
    for(size_t i=0;i<p.size();++i)if(p[i]>0){last=int(i);cum+=p[i]/z;if(u<cum)return int(i);}
    return last;
}
inline int verify_distribution(const std::vector<double>&p,const std::vector<double>&q,int draft,double ua,double ur) {
    if(p.size()!=q.size() || draft<0 || size_t(draft)>=p.size())throw std::invalid_argument("invalid speculative distribution");
    if(q[draft]<=0)return draw_distribution(p,ur);
    if(ua*q[draft]<p[draft])return draft;
    std::vector<double> residual(p.size());double z=0;
    for(size_t i=0;i<p.size();++i){residual[i]=std::max(0.0,p[i]-q[i]);z+=residual[i];}
    return draw_distribution(z>0?residual:p,ur);
}
} // namespace strata::dsv41k
