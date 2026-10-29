// Native byte CFG with per-request copyable state. Token masks are evaluated BEFORE normalization.
#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ds4 {
class Grammar {
    using Stack=std::vector<int>;
    using States=std::vector<Stack>;
    struct Node {std::map<uint8_t,int> edge;std::vector<int> tokens;};
    struct Program {
        std::vector<std::vector<Stack>> rules;
        std::vector<std::string> bytes;
        std::vector<bool> special;
        std::vector<Node> trie=std::vector<Node>(1);
        int eos=0,think_end=0;
    };
    std::shared_ptr<const Program> p;
    States states;
    bool thinking=false;

    States expand(States pending) const {
        std::set<Stack> seen,ready;
        while(!pending.empty()) {
            Stack s=std::move(pending.back());pending.pop_back();
            if(!seen.insert(s).second)continue;
            if(seen.size()>65536 || s.size()>4096)throw std::runtime_error("grammar state capacity exceeded");
            if(s.empty() || s.back()<256){ready.insert(std::move(s));continue;}
            int rule=s.back()-256;s.pop_back();
            if(rule<0 || size_t(rule)>=p->rules.size())throw std::runtime_error("invalid grammar rule reference");
            for(const auto&seq:p->rules[size_t(rule)]) {
                Stack q=s;q.insert(q.end(),seq.rbegin(),seq.rend());pending.push_back(std::move(q));
            }
        }
        return States(ready.begin(),ready.end());
    }
    States step(const States&from,uint8_t b) const {
        States pending;
        for(const auto&s:from)if(!s.empty()) {
            int x=s.back();bool matches=x>=0?x==b:(b>=(-1-x)/256 && b<=(-1-x)%256);
            if(matches){pending.push_back(s);pending.back().pop_back();}
        }
        return expand(std::move(pending));
    }
    void visit(int node,const States&s,std::vector<uint8_t>&mask) const {
        for(int t:p->trie[size_t(node)].tokens)mask[size_t(t)]=1;
        for(const auto&edge:p->trie[size_t(node)].edge) {
            auto next=step(s,edge.first);if(!next.empty())visit(edge.second,next,mask);
        }
    }
public:
    static std::shared_ptr<Grammar> load(const std::string&path,int expected_vocab) {
        std::ifstream f(path,std::ios::binary);char magic[4]={};f.read(magic,4);
        auto get=[&](){int32_t x=0;f.read(reinterpret_cast<char*>(&x),4);if(!f)throw std::runtime_error("truncated grammar bundle");return int(x);};
        if(std::string(magic,4)!="SGR1")throw std::runtime_error("invalid grammar bundle");
        int V=get(),eos=get(),think=get(),wait=get(),N=get(),root=get();
        if(V!=expected_vocab || V<=0 || N<=0 || N>65536 || eos<0 || eos>=V || think<0 || think>=V || (wait!=0 && wait!=1))
            throw std::runtime_error("grammar vocabulary/header mismatch");
        auto prog=std::make_shared<Program>();prog->eos=eos;prog->think_end=think;
        prog->bytes.resize(size_t(V));prog->special.resize(size_t(V));
        for(int t=0;t<V;++t){int n=get();if(n< -1 || n>1048576)throw std::runtime_error("invalid token byte length");
            prog->special[size_t(t)]=n<0;
            if(n>=0){auto&b=prog->bytes[size_t(t)];b.resize(size_t(n));f.read(b.data(),n);if(!f)throw std::runtime_error("truncated token bytes");
                // Empty tokens cannot make progress and are never legal answer tokens.
                if(!n)continue;
                int node=0;for(uint8_t c:b){auto it=prog->trie[size_t(node)].edge.find(c);
                    if(it==prog->trie[size_t(node)].edge.end()){int next=int(prog->trie.size());prog->trie[size_t(node)].edge[c]=next;prog->trie.emplace_back();node=next;}else node=it->second;}
                prog->trie[size_t(node)].tokens.push_back(t);
            }
        }
        prog->rules.resize(size_t(N));size_t symbols=0;
        for(auto&r:prog->rules){int na=get();if(na<1 || na>65536)throw std::runtime_error("invalid grammar alternatives");
            for(int a=0;a<na;++a){int ns=get();if(ns<0 || ns>65536 || (symbols+=size_t(ns))>1048576)throw std::runtime_error("invalid grammar size");
                Stack seq;for(int i=0;i<ns;++i){int x=get();if(x < -65536 || x>=256+N)throw std::runtime_error("invalid grammar symbol");seq.push_back(x);}r.push_back(std::move(seq));}}
        if(root<256 || root>=256+N)throw std::runtime_error("invalid grammar root");
        auto g=std::make_shared<Grammar>();g->p=prog;g->thinking=wait!=0;g->states=g->expand({{root}});return g;
    }
    bool complete() const {return !thinking && std::any_of(states.begin(),states.end(),[](const Stack&s){return s.empty();});}
    bool waiting() const {return thinking;}
    std::vector<uint8_t> allowed() const {
        std::vector<uint8_t> mask(p->bytes.size(),thinking?1:0);
        if(thinking){mask[size_t(p->eos)]=0;return mask;}
        visit(0,states,mask);mask[size_t(p->eos)]=complete();return mask;
    }
    bool accept(int token) {
        if(token<0 || size_t(token)>=p->bytes.size())return false;
        if(thinking){if(token==p->eos)return false;if(token==p->think_end)thinking=false;return true;}
        if(token==p->eos)return complete();
        if(p->special[size_t(token)] || p->bytes[size_t(token)].empty())return false;
        auto next=states;for(uint8_t b:p->bytes[size_t(token)]){next=step(next,b);if(next.empty())return false;}
        states=std::move(next);return true;
    }
};
} // namespace strata::ds4
