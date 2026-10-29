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
#include <array>
#include <sstream>

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
        std::vector<int> watch_tokens;
        int eos=0,think_end=0,root=0;
        bool tools=false,required=false;
    };
    std::shared_ptr<const Program> p;
    States states;
    bool thinking=false;
    bool active=false,in_argument=false,called=false;
    size_t marker=0,foreign=0;
    std::string tag;
    struct Proof {std::array<uint64_t,3> masks{},active{},args{},banned{};};
    // Evidence counts include speculative snapshots. They never determine decoding state.
    std::shared_ptr<Proof> proof_counts=std::make_shared<Proof>();
    static const std::string& opener(){static const std::string s="<｜DSML｜";return s;}
    static const std::string& qwen(){static const std::string s="<tool_call>";return s;}
    bool ended(const States& s) const {
        return std::any_of(s.begin(),s.end(),[](const Stack& x){return x.empty();});
    }
    bool tool_byte(uint8_t b) {
        if(!active) {
            // Qwen/recovery tool syntax must not bypass the DSML constraint.
            foreign=(b==uint8_t(qwen()[foreign]))?foreign+1:(b=='<'?1:0);
            if(foreign==qwen().size())return false;
            marker=(b==uint8_t(opener()[marker]))?marker+1:(b=='<'?1:0);
            if(marker==opener().size()) {
                marker=foreign=0;active=true;in_argument=false;tag.clear();
                states=expand({{p->root}});
            }
            return true;
        }
        auto next=step(states,b);if(next.empty())return false;
        states=std::move(next);
        if(b=='<')tag="<";
        else if(!tag.empty()) {
            tag.push_back(char(b));
            if(b=='>') {
                if(tag.rfind("<｜DSML｜ parameter ",0)==0)in_argument=true;
                if(tag=="</｜DSML｜ parameter>")in_argument=false;
                tag.clear();
            }
            else if(tag.size()>1024)tag.clear();
        }
        if(ended(states)) {active=false;in_argument=false;called=true;states.clear();tag.clear();}
        return true;
    }
    void visit_tool(int node,const Grammar& g,std::vector<uint8_t>& mask) const {
        for(int t:p->trie[size_t(node)].tokens)mask[size_t(t)]=1;
        for(const auto& edge:p->trie[size_t(node)].edge) {
            Grammar next=g;if(next.tool_byte(edge.first))visit_tool(edge.second,next,mask);
        }
    }

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
        const bool tools=std::string(magic,4)=="SGT1";
        if(!tools && std::string(magic,4)!="SGR1")throw std::runtime_error("invalid grammar bundle");
        int V=get(),eos=get(),think=get(),wait=get(),N=get(),root=get();
        if(V!=expected_vocab || V<=0 || N<=0 || N>65536 || eos<0 || eos>=V || think<0 || think>=V || (wait!=0 && wait!=1))
            throw std::runtime_error("grammar vocabulary/header mismatch");
        auto prog=std::make_shared<Program>();prog->eos=eos;prog->think_end=think;prog->root=root;prog->tools=tools;prog->required=tools && wait;
        prog->bytes.resize(size_t(V));prog->special.resize(size_t(V));
        for(int t=0;t<V;++t){int n=get();if(n< -1 || n>1048576)throw std::runtime_error("invalid token byte length");
            prog->special[size_t(t)]=n<0;
            if(n>=0){auto&b=prog->bytes[size_t(t)];b.resize(size_t(n));f.read(b.data(),n);if(!f)throw std::runtime_error("truncated token bytes");
                // Empty tokens cannot make progress and are never legal answer tokens.
                if(!n)continue;
                if(b.find('<')!=std::string::npos)prog->watch_tokens.push_back(t);
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
        auto g=std::make_shared<Grammar>();g->p=prog;g->thinking=!tools && wait!=0;
        if(!tools)g->states=g->expand({{root}});
        else {
            int n=get();if(n<0 || n>1048576)throw std::runtime_error("invalid tool grammar prefix");
            for(int i=0;i<n;++i)if(!g->accept(get()))throw std::runtime_error("tool prefix violates grammar");
        }
        return g;
    }
    bool complete() const {return p->tools?(!active && (!p->required || called)):(!thinking && ended(states));}
    bool waiting() const {return thinking;}
    bool tool_active() const {return p->tools && active;}
    bool argument_active() const {return tool_active() && in_argument;}
    std::string proof() const {
        if(!p->tools)return "";
        std::ostringstream o;o<<"ds4 tool mask proof:";
        const char* names[]={"target","draft","verify"};
        for(size_t i=0;i<3;++i)o<<" "<<names[i]<<"="<<proof_counts->masks[i]<<","<<proof_counts->active[i]<<","<<proof_counts->args[i]<<","<<proof_counts->banned[i];
        o<<" fields=masks,protocol_masks,argument_masks,banned_candidates";return o.str();
    }
    std::vector<uint8_t> allowed(const char* site=nullptr) const {
        if(p->tools) {
            std::vector<uint8_t> mask(p->bytes.size(),0);
            if(!active) {
                std::fill(mask.begin(),mask.end(),1);
                if(!marker && !foreign) {
                    for(int t:p->watch_tokens){Grammar next=*this;mask[size_t(t)]=next.accept(t);}
                } else for(size_t t=0;t<mask.size();++t){Grammar next=*this;mask[t]=next.accept(int(t));}
            } else visit_tool(0,*this,mask);
            mask[size_t(p->eos)]=complete();
            if(site) {
                const size_t i=std::string(site)=="draft"?1:(std::string(site)=="verify"?2:0);
                ++proof_counts->masks[i];proof_counts->active[i]+=active;proof_counts->args[i]+=in_argument;
                proof_counts->banned[i]+=std::count(mask.begin(),mask.end(),uint8_t(0));
            }
            return mask;
        }
        std::vector<uint8_t> mask(p->bytes.size(),thinking?1:0);
        if(thinking){mask[size_t(p->eos)]=0;return mask;}
        visit(0,states,mask);mask[size_t(p->eos)]=complete();return mask;
    }
    bool accept(int token) {
        if(token<0 || size_t(token)>=p->bytes.size())return false;
        if(p->tools) {
            if(token==p->eos)return complete();
            if(p->special[size_t(token)]) {
                if(active)return false;
                marker=foreign=0;return true;
            }
            if(p->bytes[size_t(token)].empty())return !active;
            Grammar next=*this;
            for(uint8_t b:p->bytes[size_t(token)])if(!next.tool_byte(b))return false;
            *this=std::move(next);return true;
        }
        if(thinking){if(token==p->eos)return false;if(token==p->think_end)thinking=false;return true;}
        if(token==p->eos)return complete();
        if(p->special[size_t(token)] || p->bytes[size_t(token)].empty())return false;
        auto next=states;for(uint8_t b:p->bytes[size_t(token)]){next=step(next,b);if(next.empty())return false;}
        states=std::move(next);return true;
    }
};
} // namespace strata::ds4
