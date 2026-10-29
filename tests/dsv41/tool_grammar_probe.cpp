#include "ds4_grammar.hpp"
#include "strata/dsv41/full_sampler.hpp"
#include <iostream>
#include <sstream>
using strata::ds4::Grammar;
using namespace strata::dsv41k;
int main(int argc,char** argv) {
    try {
        if(argc!=4)return 2;
        int V=std::stoi(argv[2]);auto g=Grammar::load(argv[1],V);
        int checked=0,args=0;bool ok=true;
        std::istringstream in(argv[3]);std::string part;
        while(std::getline(in,part,',')) {
            int t=std::stoi(part);auto mask=g->allowed();auto before=*g;
            if(t<0 || t>=V || !mask[size_t(t)] || !g->accept(t)){ok=false;break;}
            if(before.tool_active()) {
                // The schema-illegal candidate is the UNIQUE highest logit.
                auto bad=std::find(mask.begin(),mask.end(),uint8_t(0));
                if(bad==mask.end())return 3;
                int b=int(bad-mask.begin());std::vector<float> logits(size_t(V),-INFINITY);logits[size_t(t)]=0;logits[size_t(b)]=1000;
                SamplerParams sp;sp.top_k=0;sp.temperature=1;sp.host_allowed=&mask;
                auto p=full_distribution(logits,sp);if(p[size_t(b)]!=0 || p[size_t(t)]!=1)return 4;
                sp.host_allowed=nullptr;auto unmasked=full_distribution(logits,sp);if(unmasked[size_t(b)]!=1)return 5;
                std::vector<double> q(size_t(V),0);q[size_t(b)]=1;
                if(verify_distribution(p,q,b,0.5,0.5)!=t)return 6;
                q=p;if(verify_distribution(p,q,t,0.5,0.5)!=t)return 7;
                // A rejected token cannot mutate a speculative snapshot, even
                // across split UTF-8/header/argument/end boundaries.
                auto rejected=before;if(rejected.accept(b) || rejected.allowed()!=mask)return 8;
                if(before.allowed()!=mask)return 9;
                ++checked;args+=before.argument_active();
            }
        }
        std::cout<<"{\"valid\":"<<(ok?"true":"false")<<",\"complete\":"<<(g->complete()?"true":"false")
                 <<",\"argument_active\":"<<(g->argument_active()?"true":"false")<<",\"mask_checks\":"<<checked<<",\"argument_checks\":"<<args<<"}\n";
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
