#include "ds4_grammar.hpp"
#include <iostream>
int main(int argc,char**argv) {
    try {
        auto g=strata::ds4::Grammar::load(argv[1],258);
        if(argc<3) return 2;
        std::string h=argv[2];
        if(g->waiting()) {
            auto before=*g;
            if(g->allowed()[256] || !g->allowed()[257] || !g->accept(257) || g->waiting() || !before.waiting())return 6;
        }
        auto original=*g;
        bool ok=true;
        for(size_t i=0;i<h.size();i+=2) {
            int t=std::stoi(h.substr(i,2),nullptr,16);
            auto mask=g->allowed();
            // A rejected speculation must leave the snapshot unchanged, including partial UTF-8.
            auto snapshot=*g;
            for(int bad=0;bad<258;++bad)if(!mask[bad]) {
                if(snapshot.accept(bad) || snapshot.allowed()!=mask)return 5;
                break;
            }
            if(!mask[t] || !g->accept(t)){ok=false;break;}
        }
        auto mask=g->allowed();
        if(mask[256]!=g->complete()) return 3;
        if(original.complete())return 4;
        std::cout<<(ok?"valid":"invalid")<<" "<<(g->complete()?"complete":"partial")<<"\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
