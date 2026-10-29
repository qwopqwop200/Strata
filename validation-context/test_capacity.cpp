#include "ds4_context_capacity.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace strata::ds4::context_capacity;
int main() {
    std::vector<int> ratios(43,1);ratios[0]=ratios[1]=0;for(int i=2;i<20;++i)ratios[i]=2;
    std::vector<int> sources{2,8,14,20};Plan p;std::string error;
    assert(!requested(nullptr)&&!requested("")&&!requested("0")&&!requested("true")&&!requested("11")&&requested("1"));
    for(int64_t n:{131072,1000000,1048576}) {
        assert(make(1048576,n,40,512,128,8,true,ratios,sources,p,error));
        assert(p.context==(uint64_t)n&&p.largest_source_rows==(uint64_t)n+1);
        const uint64_t expected=3*((n+1)/2+1)+n+1;
        assert(p.source_rows==expected&&p.source_cache_bytes==expected*(288+80));
        assert(p.decode_score_bytes==((uint64_t)n+64)*16*4);
        assert(p.prefill_score_bytes==((uint64_t)n+64)*p.index_slab_rows*4);
    }
    assert(p.index_slab_rows==32&&p.source_rows==2621444&&p.source_cache_bytes==964691392);
    assert(!make(1048576,1048577,40,512,128,8,true,ratios,sources,p,error));
    assert(!make(0,1000000,40,512,128,8,true,ratios,sources,p,error));
    assert(!make(1048576,-1,40,512,128,8,true,ratios,sources,p,error));
    assert(!make(1048576,1000000,40,512,128,0,true,ratios,sources,p,error));
    assert(!make(1048576,1000000,40,512,128,8,true,ratios,{2,2},p,error));
    assert(!make(1048576,1000000,40,512,128,8,true,ratios,{40},p,error));
    assert(!make(1048576,1000000,40,512,128,8,true,ratios,{0},p,error));
    assert(make(1048576,1000000,40,512,128,8,false,ratios,sources,p,error));assert(p.prefill_score_bytes==0);
    int ids[]={0,1,128803,129279};printf("FNV64 %016llx\n",(unsigned long long)input_fingerprint(ids,4));
    puts("PASS metadata/1M partialallocation/invalidbounds/fallback/fingerprint CPU only; not actual1M model");
}
