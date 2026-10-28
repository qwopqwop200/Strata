#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include "case.inc"
namespace ds = strata::ds4;
static void req(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> static T* alloc(size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
static std::vector<float> get(float* p,size_t n){std::vector<float> r(n);ck(cudaMemcpy(r.data(),p,n*4,cudaMemcpyDeviceToHost));return r;}
static void same(const std::vector<float>& a,const std::vector<float>& b,int T,int kind){size_t diffs=0;for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],4)){uint32_t u,v;std::memcpy(&u,&a[i],4);std::memcpy(&v,&b[i],4);if(diffs++<4)printf("DIFF T=%d kind=%d index=%zu token=%zu row=%zu guard=%d old=%08x new=%08x\n",T,kind,i,i/1024,i%1024,(i>=size_t(T)*1024||i%1024>=384),u,v);}printf("COMPARE T=%d kind=%d words=%zu mismatch=%zu\n",T,kind,a.size(),diffs);req(!diffs,"whole output bits/guard");}
static void old(float* w,float* x,int T,float* y,cudaStream_t s){strata_ds4_launch(ds::gemv_f32_kernel,dim3(48,T),256,0,s,w,x,384,5120,y,1024);}
static void run(int mode,float* w,float* x,int T,float* y,cudaStream_t s){if(mode)ds::gemv_f32(w,x,384,5120,T,y,1024,s);else old(w,x,T,y,s);}
int main(int argc,char**argv){try{
 if(argc>1&&std::string(argv[1])=="--off-guard"){req(!ds::prefill_router_reuse_enabled(),"exact OFF env");puts("PASS OFF noCUDA");return 0;}
 req(ds::prefill_router_reuse_enabled(),"ON required");
 const size_t K=5120,N=384,MAXT=8192,Y=MAXT*1024+64,X=MAXT*K+64;
 std::vector<uint16_t> bf(N*K);std::ifstream f(payload_path,std::ios::binary);f.seekg(payload_offset);f.read((char*)bf.data(),bf.size()*2);req(size_t(f.gcount())==bf.size()*2,"readonly payload short");std::vector<float>w(N*K);for(size_t i=0;i<w.size();++i){uint32_t u=uint32_t(bf[i])<<16;std::memcpy(&w[i],&u,4);}
 auto dw=alloc<float>(w.size()),dx=alloc<float>(X),dy=alloc<float>(Y);ck(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice));cudaStream_t s;ck(cudaStreamCreate(&s));
 cudaFuncAttributes a,b;ck(cudaFuncGetAttributes(&a,ds::gemv_f32_kernel));ck(cudaFuncGetAttributes(&b,ds::gemv_f32_prefill_pair_kernel));printf("RESOURCE old_regs=%d new_regs=%d old_shared=%zu new_shared=%zu block=256 Wbytes=%zu max_alloc_bytes=%zu\n",a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,w.size()*4,(w.size()+X+Y)*4);
 int bits=0;std::vector<float>x(X);for(auto c:std::vector<std::pair<int,int>>{{1,0},{16,1},{17,2},{31,0},{4367,0},{8192,0}}){int T=c.first,kind=c.second;for(size_t i=0;i<X;++i){x[i]=float(int((i*37+11)%257)-128)*0.00079f;if(kind==1)x[i]=(i&1)?-0.0f:0.0f;if(kind==2){static const uint32_t vals[]={0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc00001,0xffc00017,0x3f800000};uint32_t v=vals[i%9];std::memcpy(&x[i],&v,4);}}
  ck(cudaMemcpyAsync(dx,x.data(),X*4,cudaMemcpyHostToDevice,s));ck(cudaMemsetAsync(dy,0xa5,Y*4,s));old(dw,dx,T,dy,s);ck(cudaStreamSynchronize(s));auto ref=get(dy,Y);ck(cudaMemsetAsync(dy,0xa5,Y*4,s));run(1,dw,dx,T,dy,s);ck(cudaStreamSynchronize(s));auto got=get(dy,Y);same(ref,got,T,kind);++bits;printf("BITS T=%d kind=%d fallback=%d whole_output_padding_guard=PASS\n",T,kind,T<=16);
 }
 req(!std::memcmp(get(dw,w.size()).data(),w.data(),w.size()*4),"weights changed");req(!std::memcmp(get(dx,X).data(),x.data(),X*4),"inputs changed");
 cudaEvent_t b0,e0;ck(cudaEventCreate(&b0));ck(cudaEventCreate(&e0));for(int T:{4367,8192})for(int mode:{0,1,1,0}){for(int r=0;r<2;++r){run(0,dw,dx,T,dy,s);run(1,dw,dx,T,dy,s);}ck(cudaEventRecord(b0,s));for(int r=0;r<4;++r)run(mode,dw,dx,T,dy,s);ck(cudaEventRecord(e0,s));ck(cudaEventSynchronize(e0));float ms;ck(cudaEventElapsedTime(&ms,b0,e0));printf("TIME T=%d mode=%d reps=4 ms=%.9g both_warm=1\n",T,mode,ms);}
 ck(cudaEventDestroy(b0));ck(cudaEventDestroy(e0));ck(cudaStreamSynchronize(s));ck(cudaFree(dw));ck(cudaFree(dx));ck(cudaFree(dy));ck(cudaStreamDestroy(s));printf("PASS actual_prefill_router bits=%d timings=8\n",bits);return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
