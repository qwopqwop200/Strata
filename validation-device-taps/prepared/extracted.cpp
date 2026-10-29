#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
constexpr int kWin=16,kPicks=8;
constexpr bool kHipBuild=false;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
std::vector<std::function<void()>> queue;
int copies=0; bool fail_mv=false;
int cudaSetDevice(int){return 0;}
int cudaStreamSynchronize(int){for(auto& f:queue)f();queue.clear();return 0;}
int cudaStreamWaitEvent(int,int,int){return 0;}
int cudaMemcpyAsync(void* d,const void* s,size_t n,int,int){++copies;queue.push_back([=]{std::memcpy(d,s,n);});return 0;}
int cudaMemcpy2DAsync(void* d,size_t dp,const void* s,size_t sp,size_t w,size_t h,int,int){++copies;queue.push_back([=]{for(size_t t=0;t<h;++t)std::memcpy((char*)d+t*dp,(const char*)s+t*sp,w);});return 0;}
void hc_mean(const float* h,int T,int hc,int D,float* out,int stride,int){queue.push_back([=]{for(int t=0;t<T;++t)for(int e=0;e<D;++e){float a=0;for(int s=0;s<hc;++s)a+=h[(t*hc+s)*D+e];out[t*stride+e]=a/float(hc);}});}
#define CK(x) do{if((x)!=0){err="cuda double";return false;}}while(0)
struct Dev{int id=0,st=0;float *h=nullptr,*tap=nullptr,*xe=nullptr,*ed=nullptr;};
struct Job{const float* x;int n;size_t width;};
struct Engine{
 struct Geometry{int n_mtp=1,dim=5120,hd=1,hc=4;std::vector<int>d_targets{37,38,39};}g;
 std::vector<Dev> devs{1};int d_dev=0,ev_seed=0,d_main_proj=0;std::FILE*dump_d=nullptr;float*h_taps=nullptr;
 bool par=false;struct Options{int tp=1;}o;std::FILE*dump_h=nullptr;float*d_decode_taps=nullptr;
 std::vector<float> seen;
 Job job(int,const float*x,float*,int n,size_t width,int){return{x,n,width};}
 bool mv(Dev&,std::initializer_list<Job> jobs,std::string&err){if(fail_mv){err="injected";return false;}for(auto j:jobs)queue.push_back([this,j]{seen.insert(seen.end(),j.x,j.x+j.n*j.width);});return true;}
     float* decode_taps(int T) const {
        return T >= 1 && T <= kWin && !kHipBuild && devs.size() == 1 && !par && o.tp <= 1 &&
                       dump_d == nullptr && dump_h == nullptr ? d_decode_taps : nullptr;
    }


     bool tap(Dev& d, int il, int T, std::string& err, float* device_taps = nullptr) {
        const auto it = std::find(g.d_targets.begin(), g.d_targets.end(), il);
        if (g.n_mtp == 0 || it == g.d_targets.end()) return true;
        const size_t j = (size_t) (it - g.d_targets.begin()), nt = g.d_targets.size();
        const int D = g.dim;
        cudaSetDevice(d.id);
        if (device_taps != nullptr) {
            // Identical FP32 mean kernel/launch; only the output row stride changes.
            hc_mean(d.h, T, g.hc, D, device_taps + j * D, (int) nt * D, d.st);
            return true;
        }
        hc_mean(d.h, T, g.hc, D, d.tap, D, d.st);
        CK(cudaStreamWaitEvent(d.st, ev_seed, 0));
        CK(cudaMemcpy2DAsync(h_taps + j * D, nt * D * 4, d.tap, (size_t) D * 4, (size_t) D * 4, (size_t) T,
                             cudaMemcpyDeviceToHost, d.st));
        return true;
    }
     bool seed_drafter(int64_t p0, int n, int row0, std::string& err, const float* device_taps = nullptr) {
        if (g.n_mtp == 0 || n <= 0) return true;
        if (n > kWin * kPicks) {
            err = "seeding the drafter with " + std::to_string(n) + " rows";
            return false;
        }
        Dev& d = devs[(size_t) d_dev];
        cudaSetDevice(d.id);
        const int nt = (int) g.d_targets.size(), D = g.dim, HD = g.hd;
        if (dump_d != nullptr) {   // DS4_DSPARK_DUMP: 'S', the first position, the rows, then the taps
            for (Dev& e : devs) {
                cudaSetDevice(e.id);
                CK(cudaStreamSynchronize(e.st));
            }
            cudaSetDevice(d.id);
            const int64_t hdr[4] = {'S', p0, n, (int64_t) nt * D};
            std::fwrite(hdr, 8, 4, dump_d);
            std::fwrite(h_taps + (size_t) row0 * nt * D, 4, (size_t) n * nt * D, dump_d);
            std::fflush(dump_d);
        }
        for (int a = 0; a < n; a += kWin) {   // (a chunk's last 128 rows: 16 a GEMV)
            const int na = std::min(kWin, n - a);
            const float* taps = d.xe;
            if (device_taps != nullptr) {
                taps = device_taps + (size_t) (row0 + a) * nt * D;
            } else {
                CK(cudaMemcpyAsync(d.xe, h_taps + (size_t) (row0 + a) * nt * D, (size_t) na * nt * D * 4,
                                   cudaMemcpyHostToDevice, d.st));
            }
            if (!mv(d, {job(d_main_proj, taps, d.ed + (size_t) a * D, na, (size_t) nt * D, D)}, err)) return false;
        }
        return true;
    }
};
std::vector<float> run(int T,int kept,bool direct,int interrupted=0){
 Engine e;const int D=e.g.dim,H=e.g.hc,nt=3;
 std::vector<float>h(kWin*H*D),scratch(kWin*D),host(kWin*nt*D,333),gpu(kWin*nt*D+32,777),xe(kWin*nt*D),ed(kWin*D);
 e.h_taps=host.data();e.devs[0]={0,0,h.data(),scratch.data(),xe.data(),ed.data()};std::string err;
 auto taps=[&](int count,int salt){for(int j=0;j<count;++j){
 // Same layer storage is overwritten by an intervening queued layer, just like the engine.
 queue.push_back([&,j,salt]{for(size_t i=0;i<h.size();++i)h[i]=float(int((i*17+j*31+salt)%97)-48)/16.0f;});
 assert(e.tap(e.devs[0],37+j,T,err,direct?gpu.data()+16:nullptr));}};
 if(interrupted){taps(interrupted,7);cudaStreamSynchronize(0);assert(e.seen.empty());}
 taps(nt,13);cudaStreamSynchronize(0); // Existing forward's head fence.
 assert(e.seed_drafter(0,kept,0,err,direct?gpu.data()+16:nullptr));
 cudaStreamSynchronize(0);
 for(int i=0;i<16;++i){assert(gpu[i]==777);assert(gpu[16+kWin*nt*D+i]==777);}
 if(direct)for(float v:host)assert(v==333); // No hidden host consumer/update in this path.
 assert(e.seen.size()==size_t(kept*nt*D));
 return e.seen;
}
int main(){int cases=0;
 {Engine e;float f=0;e.d_decode_taps=&f;
  assert(e.decode_taps(1)==&f&&e.decode_taps(16)==&f);
  assert(!e.decode_taps(0)&&!e.decode_taps(17));
  e.par=true;assert(!e.decode_taps(1));e.par=false;
  e.o.tp=2;assert(!e.decode_taps(1));e.o.tp=1;
  e.dump_d=stdout;assert(!e.decode_taps(1));e.dump_d=nullptr;
  e.dump_h=stdout;assert(!e.decode_taps(1));e.dump_h=nullptr;
  e.devs.resize(2);assert(!e.decode_taps(1));e.devs.resize(1);
  e.d_decode_taps=nullptr;assert(!e.decode_taps(1));
 }
 for(int T:{1,2,6,16})for(int kept=1;kept<=T;++kept)for(int interrupted:{0,1,2,3}){
 copies=0;auto old=run(T,kept,false,interrupted);assert(copies==4+interrupted);
 copies=0;auto direct=run(T,kept,true,interrupted);assert(copies==0);
 assert(old.size()==direct.size()&&std::memcmp(old.data(),direct.data(),old.size()*sizeof(float))==0);++cases;
 }
 // Failed projection consumes no input; drain before the following fresh call.
 fail_mv=true;{Engine e;std::string err;float buf[3*5120]{};e.devs[0].xe=buf;e.devs[0].ed=buf;e.h_taps=buf;assert(!e.seed_drafter(0,1,0,err,buf));assert(e.seen.empty());}cudaStreamSynchronize(0);fail_mv=false;
 assert(run(2,1,false)==run(2,1,true));
 std::printf("actual tap + seed input extraction PASS cases=%d; queued CPU doubles, not CUDA/model proof\n",cases);
}
