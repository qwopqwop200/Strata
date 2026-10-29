#ifdef NDEBUG
#undef NDEBUG
#endif
#include <pthread.h>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <cassert>
#include "ds4_pool.hpp"
static std::atomic<bool> release_workers{false};
static std::atomic<int> entered{0};
struct Start { void*(*function)(void*);void* argument; };
static void* delayed_start(void* argument) {
    std::unique_ptr<Start> start(static_cast<Start*>(argument));
    entered.fetch_add(1);
    while(!release_workers.load(std::memory_order_acquire))std::this_thread::yield();
    return start->function(start->argument);
}
extern "C" int pthread_create(pthread_t* thread,const pthread_attr_t* attr,void*(*function)(void*),void* argument) {
    using Real=int(*)(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
    static const auto real=reinterpret_cast<Real>(dlsym(RTLD_NEXT,"pthread_create"));
    assert(real);
    auto start=std::make_unique<Start>(Start{function,argument});
    const int result=real(thread,attr,delayed_start,start.get());
    if(!result)start.release();
    return result;
}
int main(int argc,char** argv) {
    const bool expect_miss=argc==2 && !std::strcmp(argv[1],"--expect-old-miss");
    {
        strata::ds4::Pool pool(2,{},0);
        while(entered.load()!=2)std::this_thread::yield();
        std::atomic<int> visits[8]{};std::atomic<int> done{0};
        std::function<void(int)> job=[&](int i){visits[i].fetch_add(1);done.fetch_add(1);};
        pool.start(8,job); // API permits no caller help, e.g. a second NUMA pool
        release_workers.store(true,std::memory_order_release);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(300);
        while(done.load()!=8&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
        const int before_help=done.load();
        pool.help();pool.wait(); // rescue the old defect, do not strand the process
        assert(before_help==(expect_miss?0:8));
        for(auto& count:visits)assert(count.load()==1);
        std::printf("PASS controlled late first start/nohelp old_expected=%d completed_before_help=%d/8\n",expect_miss,before_help);
        // Reuse a live pool. Callback captures cannot escape run()/wait().
        for(int round=0;round<32;++round) {
            auto payload=std::make_unique<std::atomic<int>>(0);
            pool.run(37,[&](int){payload->fetch_add(1);});assert(payload->load()==37);
            payload.reset();
            pool.run(0,[](int){assert(false);});
        }
        std::puts("PASS repeated dispatch exact-once, empty dispatch, captured storage lifetime");
        // A start() callback must stay alive until wait(); complete a blocked job
        // before releasing its captured storage and immediately reuse the pool.
        std::promise<void> running,release;auto go=release.get_future().share();
        auto value=std::make_unique<std::atomic<int>>(0);
        std::function<void(int)> blocked=[&](int){running.set_value();go.wait();value->store(7);};
        pool.start(1,blocked);running.get_future().wait();release.set_value();pool.wait();
        assert(value->load()==7);value.reset();pool.run(1,[](int){});
        std::puts("PASS start/wait keeps callback and output lifetime bounded");
    }
    // Caller-help contract is distinct: zero workers must still run every job.
    {
        strata::ds4::Pool caller_only(0,{},0);int done=0;
        caller_only.run(41,[&](int){++done;});assert(done==41);
        std::puts("PASS zero-worker Pool::run caller-help progress");
    }
    // Existing destructor joins an executing worker. Captures outlive the pool.
    {
        std::promise<void> running,release;auto go=release.get_future().share();
        std::atomic<int> completed{0};
        std::function<void(int)> blocked=[&](int){running.set_value();go.wait();completed.store(1);};
        auto pool=std::make_unique<strata::ds4::Pool>(2,std::vector<int>{},0);
        // Readiness itself uses run/help and preserves the production contract.
        pool->run(2,[](int){});
        pool->start(1,blocked);running.get_future().wait();
        std::atomic<bool> destroying{false};
        std::thread releaser([&]{while(!destroying.load())std::this_thread::yield();release.set_value();});
        destroying.store(true);pool.reset();releaser.join();assert(completed.load()==1);
        std::puts("PASS destructor joins running callback with valid captures");
    }
    // Destruction with no submitted work must wake and join initially idle workers.
    for(int i=0;i<8;++i){strata::ds4::Pool idle(2,{},0);}
    std::puts("PASS idle worker shutdown/join; no arithmetic or scheduling policy changed");
}
