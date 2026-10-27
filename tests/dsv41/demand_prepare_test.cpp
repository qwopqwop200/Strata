#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ds4_demand_prepare.hpp"
#include "ds4_os.hpp"
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#if defined(__linux__)
#include <sys/mman.h>
#endif
using strata::ds4::DemandPrepare;
int main() {
    // Confirm overlap without a timing threshold: compute releases a blocked prepare.
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    DemandPrepare p([&](const auto&) { entered.set_value(); gate.wait(); });
    {
        auto ticket = p.start({{nullptr, 1}});
        entered.get_future().wait();
        bool rejected = false;
        try { auto invalid = p.start({}); } catch (const std::logic_error&) { rejected = true; }
        assert(rejected);
        release.set_value(); // main/compute progressed while preparation was blocked
        ticket.finish();
    }
    assert(p.stats().batches == 1 && p.stats().ranges == 1);
    // The production optional moves the returned ticket. A moved-from ticket
    // must not drain a later batch when its scope ends.
    std::atomic<int> moved_calls{0};
    std::promise<void> next_entered, next_release;
    auto next_gate = next_release.get_future().share();
    DemandPrepare moving([&](const auto&) {
        if (++moved_calls == 2) { next_entered.set_value(); next_gate.wait(); }
    });
    {
        auto original = std::make_unique<DemandPrepare::Ticket>(moving.start({{nullptr, 1}}));
        auto moved = std::move(*original);
        moved.finish();
        auto next = moving.start({{nullptr, 1}});
        next_entered.get_future().wait();
        original.reset(); // must not wait for the later batch that compute releases below
        next_release.set_value();
        next.finish();
    }
    assert(moving.stats().batches == 2 && moved_calls == 2);
    // Exceptions propagate at finish; the slot remains reusable, and destruction drains.
    int calls = 0;
    DemandPrepare failing([&](const auto&) { if (++calls == 1) throw std::runtime_error("injected"); });
    bool caught = false;
    try { auto t = failing.start({{nullptr, 1}}); t.finish(); }
    catch (const std::runtime_error&) { caught = true; }
    assert(caught);
    { auto t = failing.start({{nullptr, 1}}); t.finish(); }
    { auto t = failing.start({}); t.finish(); }
    assert(failing.stats().batches == 3);
    // A compute exception must survive a simultaneous preparation exception. The
    // ticket destructor drains without throwing, then the same worker is reusable.
    std::promise<void> failed;
    auto failure_seen = failed.get_future();
    std::atomic<int> fault_calls{0};
    DemandPrepare unwind([&](const auto&) {
        if (++fault_calls == 1) { failed.set_value(); throw std::runtime_error("prepare"); }
    });
    bool compute_error = false;
    try {
        auto t = unwind.start({{nullptr, 1}});
        failure_seen.wait();
        throw std::logic_error("compute");
    } catch (const std::logic_error&) { compute_error = true; }
    assert(compute_error);
    { auto t = unwind.start({{nullptr, 1}}); t.finish(); }
    assert(unwind.stats().batches == 2 && fault_calls == 2);
    // Early exit cancels queued ranges; a running range must complete before release.
    std::promise<void> running, unblock;
    auto wait = unblock.get_future().share();
    std::atomic<int> completed{0};
    DemandPrepare cancelled([&](const auto&) { if (++completed == 1) { running.set_value(); wait.wait(); } });
    {
        auto t = cancelled.start(std::vector<DemandPrepare::Range>(64, {nullptr, 1}));
        running.get_future().wait();
        t.request_cancel();
        unblock.set_value();
    }
    assert(cancelled.stats().batches == 1 && completed == 1 && cancelled.stats().cancelled == 63);
#if defined(__linux__)
    // Same immutable mapped bytes, real residency/population functions and exact float results.
    constexpr size_t bytes = 8 << 20;
    void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(mapping != MAP_FAILED);
    auto* x = static_cast<float*>(mapping);
    for (size_t i = 0; i < bytes / sizeof(float); ++i) x[i] = float(int(i % 251) - 125) / 128;
    mprotect(mapping, bytes, PROT_READ);
    auto compute = [&] { double result = 0; for (size_t i = 0; i < bytes / sizeof(float); ++i) result += x[i] * x[i]; return result; };
    const std::vector<DemandPrepare::Range> ranges{{mapping, bytes}};
    strata::ds4::os::prefetch_chunked(ranges, 128 << 10);
    const double reference = compute();
    DemandPrepare mapped([](const auto& r) { strata::ds4::os::prefetch_chunked({r}, 128 << 10); });
    double candidate;
    { auto t = mapped.start(ranges); candidate = compute(); t.finish(); }
    assert(std::memcmp(&reference, &candidate, sizeof(double)) == 0);
    assert(mapped.stats().ranges == 1);
    munmap(mapping, bytes); // no in-flight pointer remains
#endif
    std::cout << "PASS overlap, single-slot ownership, failure/reuse, cancellation/drain, mapped exact numerical result\n";
}
