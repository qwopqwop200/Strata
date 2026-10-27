// One confirmed CPU demand batch in flight; no weights or result buffers are owned here.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace strata::ds4 {
class DemandPrepare {
public:
    using Range = std::pair<const void*, size_t>;
    using Prepare = std::function<void(const Range&)>;
    struct Stats { uint64_t batches = 0, ranges = 0, cancelled = 0, prepare_ns = 0, tail_ns = 0; };
    explicit DemandPrepare(Prepare prepare) : prepare_(std::move(prepare)), worker_([this] { loop(); }) {}
    DemandPrepare(const DemandPrepare&) = delete;
    DemandPrepare& operator=(const DemandPrepare&) = delete;
    ~DemandPrepare() {
        drain(true);
        { std::lock_guard<std::mutex> lock(mu_); stop_ = true; }
        cv_.notify_all();
        worker_.join();
    }
    class Ticket {
    public:
        explicit Ticket(DemandPrepare& owner) : owner_(&owner) {}
        Ticket(Ticket&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        // Early exit cancels ranges not started yet and waits for a running OS call.
        ~Ticket() { if (owner_) owner_->drain(true); }
        void request_cancel() noexcept { if (owner_) owner_->cancel_.store(true, std::memory_order_relaxed); }
        void finish() {
            auto* p = owner_;
            if (!p) return;
            auto error = p->drain(false);
            owner_ = nullptr;
            if (error) std::rethrow_exception(error);
        }
    private:
        DemandPrepare* owner_;
    };
    Ticket start(std::vector<Range> ranges) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (active_) throw std::logic_error("CPU demand preparation already in flight");
            ranges_ = std::move(ranges);
            error_ = nullptr;
            cancel_.store(false, std::memory_order_relaxed);
            active_ = pending_ = true;
            done_ = false;
        }
        cv_.notify_all();
        return Ticket(*this);
    }
    Stats stats() const { std::lock_guard<std::mutex> lock(mu_); return stats_; }
private:
    using Clock = std::chrono::steady_clock;
    static uint64_t ns(Clock::duration t) { return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count(); }
    std::exception_ptr drain(bool cancel) noexcept {
        const auto begin = Clock::now();
        std::unique_lock<std::mutex> lock(mu_);
        if (!active_) return nullptr;
        if (cancel) cancel_.store(true, std::memory_order_relaxed);
        cv_.wait(lock, [&] { return done_; });
        stats_.tail_ns += ns(Clock::now() - begin);
        active_ = false;
        ranges_.clear();  // mapped pointers cannot escape the caller's lifetime
        return error_;
    }
    void loop() {
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            cv_.wait(lock, [&] { return pending_ || stop_; });
            if (stop_) return;
            pending_ = false;
            lock.unlock();
            const auto begin = Clock::now();
            uint64_t completed = 0;
            std::exception_ptr error;
            try {
                for (const auto& range : ranges_) {
                    if (cancel_.load(std::memory_order_relaxed)) break;
                    prepare_(range);
                    ++completed;
                }
            } catch (...) { error = std::current_exception(); }
            const auto elapsed = ns(Clock::now() - begin);
            lock.lock();
            ++stats_.batches;
            stats_.ranges += completed;
            stats_.cancelled += ranges_.size() - completed;
            stats_.prepare_ns += elapsed;
            error_ = error;
            done_ = true;
            cv_.notify_all();
        }
    }
    Prepare prepare_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Range> ranges_;
    std::exception_ptr error_;
    std::atomic<bool> cancel_{false};
    bool active_ = false, pending_ = false, done_ = false, stop_ = false;
    Stats stats_;
    std::thread worker_;
};
} // namespace strata::ds4
