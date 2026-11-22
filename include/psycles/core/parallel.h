#pragma once

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace psycles {

/// Returns the default worker thread count (including the calling thread),
/// respecting a positive `PSYCLES_THREADS` environment variable override.
[[nodiscard]] inline std::size_t default_parallel_threads() noexcept {
    if (const char *raw = std::getenv("PSYCLES_THREADS");
        raw != nullptr && *raw != '\0') {
        const std::string_view sv{raw};
        std::size_t parsed = 0u;
        const auto [ptr, ec] = std::from_chars(
            sv.data(), sv.data() + sv.size(), parsed);
        if (ec == std::errc{} && ptr == sv.data() + sv.size() && parsed > 0u) {
            return std::clamp<std::size_t>(parsed, 1u, 64u);
        }
    }
    const auto hw = static_cast<std::size_t>(std::thread::hardware_concurrency());
    return std::clamp<std::size_t>(hw == 0u ? 1u : hw, 1u, 32u);
}

namespace detail {

class ParallelThreadPool {
private:
    struct Batch {
        std::function<void(std::size_t)> run;
        std::size_t count{};
        std::atomic<std::size_t> next_index{0u};
        std::atomic<bool> failed{false};
        std::mutex error_mutex;
        std::exception_ptr first_exception;
        std::mutex done_mutex;
        std::condition_variable done_cv;
        std::size_t started_workers{0u};
        std::size_t finished_workers{0u};
    };

public:
    ParallelThreadPool() {
        const auto total_threads = default_parallel_threads();
        const auto worker_count = total_threads > 1u ? total_threads - 1u : 0u;
        _workers.reserve(worker_count);
        for (std::size_t i = 0u; i < worker_count; ++i) {
            _workers.emplace_back([this]() { worker_loop(); });
        }
    }

    ~ParallelThreadPool() { shutdown(); }

    ParallelThreadPool(const ParallelThreadPool &) = delete;
    ParallelThreadPool &operator=(const ParallelThreadPool &) = delete;

    [[nodiscard]] std::size_t max_threads() const noexcept {
        return _workers.size() + 1u;
    }

    void shutdown() noexcept {
        {
            std::scoped_lock lock{_mutex};
            if (_stop) {
                return;
            }
            _stop = true;
            _active_dispatch_workers = 0u;
            _current_batch.reset();
        }
        _work_cv.notify_all();
        for (auto &worker : _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template <typename Fn>
    void parallel_for(
        std::size_t count,
        Fn &&fn,
        std::size_t max_threads_limit = 0u) {
        if (count == 0u) {
            return;
        }
        const auto pool_limit = max_threads();
        const auto effective_threads =
            max_threads_limit == 0u
                ? pool_limit
                : std::min<std::size_t>(max_threads_limit, pool_limit);
        if (count == 1u || effective_threads <= 1u) {
            for (std::size_t i = 0u; i < count; ++i) {
                fn(i);
            }
            return;
        }

        // Non-blocking submit guard: if another independent caller is already
        // saturating the pool, execute sequentially to avoid contention or
        // cross-thread deadlock.
        std::unique_lock<std::mutex> submit_lock{_submit_mutex, std::try_to_lock};
        if (!submit_lock.owns_lock()) {
            for (std::size_t i = 0u; i < count; ++i) {
                fn(i);
            }
            return;
        }

        const auto active_workers = std::min<std::size_t>(
            effective_threads - 1u,
            std::min<std::size_t>(count - 1u, _workers.size()));
        if (active_workers == 0u) {
            for (std::size_t i = 0u; i < count; ++i) {
                fn(i);
            }
            return;
        }

        auto batch = std::make_shared<Batch>();
        batch->run = [&fn](std::size_t index) { fn(index); };
        batch->count = count;

        {
            std::scoped_lock lock{_mutex};
            if (_stop) {
                submit_lock.unlock();
                for (std::size_t i = 0u; i < count; ++i) {
                    fn(i);
                }
                return;
            }
            _current_batch = batch;
            ++_generation;
            _active_dispatch_workers = active_workers;
        }
        _work_cv.notify_all();

        run_batch_items(*batch);

        std::size_t started_workers = 0u;
        {
            std::scoped_lock lock{_mutex};
            _active_dispatch_workers = 0u;
            if (_current_batch == batch) {
                _current_batch.reset();
            }
            started_workers = batch->started_workers;
        }

        {
            std::unique_lock<std::mutex> done_lock{batch->done_mutex};
            batch->done_cv.wait(done_lock, [&]() {
                return batch->finished_workers >= started_workers;
            });
        }

        if (batch->first_exception) {
            std::rethrow_exception(batch->first_exception);
        }
    }

private:
    static void run_batch_items(Batch &batch) noexcept {
        while (!batch.failed.load(std::memory_order_acquire)) {
            const auto idx =
                batch.next_index.fetch_add(1u, std::memory_order_relaxed);
            if (idx >= batch.count) {
                break;
            }
            try {
                batch.run(idx);
            } catch (...) {
                batch.failed.store(true, std::memory_order_release);
                std::scoped_lock err_lock{batch.error_mutex};
                if (!batch.first_exception) {
                    batch.first_exception = std::current_exception();
                }
                break;
            }
        }
    }

    void worker_loop() noexcept {
        std::size_t seen_generation = 0u;
        for (;;) {
            std::shared_ptr<Batch> batch;
            {
                std::unique_lock<std::mutex> lock{_mutex};
                _work_cv.wait(lock, [&]() {
                    return _stop ||
                           (_generation != seen_generation &&
                            _active_dispatch_workers > 0u &&
                            _current_batch != nullptr);
                });
                if (_stop) {
                    return;
                }
                seen_generation = _generation;
                --_active_dispatch_workers;
                batch = _current_batch;
                ++batch->started_workers;
            }
            run_batch_items(*batch);
            {
                std::scoped_lock<std::mutex> done_lock{batch->done_mutex};
                ++batch->finished_workers;
                batch->done_cv.notify_one();
            }
        }
    }

    std::mutex _submit_mutex;
    std::mutex _mutex;
    std::condition_variable _work_cv;
    std::vector<std::thread> _workers;
    std::shared_ptr<Batch> _current_batch;
    std::size_t _generation{0u};
    std::size_t _active_dispatch_workers{0u};
    bool _stop{false};
};

[[nodiscard]] inline ParallelThreadPool &global_thread_pool() {
    static ParallelThreadPool pool;
    return pool;
}

[[nodiscard]] inline bool &in_parallel_for_flag() noexcept {
    thread_local bool active = false;
    return active;
}

} // namespace detail

/// Execute `fn(index)` for `index` in `[0, count)` across up to `max_threads`
/// workers (0 uses `default_parallel_threads()`). Nested calls run sequentially
/// on the caller to prevent re-entrancy deadlocks.
template <typename Fn>
inline void parallel_for(
    std::size_t count,
    Fn &&fn,
    std::size_t max_threads = 0u) {
    if (count == 0u) {
        return;
    }
    auto &nested = detail::in_parallel_for_flag();
    if (nested || count == 1u || default_parallel_threads() <= 1u ||
        max_threads == 1u) {
        for (std::size_t i = 0u; i < count; ++i) {
            fn(i);
        }
        return;
    }
    struct Guard {
        bool &flag;
        explicit Guard(bool &f) noexcept : flag{f} { flag = true; }
        ~Guard() noexcept { flag = false; }
    } guard{nested};
    detail::global_thread_pool().parallel_for(
        count, std::forward<Fn>(fn), max_threads);
}

inline void shutdown_parallel_pool() noexcept {
    detail::global_thread_pool().shutdown();
}

} // namespace psycles
