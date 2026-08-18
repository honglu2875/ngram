// Copyright 2024 Honglu Fan (https://github.com/honglu2875).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "common.h"

namespace ngram {

// A persistent pool of worker threads.
//
// Queries are memory-latency bound, so the thing that matters is never paying
// thread-creation cost on the query path.  The pool is created once when an
// index is opened and reused for every batched call.  Work is handed out as a
// dynamically-claimed index range, which keeps stragglers from dominating when
// some positions in a batch are far more expensive than others.
//
// The job descriptor lives on the caller's stack, so the handshake has to
// guarantee that no worker can touch it after parallel_for returns: workers
// register themselves under the mutex, and the caller clears the job pointer
// and drains the registration count while holding that same mutex.
class ThreadPool {
  public:
    explicit ThreadPool(size_t num_threads) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
        }
        _num_threads = num_threads;
        // The calling thread participates in every job, so we spawn one fewer.
        _workers.reserve(num_threads - 1);
        for (size_t i = 0; i + 1 < num_threads; i++) {
            _workers.emplace_back([this] { _worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(_mu);
            _stop = true;
        }
        _cv.notify_all();
        for (auto &t : _workers) {
            if (t.joinable()) t.join();
        }
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    size_t size() const { return _num_threads; }

    // Run fn(i) for every i in [begin, end), blocking until all have completed.
    // fn must be thread-safe and must not throw.
    void parallel_for(U64 begin, U64 end, const std::function<void(U64)> &fn) {
        if (end <= begin) return;
        const U64 n = end - begin;
        if (n == 1 || _num_threads <= 1 || _workers.empty()) {
            for (U64 i = begin; i < end; i++) fn(i);
            return;
        }

        // Several chunks per worker: fine enough to balance uneven work, coarse
        // enough that the shared atomic is not the bottleneck.
        U64 chunk = n / (_num_threads * 8);
        if (chunk == 0) chunk = 1;

        Job job;
        job.fn = &fn;
        job.end = end;
        job.chunk = chunk;
        job.next.store(begin, std::memory_order_relaxed);

        {
            std::unique_lock<std::mutex> lock(_mu);
            _job = &job;
            _generation++;
        }
        _cv.notify_all();

        _run_job(job);

        std::unique_lock<std::mutex> lock(_mu);
        // No worker can register for this job past this point.
        _job = nullptr;
        _done_cv.wait(lock, [this] { return _active == 0; });
    }

  private:
    struct Job {
        const std::function<void(U64)> *fn;
        U64 end;
        U64 chunk;
        std::atomic<U64> next;
    };

    static void _run_job(Job &job) {
        for (;;) {
            U64 start = job.next.fetch_add(job.chunk, std::memory_order_relaxed);
            if (start >= job.end) break;
            U64 stop = start + job.chunk;
            if (stop > job.end) stop = job.end;
            for (U64 i = start; i < stop; i++) (*job.fn)(i);
        }
    }

    void _worker_loop() {
        U64 seen = 0;
        for (;;) {
            Job *job = nullptr;
            {
                std::unique_lock<std::mutex> lock(_mu);
                _cv.wait(lock, [this, seen] { return _stop || _generation != seen; });
                if (_stop) return;
                seen = _generation;
                job = _job;
                if (job) _active++;
            }
            if (!job) continue;

            _run_job(*job);

            {
                std::unique_lock<std::mutex> lock(_mu);
                _active--;
            }
            _done_cv.notify_all();
        }
    }

    std::vector<std::thread> _workers;
    std::mutex _mu;
    std::condition_variable _cv;
    std::condition_variable _done_cv;
    bool _stop = false;
    Job *_job = nullptr;
    U64 _generation = 0;
    int _active = 0;
    size_t _num_threads = 1;
};

}  // namespace ngram
