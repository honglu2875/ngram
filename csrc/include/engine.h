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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "pool.h"
#include "shard.h"

namespace ngram {

// Scratch space for one query's per-shard ranges.
//
// A count is only a few hundred nanoseconds warm, so a malloc/free pair on the
// query path is a measurable fraction of it.  Indexes almost always have a
// handful of shards, so keep that case entirely on the stack.
struct RangeBuf {
    static constexpr size_t kInline = 16;

    explicit RangeBuf(size_t n) {
        if (n <= kInline) {
            _p = _inline;
        } else {
            _heap.resize(n);
            _p = _heap.data();
        }
    }
    Range *data() { return _p; }
    Range &operator[](size_t i) { return _p[i]; }
    const Range &operator[](size_t i) const { return _p[i]; }

  private:
    Range _inline[kInline];
    std::vector<Range> _heap;
    Range *_p;
};

// Per-thread buffer for "pattern plus one continuation token".  Reused across
// calls so the extension costs no allocation after the first query.
template <typename T>
inline std::vector<T> &extension_scratch() {
    static thread_local std::vector<T> v;
    return v;
}

// Query engine over a sharded suffix-array index.
//
// Shards are independent: an occurrence count is the sum over shards, so no
// merge step is needed at build time.  Single queries walk shards serially --
// with the in-RAM accelerator each shard costs only a handful of probes, and
// spreading one query over threads would cost more in synchronisation than it
// saves.  Parallelism is applied at the batch level instead, which is where the
// work actually is.
template <typename T>
class Engine {
  public:
    Engine(const std::vector<std::string> &shard_dirs, size_t num_threads, bool load_bucket)
        : _pool(num_threads) {
        _shards.resize(shard_dirs.size());
        for (size_t i = 0; i < shard_dirs.size(); i++) {
            _shards[i] = std::unique_ptr<Shard<T>>(new Shard<T>());
            _shards[i]->open(shard_dirs[i], load_bucket);
            _tok_cnt += _shards[i]->tok_cnt();
            _doc_cnt += _shards[i]->doc_cnt();
        }
        if (_shards.empty()) throw Error("index has no shards");
    }

    size_t num_shards() const { return _shards.size(); }
    U64 tok_cnt() const { return _tok_cnt; }
    U64 doc_cnt() const { return _doc_cnt; }
    size_t num_threads() const { return _pool.size(); }
    const Shard<T> &shard(size_t s) const { return *_shards[s]; }

    void set_accel(bool on) {
        for (auto &s : _shards) s->set_accel(on);
    }
    bool accel() const { return _shards[0]->accel(); }

    void warm() const {
        for (auto &s : _shards) s->warm();
    }
    void evict() const {
        for (auto &s : _shards) s->evict();
    }

    // ---- core queries ---------------------------------------------------------

    // Per-shard rank ranges for pat[0..len).  `hint` may be null, or an array of
    // num_shards() ranges from a shorter prefix of the same pattern.
    void find(const T *pat, U64 len, const Range *hint, Range *out) const {
        for (size_t s = 0; s < _shards.size(); s++) {
            Range h = hint ? hint[s] : _shards[s]->full_range();
            out[s] = _shards[s]->find(pat, len, h);
        }
    }

    U64 count(const T *pat, U64 len) const {
        U64 total = 0;
        for (auto &s : _shards) total += s->find(pat, len, s->full_range()).size();
        return total;
    }

    static U64 total(const Range *r, size_t n) {
        U64 t = 0;
        for (size_t i = 0; i < n; i++) t += r[i].size();
        return t;
    }

    // count(prompt + [cont]) / count(prompt), with the continuation searched
    // inside the prompt's own range rather than from scratch.
    void prob(const T *prompt, U64 len, T cont, U64 *prompt_cnt, U64 *cont_cnt) const {
        RangeBuf r(_shards.size());
        find(prompt, len, nullptr, r.data());
        auto &ext = extension_scratch<T>();
        ext.assign(prompt, prompt + len);
        ext.push_back(cont);
        U64 pc = 0, cc = 0;
        for (size_t s = 0; s < _shards.size(); s++) {
            pc += r[s].size();
            cc += _shards[s]->find(ext.data(), len + 1, r[s]).size();
        }
        *prompt_cnt = pc;
        *cont_cnt = cc;
    }

    // Distribution over the token that follows pat[0..len).  `max_support`
    // caps the work: if the pattern occurs far more often than that, ranks are
    // sampled and counts extrapolated (this is what makes very frequent
    // contexts answerable at all).  0 means exact.
    void ntd(const T *pat, U64 len, U64 max_support, std::vector<NtdEntry> *out) const {
        RangeBuf r(_shards.size());
        find(pat, len, nullptr, r.data());
        ntd_from_ranges(len, r.data(), max_support, out);
    }

    // Only the pattern's *length* is needed here: the ranges already identify
    // the occurrences, and the next token sits `len` positions past each.
    void ntd_from_ranges(U64 len, const Range *ranges, U64 max_support,
                         std::vector<NtdEntry> *out) const {
        out->clear();
        U64 cnt = total(ranges, _shards.size());
        if (cnt == 0) return;

        U64 unit = 1;
        if (max_support > 0) {
            while (cnt > unit * max_support) unit <<= 1;
        }

        if (_shards.size() == 1) {
            _ntd_shard(*_shards[0], ranges[0], len, unit, out);
            return;
        }
        std::vector<NtdEntry> merged, part;
        for (size_t s = 0; s < _shards.size(); s++) {
            part.clear();
            _ntd_shard(*_shards[s], ranges[s], len, unit, &part);
            _merge_sorted(*out, part, &merged);
            out->swap(merged);
        }
    }

    // ---- infinite-gram --------------------------------------------------------

    // Length of the longest suffix of pat[0..len) that occurs in the corpus.
    // Occurrence count is non-increasing in suffix length, so we binary-lift to
    // bracket the answer and then bisect: O(log len) searches, each seeded from
    // the unigram table.  `ranges_out` receives the winning suffix's ranges.
    U64 longest_suffix(const T *pat, U64 len, U64 max_len, Range *ranges_out,
                       U64 *cnt_out) const {
        const size_t S = _shards.size();
        if (max_len > 0 && len > max_len) {
            pat += len - max_len;
            len = max_len;
        }
        U64 lo = 0;                // known to occur (the empty pattern always does)
        U64 hi = len + 1;          // known not to occur (nothing is longer than the prompt)
        for (U64 probe = 1; probe <= len; probe <<= 1) {
            if (count(pat + len - probe, probe) > 0) {
                lo = probe;
            } else {
                hi = probe;
                break;
            }
        }
        while (hi - lo > 1) {
            U64 mid = lo + (hi - lo) / 2;
            if (count(pat + len - mid, mid) > 0) lo = mid; else hi = mid;
        }
        find(pat + len - lo, lo, nullptr, ranges_out);
        if (cnt_out) *cnt_out = total(ranges_out, S);
        return lo;
    }

    InfgramResult infgram_prob(const T *prompt, U64 len, T cont, U64 max_len) const {
        RangeBuf r(_shards.size());
        U64 pc = 0;
        U64 l = longest_suffix(prompt, len, max_len, r.data(), &pc);
        auto &ext = extension_scratch<T>();
        ext.assign(prompt + len - l, prompt + len);
        ext.push_back(cont);
        U64 cc = 0;
        for (size_t s = 0; s < _shards.size(); s++) {
            cc += _shards[s]->find(ext.data(), l + 1, r[s]).size();
        }
        InfgramResult res;
        res.suffix_len = l;
        res.prompt_cnt = pc;
        res.cont_cnt = cc;
        res.prob = pc ? static_cast<double>(cc) / static_cast<double>(pc) : 0.0;
        return res;
    }

    U64 infgram_ntd(const T *prompt, U64 len, U64 max_support, U64 max_len,
                    std::vector<NtdEntry> *out, U64 *prompt_cnt) const {
        RangeBuf r(_shards.size());
        U64 pc = 0;
        U64 l = longest_suffix(prompt, len, max_len, r.data(), &pc);
        ntd_from_ranges(l, r.data(), max_support, out);
        if (prompt_cnt) *prompt_cnt = pc;
        return l;
    }

    // ---- batched --------------------------------------------------------------

    // counts[b] = count(pats + b*stride, lens[b])
    void count_batch(const T *pats, const U32 *lens, U64 stride, U64 n, U64 *counts) const {
        _pool.parallel_for(0, n, [&](U64 b) {
            counts[b] = count(pats + b * stride, lens[b]);
        });
    }

    void prob_batch(const T *prompts, const U32 *lens, U64 stride, const T *conts, U64 n,
                    U64 *prompt_cnts, U64 *cont_cnts) const {
        _pool.parallel_for(0, n, [&](U64 b) {
            prob(prompts + b * stride, lens[b], conts[b], &prompt_cnts[b], &cont_cnts[b]);
        });
    }

    // For each row, walk left to right computing the infinite-gram probability
    // of the token actually observed at each position.
    //
    // The longest matching suffix can grow by at most one per step, so each
    // position starts from the previous answer plus one and walks down on
    // failure.  Total downward steps across a row are bounded by the total
    // upward steps, making this amortised O(1) searches per position instead of
    // O(log n) -- the single biggest win for scoring long sequences.
    void infgram_batch(const T *tokens, U64 rows, U64 cols, U64 max_len, double *probs,
                       U32 *suffix_lens, U64 *prompt_cnts, U64 *cont_cnts) const {
        const size_t S = _shards.size();
        _pool.parallel_for(0, rows, [&](U64 row) {
            const T *seq = tokens + row * cols;
            RangeBuf r(S);
            auto &ext = extension_scratch<T>();
            U64 prev = 0;
            for (U64 t = 0; t < cols; t++) {
                U64 cap = t;
                if (max_len > 0 && cap > max_len) cap = max_len;
                U64 l = prev + 1;
                if (l > cap) l = cap;
                for (;;) {
                    find(seq + t - l, l, nullptr, r.data());
                    if (l == 0 || total(r.data(), S) > 0) break;
                    l--;
                }
                U64 pc = total(r.data(), S);
                ext.assign(seq + t - l, seq + t);
                ext.push_back(seq[t]);
                U64 cc = 0;
                for (size_t s = 0; s < S; s++) {
                    cc += _shards[s]->find(ext.data(), l + 1, r[s]).size();
                }
                U64 idx = row * cols + t;
                probs[idx] = pc ? static_cast<double>(cc) / static_cast<double>(pc) : 0.0;
                suffix_lens[idx] = static_cast<U32>(l);
                if (prompt_cnts) prompt_cnts[idx] = pc;
                if (cont_cnts) cont_cnts[idx] = cc;
                prev = l;
            }
        });
    }

    // Next-token distribution at every position of every row, returned as a
    // ragged (CSR-style) structure.  `infinite` selects between a fixed context
    // window and the longest matching suffix.
    void ntd_batch(const T *tokens, U64 rows, U64 cols, U64 max_ctx, U64 max_support,
                   bool infinite, std::vector<std::vector<NtdEntry>> *out) const {
        out->assign(rows * cols, std::vector<NtdEntry>());
        const size_t S = _shards.size();
        _pool.parallel_for(0, rows * cols, [&](U64 idx) {
            U64 row = idx / cols, t = idx % cols;
            const T *seq = tokens + row * cols;
            RangeBuf r(S);
            U64 l;
            if (infinite) {
                l = longest_suffix(seq, t, max_ctx, r.data(), nullptr);
            } else {
                l = (max_ctx > 0 && t > max_ctx) ? max_ctx : t;
                find(seq + t - l, l, nullptr, r.data());
            }
            ntd_from_ranges(l, r.data(), max_support, &(*out)[idx]);
        });
    }

  private:
    // Walk the rank range, splitting until both ends agree on the next token.
    // Cost is O(distinct * log range) probes.  Iterative on purpose: the
    // reference implementation spawns a thread per recursion node, which
    // dominates its own runtime for anything but tiny ranges.
    void _ntd_shard(const Shard<T> &sh, Range r, U64 len, U64 unit,
                    std::vector<NtdEntry> *out) const {
        if (r.empty()) return;
        U64 lo = r.lo, hi = r.hi;

        // At most one occurrence can sit flush against the end of the shard, and
        // it sorts first among equals because shorter suffixes come first.
        T dummy;
        if (!sh.next_token(lo, len, &dummy)) {
            lo++;
            if (lo >= hi) return;
        }

        struct Frame {
            U64 lo, hi;
            T tlo, thi;
            bool has_lo, has_hi;
        };
        std::vector<Frame> stack;
        stack.push_back(Frame{lo, hi, 0, 0, false, false});

        while (!stack.empty()) {
            Frame f = stack.back();
            stack.pop_back();
            if (f.lo >= f.hi) continue;

            if (f.hi - f.lo < 4 * unit) {
                // Base case.  With unit == 1 this enumerates the (at most three)
                // remaining ranks exactly; with unit > 1 it samples one rank per
                // unit and credits the whole unit to it.
                for (U64 rank = f.lo; rank < f.hi; rank += unit) {
                    U64 span = (rank + unit <= f.hi) ? unit : (f.hi - rank);
                    U64 probe = rank + span / 2;
                    T tok;
                    if (!sh.next_token(probe, len, &tok)) continue;
                    _emit(out, tok, span);
                }
                continue;
            }

            T a, b;
            if (f.has_lo) {
                a = f.tlo;
            } else if (!sh.next_token(f.lo, len, &a)) {
                continue;
            }
            if (f.has_hi) {
                b = f.thi;
            } else if (!sh.next_token(f.hi - 1, len, &b)) {
                continue;
            }
            if (a == b) {
                _emit(out, a, f.hi - f.lo);
                continue;
            }
            U64 mid = f.lo + (f.hi - f.lo) / 2;
            stack.push_back(Frame{mid, f.hi, 0, b, false, true});
            stack.push_back(Frame{f.lo, mid, a, 0, true, false});
        }
    }

    // Emissions arrive in increasing token order because the rank range is
    // sorted by what follows the pattern, so coalescing the tail is enough.
    static void _emit(std::vector<NtdEntry> *out, T tok, U64 n) {
        if (!out->empty() && out->back().token == static_cast<U32>(tok)) {
            out->back().count += n;
        } else {
            out->push_back(NtdEntry{static_cast<U32>(tok), n});
        }
    }

    static void _merge_sorted(const std::vector<NtdEntry> &a, const std::vector<NtdEntry> &b,
                              std::vector<NtdEntry> *out) {
        out->clear();
        out->reserve(a.size() + b.size());
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i].token < b[j].token) out->push_back(a[i++]);
            else if (b[j].token < a[i].token) out->push_back(b[j++]);
            else {
                out->push_back(NtdEntry{a[i].token, a[i].count + b[j].count});
                i++;
                j++;
            }
        }
        while (i < a.size()) out->push_back(a[i++]);
        while (j < b.size()) out->push_back(b[j++]);
    }

    std::vector<std::unique_ptr<Shard<T>>> _shards;
    mutable ThreadPool _pool;
    U64 _tok_cnt = 0;
    U64 _doc_cnt = 0;
};

}  // namespace ngram
