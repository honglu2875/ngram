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
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "engine.h"

namespace ngram {

// Token-width-erased view of an Engine.
//
// The index knows its own token width, so dispatching once at the call boundary
// keeps the hot loops monomorphic while letting the Python layer see a single
// class.  Batched entry points dispatch once for the whole batch.
class AnyEngine {
  public:
    virtual ~AnyEngine() = default;

    virtual int token_width() const = 0;
    virtual U64 tok_cnt() const = 0;
    virtual U64 doc_cnt() const = 0;
    virtual size_t num_shards() const = 0;
    virtual size_t num_threads() const = 0;
    virtual U64 shard_tok_cnt(size_t s) const = 0;

    virtual void set_accel(bool on) = 0;
    virtual bool accel() const = 0;
    virtual void warm() const = 0;
    virtual void evict() const = 0;

    virtual U64 count(const void *pat, U64 len) const = 0;
    virtual void count_batch(const void *pats, const U32 *lens, U64 stride, U64 n,
                             U64 *out) const = 0;

    virtual void prob(const void *prompt, U64 len, U64 cont, U64 *prompt_cnt,
                      U64 *cont_cnt) const = 0;
    virtual void prob_batch(const void *prompts, const U32 *lens, U64 stride, const void *conts,
                            U64 n, U64 *prompt_cnts, U64 *cont_cnts) const = 0;

    virtual void ntd(const void *pat, U64 len, U64 max_support,
                     std::vector<NtdEntry> *out) const = 0;

    virtual U64 longest_suffix(const void *prompt, U64 len, U64 max_len, U64 *cnt) const = 0;
    virtual InfgramResult infgram_prob(const void *prompt, U64 len, U64 cont,
                                       U64 max_len) const = 0;
    virtual U64 infgram_ntd(const void *prompt, U64 len, U64 max_support, U64 max_len,
                            std::vector<NtdEntry> *out, U64 *prompt_cnt) const = 0;

    virtual void infgram_batch(const void *tokens, U64 rows, U64 cols, U64 max_len, double *probs,
                               U32 *suffix_lens, U64 *prompt_cnts, U64 *cont_cnts) const = 0;

    virtual void ntd_batch(const void *tokens, U64 rows, U64 cols, U64 max_ctx, U64 max_support,
                           bool infinite, std::vector<I64> *indptr, std::vector<U32> *tokens_out,
                           std::vector<U64> *counts_out) const = 0;

    // Up to `maxnum` occurrences of the pattern, sampled without replacement
    // across shards.  Returns the true total count regardless of how many were
    // sampled.
    virtual U64 sample_occurrences(const void *pat, U64 len, U64 maxnum, U64 seed,
                                   std::vector<U32> *shard_ids,
                                   std::vector<U64> *positions) const = 0;

    // Tokens around a position, clipped to the containing document.
    virtual void get_span(U32 shard, U64 pos, U64 before, U64 after, std::vector<U32> *out,
                          U64 *doc_id, U64 *offset_in_doc) const = 0;
};

template <typename T>
class TypedEngine final : public AnyEngine {
  public:
    TypedEngine(const std::vector<std::string> &dirs, size_t threads, bool load_bucket)
        : _e(dirs, threads, load_bucket) {}

    int token_width() const override { return static_cast<int>(sizeof(T)); }
    U64 tok_cnt() const override { return _e.tok_cnt(); }
    U64 doc_cnt() const override { return _e.doc_cnt(); }
    size_t num_shards() const override { return _e.num_shards(); }
    size_t num_threads() const override { return _e.num_threads(); }
    U64 shard_tok_cnt(size_t s) const override { return _e.shard(s).tok_cnt(); }

    void set_accel(bool on) override { _e.set_accel(on); }
    bool accel() const override { return _e.accel(); }
    void warm() const override { _e.warm(); }
    void evict() const override { _e.evict(); }

    U64 count(const void *pat, U64 len) const override {
        return _e.count(static_cast<const T *>(pat), len);
    }

    void count_batch(const void *pats, const U32 *lens, U64 stride, U64 n,
                     U64 *out) const override {
        _e.count_batch(static_cast<const T *>(pats), lens, stride, n, out);
    }

    void prob(const void *prompt, U64 len, U64 cont, U64 *pc, U64 *cc) const override {
        _e.prob(static_cast<const T *>(prompt), len, static_cast<T>(cont), pc, cc);
    }

    void prob_batch(const void *prompts, const U32 *lens, U64 stride, const void *conts, U64 n,
                    U64 *pc, U64 *cc) const override {
        _e.prob_batch(static_cast<const T *>(prompts), lens, stride, static_cast<const T *>(conts),
                      n, pc, cc);
    }

    void ntd(const void *pat, U64 len, U64 max_support,
             std::vector<NtdEntry> *out) const override {
        _e.ntd(static_cast<const T *>(pat), len, max_support, out);
    }

    U64 longest_suffix(const void *prompt, U64 len, U64 max_len, U64 *cnt) const override {
        std::vector<Range> r(_e.num_shards());
        return _e.longest_suffix(static_cast<const T *>(prompt), len, max_len, r.data(), cnt);
    }

    InfgramResult infgram_prob(const void *prompt, U64 len, U64 cont,
                               U64 max_len) const override {
        return _e.infgram_prob(static_cast<const T *>(prompt), len, static_cast<T>(cont), max_len);
    }

    U64 infgram_ntd(const void *prompt, U64 len, U64 max_support, U64 max_len,
                    std::vector<NtdEntry> *out, U64 *prompt_cnt) const override {
        return _e.infgram_ntd(static_cast<const T *>(prompt), len, max_support, max_len, out,
                              prompt_cnt);
    }

    void infgram_batch(const void *tokens, U64 rows, U64 cols, U64 max_len, double *probs,
                       U32 *suffix_lens, U64 *pc, U64 *cc) const override {
        _e.infgram_batch(static_cast<const T *>(tokens), rows, cols, max_len, probs, suffix_lens,
                         pc, cc);
    }

    void ntd_batch(const void *tokens, U64 rows, U64 cols, U64 max_ctx, U64 max_support,
                   bool infinite, std::vector<I64> *indptr, std::vector<U32> *tokens_out,
                   std::vector<U64> *counts_out) const override {
        std::vector<std::vector<NtdEntry>> per_pos;
        _e.ntd_batch(static_cast<const T *>(tokens), rows, cols, max_ctx, max_support, infinite,
                     &per_pos);
        indptr->resize(per_pos.size() + 1);
        (*indptr)[0] = 0;
        for (size_t i = 0; i < per_pos.size(); i++)
            (*indptr)[i + 1] = (*indptr)[i] + static_cast<I64>(per_pos[i].size());
        U64 nnz = static_cast<U64>(indptr->back());
        tokens_out->resize(nnz);
        counts_out->resize(nnz);
        U64 k = 0;
        for (const auto &v : per_pos) {
            for (const auto &e : v) {
                (*tokens_out)[k] = e.token;
                (*counts_out)[k] = e.count;
                k++;
            }
        }
    }

    U64 sample_occurrences(const void *pat, U64 len, U64 maxnum, U64 seed,
                           std::vector<U32> *shard_ids,
                           std::vector<U64> *positions) const override {
        const size_t S = _e.num_shards();
        std::vector<Range> r(S);
        _e.find(static_cast<const T *>(pat), len, nullptr, r.data());
        U64 total = Engine<T>::total(r.data(), S);
        shard_ids->clear();
        positions->clear();
        if (total == 0 || maxnum == 0) return total;

        std::mt19937_64 gen(seed);
        U64 want = maxnum < total ? maxnum : total;
        // Sample distinct global ranks, then map each back to its shard.
        std::vector<U64> picks;
        picks.reserve(want);
        if (want == total) {
            for (U64 i = 0; i < total; i++) picks.push_back(i);
        } else {
            std::vector<U64> seen;
            seen.reserve(want);
            while (picks.size() < want) {
                U64 x = std::uniform_int_distribution<U64>(0, total - 1)(gen);
                if (std::find(seen.begin(), seen.end(), x) != seen.end()) continue;
                seen.push_back(x);
                picks.push_back(x);
            }
        }
        for (U64 g : picks) {
            for (size_t s = 0; s < S; s++) {
                if (g < r[s].size()) {
                    shard_ids->push_back(static_cast<U32>(s));
                    positions->push_back(_e.shard(s).sa_at(r[s].lo + g));
                    break;
                }
                g -= r[s].size();
            }
        }
        return total;
    }

    void get_span(U32 shard, U64 pos, U64 before, U64 after, std::vector<U32> *out, U64 *doc_id,
                  U64 *offset_in_doc) const override {
        if (shard >= _e.num_shards()) throw Error("shard index out of range");
        const Shard<T> &sh = _e.shard(shard);
        if (pos >= sh.tok_cnt()) throw Error("token position out of range");
        U64 d = sh.doc_of(pos);
        U64 ds = sh.doc_start(d), de = sh.doc_end(d);
        U64 lo = pos > before ? pos - before : 0;
        if (lo < ds) lo = ds;
        U64 hi = pos + after;
        if (hi > de) hi = de;
        out->clear();
        out->reserve(hi - lo);
        const T *toks = sh.tokens();
        for (U64 i = lo; i < hi; i++) out->push_back(static_cast<U32>(toks[i]));
        if (doc_id) *doc_id = d;
        if (offset_in_doc) *offset_in_doc = pos - ds;
    }

  private:
    Engine<T> _e;
};

inline AnyEngine *open_engine(const std::vector<std::string> &dirs, int token_width,
                              size_t threads, bool load_bucket) {
    if (token_width == 2) return new TypedEngine<U16>(dirs, threads, load_bucket);
    if (token_width == 4) return new TypedEngine<U32>(dirs, threads, load_bucket);
    throw Error("unsupported token width " + std::to_string(token_width));
}

}  // namespace ngram
