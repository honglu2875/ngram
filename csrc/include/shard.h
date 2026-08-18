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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>
#include <string>
#include <vector>

#include "common.h"

namespace ngram {

// Magic and layout of the `bucket` acceleration file.  See docs/format.md.
static constexpr char kBucketMagic[8] = {'N', 'G', 'R', 'A', 'M', 'B', 'K', '1'};
static constexpr U64 kBucketHeaderBytes = 64;

// A read-only memory map that unmaps itself.
class Mapping {
  public:
    Mapping() = default;
    ~Mapping() { reset(); }

    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
    Mapping(Mapping &&o) noexcept { *this = std::move(o); }
    Mapping &operator=(Mapping &&o) noexcept {
        if (this != &o) {
            reset();
            _addr = o._addr;
            _size = o._size;
            _path = std::move(o._path);
            o._addr = nullptr;
            o._size = 0;
        }
        return *this;
    }

    void open(const std::string &path, bool random_access) {
        reset();
        _path = path;
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw Error("cannot open " + path + ": " + std::strerror(errno));
        struct stat st;
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw Error("cannot stat " + path);
        }
        _size = static_cast<U64>(st.st_size);
        if (_size == 0) {
            ::close(fd);
            throw Error("empty file " + path);
        }
        void *p = ::mmap(nullptr, _size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) throw Error("cannot mmap " + path + ": " + std::strerror(errno));
        _addr = static_cast<U8 *>(p);
        // Binary search touches one scattered cache line per probe.  Default
        // readahead would fault in 128 KiB to serve a 5-byte read, so tell the
        // kernel not to bother.
        if (random_access) ::madvise(_addr, _size, MADV_RANDOM);
    }

    void reset() {
        if (_addr) ::munmap(_addr, _size);
        _addr = nullptr;
        _size = 0;
    }

    // Pull the whole mapping into page cache and *wait* for it.
    //
    // MADV_WILLNEED only queues readahead, so a caller that measures right
    // after it would still be timing disk reads.  Touching one byte per page
    // afterwards makes the call synchronous, which is the behaviour anyone
    // asking to "warm" an index actually wants.
    void warm() const {
        if (!_addr) return;
        ::madvise(_addr, _size, MADV_WILLNEED);
        const U64 page = 4096;
        volatile U64 sink = 0;
        for (U64 off = 0; off < _size; off += page) sink += _addr[off];
        (void)sink;
    }

    // Genuinely drop this file from page cache, so the next probe costs a real
    // disk read.  MADV_DONTNEED alone only tears down this process's page-table
    // entries -- for a MAP_SHARED file mapping the pages stay cached, and a
    // "cold" benchmark built on it would measure minor faults.  Dropping the
    // PTEs first is what lets the subsequent fadvise actually reclaim them.
    void evict() const {
        if (!_addr) return;
        ::madvise(_addr, _size, MADV_DONTNEED);
        int fd = ::open(_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }

    const U8 *data() const { return _addr; }
    U64 size() const { return _size; }

  private:
    U8 *_addr = nullptr;
    U64 _size = 0;
    std::string _path;
};

// One independently-built piece of the corpus: a token array, its suffix array,
// a document-start table, and an in-RAM search accelerator.
//
// Suffix array entries are *token* indices in [0, tok_cnt), packed little-endian
// into `ptr_size` bytes each.  Token indices need one byte fewer than byte
// offsets at every scale that matters (4 bytes up to 4.29B tokens, 5 up to
// 1.1T), and they drop a multiply from the innermost loop.
template <typename T>
class Shard {
  public:
    void open(const std::string &dir, bool load_bucket = true) {
        _ds_map.open(dir + "/tokenized", /*random_access=*/true);
        _sa_map.open(dir + "/table", /*random_access=*/true);
        _od_map.open(dir + "/offset", /*random_access=*/true);

        _ds = reinterpret_cast<const T *>(_ds_map.data());
        _tok_cnt = _ds_map.size() / sizeof(T);
        _od = reinterpret_cast<const U64 *>(_od_map.data());
        _doc_cnt = _od_map.size() / sizeof(U64);

        // `table` carries kPtrPad trailing bytes so that the 8-byte unaligned
        // load in sa_at() is always inside the mapping.
        if (_sa_map.size() < kPtrPad || _tok_cnt == 0)
            throw Error("corrupt shard at " + dir + ": table too small");
        U64 payload = _sa_map.size() - kPtrPad;
        if (payload % _tok_cnt != 0)
            throw Error("corrupt shard at " + dir + ": table size not a multiple of token count");
        _ptr_size = static_cast<U8>(payload / _tok_cnt);
        if (_ptr_size < 1 || _ptr_size > 8)
            throw Error("corrupt shard at " + dir + ": implausible pointer size");
        _ptr_mask = (_ptr_size >= 8) ? ~U64(0) : ((U64(1) << (8 * _ptr_size)) - 1);
        _sa = _sa_map.data();

        if (load_bucket) _load_bucket(dir + "/bucket");
    }

    U64 tok_cnt() const { return _tok_cnt; }
    U64 doc_cnt() const { return _doc_cnt; }
    U8 ptr_size() const { return _ptr_size; }
    const T *tokens() const { return _ds; }
    bool has_bucket() const { return _has_bucket; }

    // Disable the accelerator without rebuilding; used to prove that it does
    // not change any answer.
    void set_accel(bool on) { _accel = on && _has_bucket; }
    bool accel() const { return _accel; }

    void warm() const {
        _ds_map.warm();
        _sa_map.warm();
        _od_map.warm();
    }
    void evict() const {
        _ds_map.evict();
        _sa_map.evict();
        _od_map.evict();
    }

    // ---- primitive accessors -------------------------------------------------

    // Token index of the i-th smallest suffix.
    inline U64 sa_at(U64 i) const {
        U64 v;
        std::memcpy(&v, _sa + i * _ptr_size, sizeof(U64));
        return v & _ptr_mask;
    }

    inline void prefetch_rank(U64 i) const {
        __builtin_prefetch(_sa + i * _ptr_size, 0, 1);
    }

    // Compare the suffix starting at token position p against pat[0..len).
    // Returns <0, 0, >0.  A suffix that runs out before the pattern ends counts
    // as smaller, matching the suffix-array convention; a suffix that merely
    // *starts with* the pattern compares equal, which is what makes
    // [lower_bound, upper_bound) the full set of occurrences.
    inline int cmp_suffix(U64 p, const T *pat, U64 len) const {
        U64 avail = _tok_cnt - p;
        U64 m = len < avail ? len : avail;
        const T *s = _ds + p;
        for (U64 i = 0; i < m; i++) {
            if (s[i] != pat[i]) return s[i] < pat[i] ? -1 : 1;
        }
        return (m < len) ? -1 : 0;
    }

    // ---- search --------------------------------------------------------------

    // Rank range of all suffixes having pat[0..len) as a prefix.  `hint` bounds
    // the search; pass full_range() when nothing is known.  Passing the range of
    // pat[0..len-1] turns extending an n-gram into a search over just that
    // range, which is where most of the speed comes from.
    Range find(const T *pat, U64 len, Range hint) const {
        if (len == 0) return full_range();

        U64 lo = hint.lo, hi = hint.hi;
        if (hi > _tok_cnt) hi = _tok_cnt;
        if (lo > hi) return Range{lo, lo};

        if (_accel) {
            // Level 1: jump straight to the block of suffixes starting with the
            // pattern's first token.  Replaces ~2*log2(V) cold probes.
            if (lo == 0 && hi == _tok_cnt) {
                U64 t = static_cast<U64>(pat[0]);
                if (t + 1 < _uni.size()) {
                    lo = _uni[t];
                    hi = _uni[t + 1];
                    if (lo >= hi) return Range{lo, lo};
                }
            }
            // Level 2: binary search the RAM-resident sampled prefixes to reach
            // a window of at most one stride before touching the mapped table.
            if (!_pivots.empty() && hi - lo > 2 * _pivot_stride) {
                _narrow_by_pivots(pat, len, lo, hi);
                if (lo >= hi) return Range{lo, lo};
            }
        }

        U64 first = _lower_bound(pat, len, lo, hi);
        if (first >= hi) return Range{first, first};
        U64 last = _upper_bound(pat, len, first, hi);
        return Range{first, last};
    }

    Range full_range() const { return Range{0, _tok_cnt}; }

    // Token following the occurrence at rank `i` of a pattern of length `len`.
    // Returns false when the occurrence sits at the very end of the shard.
    inline bool next_token(U64 i, U64 len, T *out) const {
        U64 p = sa_at(i) + len;
        if (p >= _tok_cnt) return false;
        *out = _ds[p];
        return true;
    }

    // ---- documents -----------------------------------------------------------

    // Index of the document containing token position p.
    U64 doc_of(U64 p) const {
        if (_doc_cnt == 0) return 0;
        U64 lo = 0, hi = _doc_cnt;
        while (lo < hi) {
            U64 mid = lo + (hi - lo) / 2;
            if (_od[mid] <= p) lo = mid + 1; else hi = mid;
        }
        return lo == 0 ? 0 : lo - 1;
    }

    U64 doc_start(U64 d) const { return d < _doc_cnt ? _od[d] : 0; }
    U64 doc_end(U64 d) const { return d + 1 < _doc_cnt ? _od[d + 1] : _tok_cnt; }

    static constexpr U64 kPtrPad = 8;

  private:
    void _load_bucket(const std::string &path) {
        Mapping m;
        try {
            m.open(path, /*random_access=*/false);
        } catch (const Error &) {
            _has_bucket = false;
            _accel = false;
            return;
        }
        const U8 *p = m.data();
        if (m.size() < kBucketHeaderBytes || std::memcmp(p, kBucketMagic, 8) != 0)
            throw Error("corrupt bucket file " + path);

        U32 token_width, vocab_slots, pivot_len;
        U64 tok_cnt, pivot_stride, num_pivots;
        std::memcpy(&token_width, p + 8, 4);
        std::memcpy(&vocab_slots, p + 12, 4);
        std::memcpy(&pivot_len, p + 16, 4);
        std::memcpy(&tok_cnt, p + 24, 8);
        std::memcpy(&pivot_stride, p + 32, 8);
        std::memcpy(&num_pivots, p + 40, 8);

        if (token_width != sizeof(T))
            throw Error("bucket file " + path + " has a different token width");
        if (tok_cnt != _tok_cnt)
            throw Error("bucket file " + path + " disagrees with the shard token count");

        U64 uni_bytes = (U64(vocab_slots) + 1) * sizeof(U64);
        U64 piv_bytes = num_pivots * pivot_len * sizeof(T);
        if (m.size() < kBucketHeaderBytes + uni_bytes + piv_bytes)
            throw Error("truncated bucket file " + path);

        // Copied out of the mapping deliberately: these are the hot structures
        // and we want them in anonymous RAM, not competing with page cache.
        _uni.resize(U64(vocab_slots) + 1);
        std::memcpy(_uni.data(), p + kBucketHeaderBytes, uni_bytes);
        _pivots.resize(num_pivots * pivot_len);
        if (piv_bytes) std::memcpy(_pivots.data(), p + kBucketHeaderBytes + uni_bytes, piv_bytes);

        _pivot_len = pivot_len;
        _pivot_stride = pivot_stride;
        _num_pivots = num_pivots;
        _has_bucket = true;
        _accel = true;
    }

    // Compare the sampled prefix of pivot `k` against the pattern.  Returns 0
    // when the comparison cannot be resolved from the sample alone, which is
    // safe: it only means we decline to prune there.
    inline int _cmp_pivot(U64 k, const T *pat, U64 len) const {
        const T *key = _pivots.data() + k * _pivot_len;
        U64 m = len < _pivot_len ? len : _pivot_len;
        for (U64 i = 0; i < m; i++) {
            if (key[i] != pat[i]) return key[i] < pat[i] ? -1 : 1;
        }
        return 0;
    }

    // Shrink [lo, hi) using the in-RAM pivots.  Pivot keys are sorted, so
    // _cmp_pivot is monotone in k and ordinary binary search applies.
    void _narrow_by_pivots(const T *pat, U64 len, U64 &lo, U64 &hi) const {
        U64 a = (lo + _pivot_stride - 1) / _pivot_stride;  // first pivot at rank >= lo
        U64 b = (hi - 1) / _pivot_stride;                  // last pivot at rank <= hi-1
        if (b >= _num_pivots) b = _num_pivots - 1;
        if (a > b) return;

        // First pivot that is not strictly below the pattern.
        U64 l = a, r = b + 1;
        while (l < r) {
            U64 mid = l + (r - l) / 2;
            if (_cmp_pivot(mid, pat, len) < 0) l = mid + 1; else r = mid;
        }
        const U64 pa = l;
        // First pivot that is strictly above the pattern.
        l = pa;
        r = b + 1;
        while (l < r) {
            U64 mid = l + (r - l) / 2;
            if (_cmp_pivot(mid, pat, len) <= 0) l = mid + 1; else r = mid;
        }
        const U64 pb = l;

        if (pa > a) {
            U64 cand = (pa - 1) * _pivot_stride;
            if (cand > lo) lo = cand;
        }
        if (pb <= b) {
            U64 cand = pb * _pivot_stride;
            if (cand < hi) hi = cand;
        }
    }

    inline U64 _lower_bound(const T *pat, U64 len, U64 lo, U64 hi) const {
        while (lo < hi) {
            U64 mid = lo + (hi - lo) / 2;
            // Speculatively pull in both possible next probes.  On a resident
            // index this converts a serial dependent-load chain into an
            // overlapped one.
            prefetch_rank(lo + (mid - lo) / 2);
            prefetch_rank(mid + 1 + (hi - mid - 1) / 2);
            if (cmp_suffix(sa_at(mid), pat, len) < 0) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    inline U64 _upper_bound(const T *pat, U64 len, U64 lo, U64 hi) const {
        while (lo < hi) {
            U64 mid = lo + (hi - lo) / 2;
            prefetch_rank(lo + (mid - lo) / 2);
            prefetch_rank(mid + 1 + (hi - mid - 1) / 2);
            if (cmp_suffix(sa_at(mid), pat, len) <= 0) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    Mapping _ds_map, _sa_map, _od_map;
    const T *_ds = nullptr;
    const U8 *_sa = nullptr;
    const U64 *_od = nullptr;
    U64 _tok_cnt = 0;
    U64 _doc_cnt = 0;
    U8 _ptr_size = 0;
    U64 _ptr_mask = 0;

    std::vector<U64> _uni;
    std::vector<T> _pivots;
    U64 _pivot_len = 0;
    U64 _pivot_stride = 0;
    U64 _num_pivots = 0;
    bool _has_bucket = false;
    bool _accel = false;
};

}  // namespace ngram
