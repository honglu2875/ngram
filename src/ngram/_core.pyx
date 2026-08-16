# Copyright 2024 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""Cython bindings for the suffix-array engine.

This module is deliberately thin: it converts numpy arrays to raw pointers,
releases the GIL, and calls into C++.  All of the actual logic lives in
``csrc/include``.
"""

from libc.stdint cimport int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.string cimport memcpy
from libcpp cimport bool as cbool
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np

cimport numpy as cnp

cnp.import_array()


cdef extern from "common.h" namespace "ngram" nogil:
    cdef struct NtdEntry:
        uint32_t token
        uint64_t count

    cdef struct InfgramResult:
        uint64_t suffix_len
        uint64_t prompt_cnt
        uint64_t cont_cnt
        double prob

    uint8_t ptr_size_for(uint64_t n)


cdef extern from "build.h" namespace "ngram" nogil:
    void build_sa_u16(const uint16_t *data, int64_t n, int64_t *sa, int64_t *freq,
                      int threads) except +
    void build_sa_u32(const uint32_t *data, int64_t n, int64_t alphabet, int64_t *sa,
                      int64_t *freq, int threads) except +
    void pack_sa(const int64_t *sa, uint64_t n, uint8_t ptr_size, uint8_t *out) except +
    void unigram_offsets(const int64_t *freq, uint64_t slots, uint64_t *out) except +
    void build_pivots[T](const T *data, uint64_t n, const int64_t *sa, uint64_t stride,
                         uint64_t pivot_len, T *out) except +


cdef extern from "facade.h" namespace "ngram" nogil:
    cdef cppclass AnyEngine:
        int token_width()
        uint64_t tok_cnt()
        uint64_t doc_cnt()
        size_t num_shards()
        size_t num_threads()
        uint64_t shard_tok_cnt(size_t s) except +

        void set_accel(cbool on)
        cbool accel()
        void warm()
        void evict()

        uint64_t count(const void *pat, uint64_t n) except +
        void count_batch(const void *pats, const uint32_t *lens, uint64_t stride,
                         uint64_t n, uint64_t *out) except +
        void prob(const void *prompt, uint64_t n, uint64_t cont, uint64_t *pc,
                  uint64_t *cc) except +
        void prob_batch(const void *prompts, const uint32_t *lens, uint64_t stride,
                        const void *conts, uint64_t n, uint64_t *pc, uint64_t *cc) except +
        void ntd(const void *pat, uint64_t n, uint64_t max_support,
                 vector[NtdEntry] *out) except +
        uint64_t longest_suffix(const void *prompt, uint64_t n, uint64_t max_len,
                                uint64_t *cnt) except +
        InfgramResult infgram_prob(const void *prompt, uint64_t n, uint64_t cont,
                                   uint64_t max_len) except +
        uint64_t infgram_ntd(const void *prompt, uint64_t n, uint64_t max_support,
                             uint64_t max_len, vector[NtdEntry] *out,
                             uint64_t *prompt_cnt) except +
        void infgram_batch(const void *tokens, uint64_t rows, uint64_t cols, uint64_t max_len,
                           double *probs, uint32_t *suffix_lens, uint64_t *pc,
                           uint64_t *cc) except +
        void ntd_batch(const void *tokens, uint64_t rows, uint64_t cols, uint64_t max_ctx,
                       uint64_t max_support, cbool infinite, vector[int64_t] *indptr,
                       vector[uint32_t] *tokens_out, vector[uint64_t] *counts_out) except +
        uint64_t sample_occurrences(const void *pat, uint64_t n, uint64_t maxnum,
                                    uint64_t seed, vector[uint32_t] *shard_ids,
                                    vector[uint64_t] *positions) except +
        void get_span(uint32_t shard, uint64_t pos, uint64_t before, uint64_t after,
                      vector[uint32_t] *out, uint64_t *doc_id,
                      uint64_t *offset_in_doc) except +

    AnyEngine *open_engine(const vector[string] &dirs, int token_width, size_t threads,
                           cbool load_bucket) except +


# ---------------------------------------------------------------------------
# Build-time helpers
# ---------------------------------------------------------------------------

def ptr_size(uint64_t n):
    """Bytes needed per suffix-array entry for a shard of ``n`` tokens."""
    return int(ptr_size_for(n))


def build_suffix_array(tokens, int threads=0, int alphabet=0):
    """Build the suffix array of a 1-D uint16 or uint32 token array.

    Returns ``(sa, freq)`` where ``sa`` is int64 token positions in suffix order
    and ``freq`` holds per-symbol occurrence counts.
    """
    cdef cnp.ndarray arr = np.ascontiguousarray(tokens)
    cdef int64_t n = arr.shape[0]
    if arr.ndim != 1:
        raise ValueError("tokens must be 1-D")
    if n == 0:
        raise ValueError("cannot index an empty corpus")

    cdef int64_t slots
    if arr.dtype == np.uint16:
        slots = 65536
    elif arr.dtype == np.uint32:
        slots = alphabet if alphabet > 0 else int(arr.max()) + 1
    else:
        raise TypeError("tokens must be uint16 or uint32, got %s" % arr.dtype)

    sa = np.empty(n, dtype=np.int64)
    freq = np.zeros(slots, dtype=np.int64)

    cdef int64_t[::1] sa_v = sa
    cdef int64_t[::1] fq_v = freq
    cdef const uint16_t[::1] t16
    cdef const uint32_t[::1] t32

    if arr.dtype == np.uint16:
        t16 = arr
        with nogil:
            build_sa_u16(&t16[0], n, &sa_v[0], &fq_v[0], threads)
    else:
        t32 = arr
        with nogil:
            build_sa_u32(&t32[0], n, slots, &sa_v[0], &fq_v[0], threads)
    return sa, freq


def pack_suffix_array(sa, int width, out=None):
    """Pack an int64 suffix array into ``width`` little-endian bytes per entry.

    Eight zero bytes are appended so the reader's unaligned 8-byte load stays
    inside the mapping.  Pass ``out`` (a writable uint8 buffer of exactly
    ``len(sa) * width + 8`` bytes, typically a memmap) to write the table
    straight to disk instead of holding a second copy in memory.
    """
    cdef const int64_t[::1] sa_v = np.ascontiguousarray(sa, dtype=np.int64)
    cdef uint64_t n = sa_v.shape[0]
    if out is None:
        out = np.empty(n * width + 8, dtype=np.uint8)
    elif out.dtype != np.uint8 or out.shape != (n * width + 8,):
        raise ValueError("out must be a uint8 buffer of length n * width + 8")
    cdef uint8_t[::1] out_v = out
    with nogil:
        pack_sa(&sa_v[0], n, <uint8_t>width, &out_v[0])
    return out


def unigram_table(freq):
    """Prefix-sum symbol counts into suffix-array block starts."""
    cdef const int64_t[::1] fq = np.ascontiguousarray(freq, dtype=np.int64)
    cdef uint64_t slots = fq.shape[0]
    out = np.empty(slots + 1, dtype=np.uint64)
    cdef uint64_t[::1] out_v = out
    with nogil:
        unigram_offsets(&fq[0], slots, &out_v[0])
    return out


def sample_pivots(tokens, sa, uint64_t stride, uint64_t pivot_len):
    """Gather the leading ``pivot_len`` tokens of every ``stride``-th suffix."""
    cdef cnp.ndarray arr = np.ascontiguousarray(tokens)
    cdef const int64_t[::1] sa_v = np.ascontiguousarray(sa, dtype=np.int64)
    cdef uint64_t n = arr.shape[0]
    cdef uint64_t num = (n + stride - 1) // stride
    out = np.empty(num * pivot_len, dtype=arr.dtype)

    cdef const uint16_t[::1] t16
    cdef const uint32_t[::1] t32
    cdef uint16_t[::1] o16
    cdef uint32_t[::1] o32

    if arr.dtype == np.uint16:
        t16 = arr
        o16 = out
        with nogil:
            build_pivots[uint16_t](&t16[0], n, &sa_v[0], stride, pivot_len, &o16[0])
    elif arr.dtype == np.uint32:
        t32 = arr
        o32 = out
        with nogil:
            build_pivots[uint32_t](&t32[0], n, &sa_v[0], stride, pivot_len, &o32[0])
    else:
        raise TypeError("tokens must be uint16 or uint32")
    return out.reshape(num, pivot_len)


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

cdef class Engine:
    """Low-level handle on a set of index shards.

    Prefer :class:`ngram.InfiniGram`, which adds config loading, tokenizer
    integration, and friendlier types.
    """

    cdef AnyEngine *_eng
    cdef readonly int token_width
    cdef readonly object dtype

    def __cinit__(self, shard_dirs, int token_width, size_t threads=0,
                  bint load_bucket=True):
        cdef vector[string] dirs
        for d in shard_dirs:
            dirs.push_back(<string>(str(d).encode("utf-8")))
        if dirs.size() == 0:
            raise ValueError("index has no shards")
        self._eng = open_engine(dirs, token_width, threads, load_bucket)
        self.token_width = token_width
        self.dtype = np.uint16 if token_width == 2 else np.uint32

    def __dealloc__(self):
        if self._eng != NULL:
            del self._eng
            self._eng = NULL

    # -- properties ---------------------------------------------------------

    @property
    def tok_cnt(self):
        return int(self._eng.tok_cnt())

    @property
    def doc_cnt(self):
        return int(self._eng.doc_cnt())

    @property
    def num_shards(self):
        return int(self._eng.num_shards())

    @property
    def num_threads(self):
        return int(self._eng.num_threads())

    @property
    def accel(self):
        return bool(self._eng.accel())

    def shard_tok_cnt(self, size_t s):
        return int(self._eng.shard_tok_cnt(s))

    def set_accel(self, bint on):
        """Toggle the in-RAM jump table.  Answers must not change."""
        self._eng.set_accel(on)

    def warm(self):
        """Hint the kernel to pull the whole index into page cache."""
        self._eng.warm()

    def evict(self):
        """Drop the index from page cache (used to measure cold latency)."""
        self._eng.evict()

    # -- conversion ---------------------------------------------------------

    cdef inline cnp.ndarray _as1d(self, obj):
        cdef cnp.ndarray a = np.ascontiguousarray(obj, dtype=self.dtype)
        if a.ndim != 1:
            raise ValueError("expected a 1-D token sequence")
        return a

    # -- queries ------------------------------------------------------------

    def count(self, pat):
        cdef cnp.ndarray a = self._as1d(pat)
        cdef uint64_t n = a.shape[0]
        cdef const void *p = cnp.PyArray_DATA(a)
        cdef uint64_t res
        with nogil:
            res = self._eng.count(p, n)
        return int(res)

    def count_batch(self, pats, lens=None):
        """Counts for a (batch, length) array of patterns.

        ``lens`` optionally gives a shorter valid prefix length per row.
        """
        cdef cnp.ndarray a = np.ascontiguousarray(pats, dtype=self.dtype)
        if a.ndim != 2:
            raise ValueError("pats must be 2-D")
        cdef uint64_t rows = a.shape[0]
        cdef uint64_t stride = a.shape[1]
        cdef cnp.ndarray l = (np.full(rows, stride, dtype=np.uint32) if lens is None
                              else np.ascontiguousarray(lens, dtype=np.uint32))
        if l.shape[0] != <Py_ssize_t>rows:
            raise ValueError("lens must have one entry per row")
        out = np.empty(rows, dtype=np.uint64)
        cdef uint64_t[::1] out_v = out
        cdef const uint32_t[::1] l_v = l
        cdef const void *p = cnp.PyArray_DATA(a)
        if rows == 0:
            return out
        with nogil:
            self._eng.count_batch(p, &l_v[0], stride, rows, &out_v[0])
        return out

    def prob(self, prompt, cont):
        """Returns ``(prob, prompt_count, cont_count)``."""
        cdef cnp.ndarray a = self._as1d(prompt)
        cdef uint64_t n = a.shape[0]
        cdef uint64_t c = cont
        cdef uint64_t pc = 0, cc = 0
        cdef const void *p = cnp.PyArray_DATA(a)
        with nogil:
            self._eng.prob(p, n, c, &pc, &cc)
        return ((float(cc) / pc if pc else 0.0), int(pc), int(cc))

    def prob_batch(self, prompts, conts, lens=None):
        cdef cnp.ndarray a = np.ascontiguousarray(prompts, dtype=self.dtype)
        if a.ndim != 2:
            raise ValueError("prompts must be 2-D")
        cdef uint64_t rows = a.shape[0]
        cdef uint64_t stride = a.shape[1]
        cdef cnp.ndarray cs = np.ascontiguousarray(conts, dtype=self.dtype)
        cdef cnp.ndarray l = (np.full(rows, stride, dtype=np.uint32) if lens is None
                              else np.ascontiguousarray(lens, dtype=np.uint32))
        pc = np.empty(rows, dtype=np.uint64)
        cc = np.empty(rows, dtype=np.uint64)
        cdef uint64_t[::1] pc_v = pc
        cdef uint64_t[::1] cc_v = cc
        cdef const uint32_t[::1] l_v = l
        cdef const void *p = cnp.PyArray_DATA(a)
        cdef const void *q = cnp.PyArray_DATA(cs)
        if rows == 0:
            return pc, cc
        with nogil:
            self._eng.prob_batch(p, &l_v[0], stride, q, rows, &pc_v[0], &cc_v[0])
        return pc, cc

    def ntd(self, pat, uint64_t max_support=0):
        """Next-token distribution.  Returns ``(token_ids, counts)``."""
        cdef cnp.ndarray a = self._as1d(pat)
        cdef uint64_t n = a.shape[0]
        cdef vector[NtdEntry] res
        cdef const void *p = cnp.PyArray_DATA(a)
        with nogil:
            self._eng.ntd(p, n, max_support, &res)
        return _unpack_ntd(res)

    def longest_suffix(self, prompt, uint64_t max_len=0):
        """Longest suffix of ``prompt`` occurring in the corpus.

        Returns ``(suffix_len, count)``.
        """
        cdef cnp.ndarray a = self._as1d(prompt)
        cdef uint64_t n = a.shape[0]
        cdef uint64_t cnt = 0, l
        cdef const void *p = cnp.PyArray_DATA(a)
        with nogil:
            l = self._eng.longest_suffix(p, n, max_len, &cnt)
        return int(l), int(cnt)

    def infgram_prob(self, prompt, cont, uint64_t max_len=0):
        """Returns ``(prob, suffix_len, prompt_count, cont_count)``."""
        cdef cnp.ndarray a = self._as1d(prompt)
        cdef uint64_t n = a.shape[0]
        cdef uint64_t c = cont
        cdef InfgramResult r
        cdef const void *p = cnp.PyArray_DATA(a)
        with nogil:
            r = self._eng.infgram_prob(p, n, c, max_len)
        return r.prob, int(r.suffix_len), int(r.prompt_cnt), int(r.cont_cnt)

    def infgram_ntd(self, prompt, uint64_t max_support=0, uint64_t max_len=0):
        """Returns ``(token_ids, counts, suffix_len, prompt_count)``."""
        cdef cnp.ndarray a = self._as1d(prompt)
        cdef uint64_t n = a.shape[0]
        cdef vector[NtdEntry] res
        cdef uint64_t pc = 0, l
        cdef const void *p = cnp.PyArray_DATA(a)
        with nogil:
            l = self._eng.infgram_ntd(p, n, max_support, max_len, &res, &pc)
        ids, counts = _unpack_ntd(res)
        return ids, counts, int(l), int(pc)

    def infgram_batch(self, tokens, uint64_t max_len=0):
        """Infinite-gram probability of every observed token in a 2-D batch.

        Returns ``(probs, suffix_lens, prompt_counts, cont_counts)``, each
        shaped like ``tokens``.  Position 0 of each row has an empty context.
        """
        cdef cnp.ndarray a = np.ascontiguousarray(tokens, dtype=self.dtype)
        if a.ndim != 2:
            raise ValueError("tokens must be 2-D")
        cdef uint64_t rows = a.shape[0]
        cdef uint64_t cols = a.shape[1]
        probs = np.zeros((rows, cols), dtype=np.float64)
        slens = np.zeros((rows, cols), dtype=np.uint32)
        pc = np.zeros((rows, cols), dtype=np.uint64)
        cc = np.zeros((rows, cols), dtype=np.uint64)
        cdef double[:, ::1] pr_v = probs
        cdef uint32_t[:, ::1] sl_v = slens
        cdef uint64_t[:, ::1] pc_v = pc
        cdef uint64_t[:, ::1] cc_v = cc
        cdef const void *p = cnp.PyArray_DATA(a)
        if rows == 0 or cols == 0:
            return probs, slens, pc, cc
        with nogil:
            self._eng.infgram_batch(p, rows, cols, max_len, &pr_v[0, 0], &sl_v[0, 0],
                                    &pc_v[0, 0], &cc_v[0, 0])
        return probs, slens, pc, cc

    def ntd_batch(self, tokens, uint64_t max_ctx=0, uint64_t max_support=0,
                  bint infinite=False):
        """Next-token distributions at every position of a 2-D batch.

        Returns ``(indptr, token_ids, counts)`` in CSR layout over
        ``rows * cols`` positions -- the distribution for flat position ``i`` is
        ``token_ids[indptr[i]:indptr[i+1]]`` with matching counts.
        """
        cdef cnp.ndarray a = np.ascontiguousarray(tokens, dtype=self.dtype)
        if a.ndim != 2:
            raise ValueError("tokens must be 2-D")
        cdef uint64_t rows = a.shape[0]
        cdef uint64_t cols = a.shape[1]
        cdef vector[int64_t] indptr
        cdef vector[uint32_t] ids
        cdef vector[uint64_t] counts
        cdef const void *p = cnp.PyArray_DATA(a)
        if rows == 0 or cols == 0:
            return (np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.uint32),
                    np.zeros(0, dtype=np.uint64))
        with nogil:
            self._eng.ntd_batch(p, rows, cols, max_ctx, max_support, infinite,
                                &indptr, &ids, &counts)
        return (_to_np_i64(indptr), _to_np_u32(ids), _to_np_u64(counts))

    def sample_occurrences(self, pat, uint64_t maxnum=10, uint64_t seed=0):
        """Returns ``(total_count, shard_ids, positions)``."""
        cdef cnp.ndarray a = self._as1d(pat)
        cdef uint64_t n = a.shape[0]
        cdef vector[uint32_t] shards
        cdef vector[uint64_t] positions
        cdef uint64_t total
        cdef const void *p = cnp.PyArray_DATA(a)
        with nogil:
            total = self._eng.sample_occurrences(p, n, maxnum, seed, &shards, &positions)
        return int(total), _to_np_u32(shards), _to_np_u64(positions)

    def get_span(self, uint32_t shard, uint64_t pos, uint64_t before=64, uint64_t after=64):
        """Tokens around a position, clipped to its document.

        Returns ``(tokens, doc_id, offset_in_doc)``.
        """
        cdef vector[uint32_t] out
        cdef uint64_t doc_id = 0, off = 0
        with nogil:
            self._eng.get_span(shard, pos, before, after, &out, &doc_id, &off)
        return _to_np_u32(out), int(doc_id), int(off)


# ---------------------------------------------------------------------------
# vector -> ndarray helpers
# ---------------------------------------------------------------------------

cdef _unpack_ntd(const vector[NtdEntry] &res):
    cdef Py_ssize_t n = res.size()
    ids = np.empty(n, dtype=np.uint32)
    counts = np.empty(n, dtype=np.uint64)
    cdef uint32_t[::1] iv = ids
    cdef uint64_t[::1] cv = counts
    cdef Py_ssize_t i
    for i in range(n):
        iv[i] = res[i].token
        cv[i] = res[i].count
    return ids, counts


# std::vector storage is contiguous, so these are single memcpys rather than
# element-by-element loops.  ntd_batch can return millions of entries.

cdef _to_np_u32(const vector[uint32_t] &v):
    cdef Py_ssize_t n = v.size()
    out = np.empty(n, dtype=np.uint32)
    cdef uint32_t[::1] o = out
    if n:
        memcpy(&o[0], v.const_data(), n * sizeof(uint32_t))
    return out


cdef _to_np_u64(const vector[uint64_t] &v):
    cdef Py_ssize_t n = v.size()
    out = np.empty(n, dtype=np.uint64)
    cdef uint64_t[::1] o = out
    if n:
        memcpy(&o[0], v.const_data(), n * sizeof(uint64_t))
    return out


cdef _to_np_i64(const vector[int64_t] &v):
    cdef Py_ssize_t n = v.size()
    out = np.empty(n, dtype=np.int64)
    cdef int64_t[::1] o = out
    if n:
        memcpy(&o[0], v.const_data(), n * sizeof(int64_t))
    return out
