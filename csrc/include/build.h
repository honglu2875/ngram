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

#include <cstring>
#include <vector>

#include "common.h"
#include "libsais16x64.h"
#include "libsais64.h"

namespace ngram {

// Suffix array construction.
//
// libsais works on the token array directly -- there is no need to flatten
// tokens to bytes and filter aligned positions, which is what the reference
// infini-gram indexer does.  The resulting order is true numeric token order,
// which is what makes the unigram jump table a plain index by token id.
//
// Peak memory is 8 bytes per token for SA plus the input itself.

// Build the suffix array of a uint16 token array.  `freq` receives the 65536
// symbol counts, which the caller turns into the unigram jump table for free.
inline void build_sa_u16(const U16 *data, I64 n, I64 *sa, I64 *freq, int threads) {
    I64 rc;
#if defined(LIBSAIS_OPENMP)
    rc = libsais16x64_omp(data, sa, n, 0, freq, threads);
#else
    (void)threads;
    rc = libsais16x64(data, sa, n, 0, freq);
#endif
    if (rc != 0) throw Error("libsais16x64 failed with code " + std::to_string(rc));
}

// Build the suffix array of a uint32 token array.  libsais needs an int64 copy
// of the input for integer alphabets, so this path costs 16 bytes per token
// rather than 10; it exists for tokenizers whose vocabulary exceeds 16 bits.
inline void build_sa_u32(const U32 *data, I64 n, I64 alphabet, I64 *sa, I64 *freq, int threads) {
    std::vector<I64> tmp(static_cast<size_t>(n) + 1);
    for (I64 i = 0; i < n; i++) tmp[i] = static_cast<I64>(data[i]);
    if (freq) {
        std::memset(freq, 0, sizeof(I64) * static_cast<size_t>(alphabet));
        for (I64 i = 0; i < n; i++) freq[data[i]]++;
    }
    I64 rc;
#if defined(LIBSAIS_OPENMP)
    rc = libsais64_long_omp(tmp.data(), sa, n, alphabet, 0, threads);
#else
    (void)threads;
    rc = libsais64_long(tmp.data(), sa, n, alphabet, 0);
#endif
    if (rc != 0) throw Error("libsais64_long failed with code " + std::to_string(rc));
}

// Pack int64 suffix array entries into `ptr_size` little-endian bytes each,
// appending Shard::kPtrPad zero bytes so the reader's 8-byte unaligned load
// never runs past the mapping.  `out` must hold n * ptr_size + 8 bytes.
inline void pack_sa(const I64 *sa, U64 n, U8 ptr_size, U8 *out) {
    if (ptr_size == 8) {
        std::memcpy(out, sa, n * 8);
    } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (I64 i = 0; i < static_cast<I64>(n); i++) {
            U64 v = static_cast<U64>(sa[i]);
            std::memcpy(out + static_cast<U64>(i) * ptr_size, &v, ptr_size);
        }
    }
    std::memset(out + n * ptr_size, 0, 8);
}

// Gather the first `pivot_len` tokens of every `stride`-th suffix.
//
// Suffixes shorter than pivot_len are padded with token 0.  That keeps the key
// array in exactly the same order as the suffix array: a padded key can never
// compare greater than the true suffix would (0 is the minimum symbol), so
// pruning against it can only ever be conservative, never wrong.
template <typename T>
void build_pivots(const T *data, U64 n, const I64 *sa, U64 stride, U64 pivot_len, T *out) {
    U64 num = (n + stride - 1) / stride;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (I64 k = 0; k < static_cast<I64>(num); k++) {
        U64 p = static_cast<U64>(sa[static_cast<U64>(k) * stride]);
        T *dst = out + static_cast<U64>(k) * pivot_len;
        U64 avail = n - p;
        U64 m = pivot_len < avail ? pivot_len : avail;
        for (U64 j = 0; j < m; j++) dst[j] = data[p + j];
        for (U64 j = m; j < pivot_len; j++) dst[j] = 0;
    }
}

// Turn libsais' symbol counts into starting ranks: uni[t] is the rank of the
// first suffix beginning with token t, so [uni[t], uni[t+1]) is that token's
// block.  `out` must hold slots + 1 entries.
inline void unigram_offsets(const I64 *freq, U64 slots, U64 *out) {
    U64 acc = 0;
    for (U64 t = 0; t < slots; t++) {
        out[t] = acc;
        acc += static_cast<U64>(freq[t]);
    }
    out[slots] = acc;
}

}  // namespace ngram
