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

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ngram {

using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;
using I64 = int64_t;

// A half-open interval of suffix-array ranks: the suffixes SA[lo], ..., SA[hi-1]
// are exactly those sharing some common prefix.  `hi - lo` is the occurrence
// count of that prefix in the shard.
struct Range {
    U64 lo;
    U64 hi;

    U64 size() const { return hi > lo ? hi - lo : 0; }
    bool empty() const { return hi <= lo; }
};

// One entry of a next-token distribution.
struct NtdEntry {
    U32 token;
    U64 count;
};

// Result of an infinite-gram query: `suffix_len` is the length of the longest
// suffix of the prompt that occurs in the corpus at all.
struct InfgramResult {
    U64 suffix_len;
    U64 prompt_cnt;
    U64 cont_cnt;
    double prob;
};

class Error : public std::runtime_error {
  public:
    explicit Error(const std::string &what) : std::runtime_error(what) {}
};

// Number of bytes needed to address `n` distinct values.  The suffix array
// stores token indices in [0, n), so 4 bytes suffice up to 4.29B tokens per
// shard and 5 bytes up to 1.1T.
inline U8 ptr_size_for(U64 n) {
    U8 w = 1;
    while (w < 8 && (n >> (8 * w)) != 0) w++;
    return w;
}

}  // namespace ngram
