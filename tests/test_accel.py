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

"""The in-RAM jump table must speed queries up without changing any answer.

A tiny corpus never exercises pivot pruning, so these tests deliberately use a
corpus large enough (and a vocabulary small enough) that every search range is
many strides wide.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ngram import InfiniGram
from ngram.format import read_bucket

from .conftest import Oracle, build_from_array, make_corpus

CORPUS_TOKENS = 300_000
VOCAB = 6
STRIDE = 512
PIVOT_LEN = 8


@pytest.fixture(scope="module")
def dense_corpus():
    # Six symbols over 300k tokens: every 1-token block is ~50k ranks wide, so
    # pruning has real work to do at every level.
    return make_corpus(seed=3, n=CORPUS_TOKENS, vocab=VOCAB, sep=None)


@pytest.fixture(scope="module")
def dense_index(tmp_path_factory, dense_corpus):
    d = str(tmp_path_factory.mktemp("dense"))
    path = build_from_array(
        d,
        dense_corpus,
        sep=None,
        shard_tokens=10**9,
        pivot_stride=STRIDE,
        pivot_len=PIVOT_LEN,
    )
    return InfiniGram(path)


def test_pivot_table_is_written_and_sorted(dense_index, dense_corpus):
    d = dense_index.config.shard_dir(dense_index.path, 0)
    b = read_bucket(os.path.join(d, "bucket"))
    pivots = b["pivots"]
    assert b["pivot_stride"] == STRIDE
    assert pivots.shape == ((CORPUS_TOKENS + STRIDE - 1) // STRIDE, PIVOT_LEN)
    # Keys must be non-decreasing, or binary search over them is invalid.
    for i in range(pivots.shape[0] - 1):
        a, c = pivots[i], pivots[i + 1]
        diff = np.flatnonzero(a != c)
        if diff.size:
            assert a[diff[0]] < c[diff[0]], i


def test_pivots_match_the_suffixes_they_sample(dense_index, dense_corpus):
    """Each pivot key must be the true prefix of the suffix at its rank."""
    d = dense_index.config.shard_dir(dense_index.path, 0)
    b = read_bucket(os.path.join(d, "bucket"))
    tokens = np.fromfile(os.path.join(d, "tokenized"), dtype=np.uint16)
    n = tokens.shape[0]
    width = dense_index.config.shards[0]["ptr_size"]
    raw = np.fromfile(os.path.join(d, "table"), dtype=np.uint8)
    padded = np.zeros((n, 8), dtype=np.uint8)
    padded[:, :width] = raw[: n * width].reshape(n, width)
    sa = padded.view(np.uint64).reshape(n)

    rng = np.random.default_rng(5)
    for k in rng.integers(0, b["pivots"].shape[0], size=200):
        rank = int(k) * STRIDE
        p = int(sa[rank])
        expected = np.zeros(PIVOT_LEN, dtype=np.uint16)
        avail = min(PIVOT_LEN, n - p)
        expected[:avail] = tokens[p : p + avail]  # short suffixes zero-pad
        assert np.array_equal(b["pivots"][int(k)], expected), k


def test_answers_identical_with_and_without_accel(dense_index, dense_corpus):
    rng = np.random.default_rng(6)
    pats = []
    for _ in range(400):
        L = int(rng.integers(1, 14))
        if rng.random() < 0.8:
            s = int(rng.integers(0, CORPUS_TOKENS - L))
            pats.append(np.asarray(dense_corpus[s : s + L]))
        else:
            pats.append(rng.integers(0, VOCAB, size=L, dtype=np.uint16))

    fast = [dense_index.count(p) for p in pats]
    dense_index.set_accel(False)
    try:
        slow = [dense_index.count(p) for p in pats]
    finally:
        dense_index.set_accel(True)
    assert fast == slow
    # The pruning must actually be reachable: long patterns beyond PIVOT_LEN
    # exercise the ambiguous-comparison branch too.
    assert max(len(p) for p in pats) > PIVOT_LEN


def test_counts_still_match_brute_force_at_scale(dense_index, dense_corpus):
    """Independent check that pruning has not silently lost occurrences."""
    rng = np.random.default_rng(8)
    for _ in range(60):
        L = int(rng.integers(1, 10))
        s = int(rng.integers(0, CORPUS_TOKENS - L))
        pat = np.asarray(dense_corpus[s : s + L])
        win = np.lib.stride_tricks.sliding_window_view(dense_corpus, L)
        expected = int((win == pat).all(axis=1).sum())
        assert dense_index.count(pat) == expected, pat


def test_ntd_matches_brute_force_at_scale(dense_index, dense_corpus):
    rng = np.random.default_rng(9)
    for _ in range(15):
        L = int(rng.integers(1, 6))
        s = int(rng.integers(0, CORPUS_TOKENS - L - 1))
        pat = np.asarray(dense_corpus[s : s + L])
        win = np.lib.stride_tricks.sliding_window_view(dense_corpus, L)
        hits = np.flatnonzero((win == pat).all(axis=1))
        hits = hits[hits + L < CORPUS_TOKENS]
        nxt = dense_corpus[hits + L]
        vals, counts = np.unique(nxt, return_counts=True)
        res = dense_index.ntd(pat)
        assert list(res.token_ids) == [int(v) for v in vals]
        assert list(res.counts) == [int(c) for c in counts]


def test_long_repeated_pattern_is_found(dense_index, dense_corpus):
    """Patterns longer than the sampled pivot prefix still resolve exactly."""
    s = 12345
    for L in (PIVOT_LEN, PIVOT_LEN + 1, PIVOT_LEN * 3, 40):
        pat = np.asarray(dense_corpus[s : s + L])
        assert dense_index.count(pat) >= 1
        win = np.lib.stride_tricks.sliding_window_view(dense_corpus, L)
        assert dense_index.count(pat) == int((win == pat).all(axis=1).sum())


def test_cold_cache_gives_same_answers(dense_index, dense_corpus):
    """Evicting the mapping must change latency, never results."""
    pats = [np.asarray(dense_corpus[i : i + 5]) for i in range(0, 5000, 500)]
    warm = [dense_index.count(p) for p in pats]
    dense_index.evict()
    cold = [dense_index.count(p) for p in pats]
    dense_index.warm()
    assert warm == cold
