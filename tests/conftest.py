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

"""Shared fixtures: synthetic corpora plus a brute-force reference model.

The reference is deliberately dumb -- it enumerates every n-gram with a
``Counter`` -- so that any disagreement points at the index, not at a second
clever implementation making the same mistake.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import numpy as np
import pytest

from ngram import InfiniGram, build_index


class Oracle:
    """Ground truth over a token array, computed the obvious slow way."""

    def __init__(self, tokens: np.ndarray, sep: int | None):
        self.tokens = np.asarray(tokens)
        self.sep = sep
        self.n = int(self.tokens.shape[0])
        self._counts: dict[int, Counter] = {}
        self._next: dict[int, defaultdict] = {}

    def _index_len(self, L: int) -> None:
        if L in self._counts:
            return
        counts: Counter = Counter()
        nxt: defaultdict = defaultdict(Counter)
        toks = self.tokens
        for i in range(self.n - L + 1):
            key = tuple(int(x) for x in toks[i : i + L])
            counts[key] += 1
            if i + L < self.n:
                nxt[key][int(toks[i + L])] += 1
        self._counts[L] = counts
        self._next[L] = nxt

    def count(self, pat) -> int:
        pat = tuple(int(x) for x in np.asarray(pat).reshape(-1))
        if len(pat) == 0:
            return self.n
        self._index_len(len(pat))
        return int(self._counts[len(pat)].get(pat, 0))

    def ntd(self, pat) -> dict:
        pat = tuple(int(x) for x in np.asarray(pat).reshape(-1))
        if len(pat) == 0:
            # Empty context: every position except the last has a successor.
            return dict(Counter(int(t) for t in self.tokens[1:]))
        self._index_len(len(pat))
        return dict(self._next[len(pat)].get(pat, {}))

    def longest_suffix(self, prompt, max_len: int = 0) -> int:
        prompt = [int(x) for x in np.asarray(prompt).reshape(-1)]
        upper = len(prompt)
        if max_len:
            upper = min(upper, max_len)
        for L in range(upper, 0, -1):
            if self.count(prompt[len(prompt) - L :]) > 0:
                return L
        return 0

    def infgram_prob(self, prompt, cont, max_len: int = 0):
        L = self.longest_suffix(prompt, max_len)
        prompt = [int(x) for x in np.asarray(prompt).reshape(-1)]
        suffix = prompt[len(prompt) - L :] if L else []
        pc = self.count(suffix)
        cc = self.count(list(suffix) + [int(cont)])
        return (cc / pc if pc else 0.0), L, pc, cc


def make_corpus(seed: int, n: int, vocab: int, sep: int | None, n_docs: int = 0):
    """Random tokens with optional separators sprinkled at document starts."""
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, vocab, size=n, dtype=np.uint16)
    if sep is not None:
        k = n_docs or max(1, n // 200)
        pos = np.unique(rng.choice(n, size=k, replace=False))
        toks[pos] = sep
        toks[0] = sep
    return toks


def build_from_array(tmpdir, tokens, *, sep, shard_tokens, dtype="u16", **kw):
    """Write a raw token file and index it."""
    raw = os.path.join(tmpdir, "corpus.bin")
    np.ascontiguousarray(tokens).tofile(raw)
    out = os.path.join(tmpdir, "idx")
    build_index(
        [raw],
        out,
        fmt="raw",
        token_dtype=dtype,
        doc_sep_token=sep,
        shard_tokens=shard_tokens,
        verbose=False,
        **kw,
    )
    return out


@pytest.fixture(scope="session")
def small_corpus():
    """Small vocabulary so that repeats -- and thus long matches -- are common."""
    return make_corpus(seed=7, n=30_000, vocab=24, sep=999)


@pytest.fixture(scope="session")
def small_oracle(small_corpus):
    return Oracle(small_corpus, sep=999)


@pytest.fixture(scope="session")
def small_index(tmp_path_factory, small_corpus):
    d = str(tmp_path_factory.mktemp("small"))
    path = build_from_array(d, small_corpus, sep=999, shard_tokens=10**9)
    return InfiniGram(path)


@pytest.fixture(scope="session")
def sharded_index(tmp_path_factory, small_corpus):
    """The same corpus split across several shards."""
    d = str(tmp_path_factory.mktemp("sharded"))
    path = build_from_array(d, small_corpus, sep=999, shard_tokens=7_000)
    return InfiniGram(path)
