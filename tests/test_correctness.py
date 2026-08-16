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

"""Every query path checked against brute force.

An index that is subtly wrong is worse than no index, so these run first and
everything else is downstream of them passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ngram import InfiniGram

from .conftest import Oracle, build_from_array, make_corpus


def _sample_patterns(rng, tokens, oracle, count=200, max_len=6):
    """A mix of substrings that definitely occur and random ones that mostly don't."""
    n = len(tokens)
    out = []
    for _ in range(count):
        L = int(rng.integers(1, max_len + 1))
        if rng.random() < 0.7:
            s = int(rng.integers(0, n - L))
            pat = np.asarray(tokens[s : s + L])
        else:
            pat = rng.integers(0, 24, size=L, dtype=np.uint16)
        if oracle.sep is not None and (pat == oracle.sep).any():
            continue
        out.append(pat)
    return out


# --------------------------------------------------------------------------
# count
# --------------------------------------------------------------------------


def test_count_matches_oracle(small_index, small_oracle, small_corpus):
    rng = np.random.default_rng(11)
    for pat in _sample_patterns(rng, small_corpus, small_oracle):
        assert small_index.count(pat) == small_oracle.count(pat), pat


def test_count_empty_pattern_is_corpus_size(small_index, small_corpus):
    assert small_index.count([]) == len(small_corpus)


def test_count_of_absent_ngram_is_zero(small_index):
    # Tokens 100..104 never appear in a vocab-24 corpus.
    assert small_index.count([100, 101, 102]) == 0


def test_unigram_counts_match(small_index, small_corpus):
    vals, counts = np.unique(small_corpus, return_counts=True)
    for v, c in zip(vals, counts):
        if v == 999:
            continue
        assert small_index.count([int(v)]) == int(c)


def test_count_batch_matches_count(small_index, small_oracle, small_corpus):
    rng = np.random.default_rng(12)
    pats = _sample_patterns(rng, small_corpus, small_oracle, count=64)
    got = small_index.count_batch(pats)
    assert [int(x) for x in got] == [small_oracle.count(p) for p in pats]


# --------------------------------------------------------------------------
# probabilities and distributions
# --------------------------------------------------------------------------


def test_prob_matches_oracle(small_index, small_oracle, small_corpus):
    rng = np.random.default_rng(13)
    for pat in _sample_patterns(rng, small_corpus, small_oracle, count=80, max_len=4):
        cont = int(rng.integers(0, 24))
        r = small_index.prob(pat, cont)
        assert r.prompt_count == small_oracle.count(pat)
        assert r.cont_count == small_oracle.count(list(pat) + [cont])
        expected = r.cont_count / r.prompt_count if r.prompt_count else 0.0
        assert r.prob == pytest.approx(expected)


def test_ntd_matches_oracle(small_index, small_oracle, small_corpus):
    rng = np.random.default_rng(14)
    for pat in _sample_patterns(rng, small_corpus, small_oracle, count=60, max_len=4):
        res = small_index.ntd(pat, exclude_sep=False)
        got = {int(t): int(c) for t, c in zip(res.token_ids, res.counts)}
        assert got == small_oracle.ntd(pat), pat


def test_ntd_support_is_sorted_and_sums_to_count(small_index, small_corpus):
    for start in (0, 100, 5000):
        pat = small_corpus[start : start + 3]
        if 999 in pat:
            continue
        res = small_index.ntd(pat, exclude_sep=False)
        assert list(res.token_ids) == sorted(res.token_ids)
        # The one occurrence flush against the end of a shard has no successor.
        assert res.counts.sum() <= small_index.count(pat)
        assert res.counts.sum() >= small_index.count(pat) - small_index.num_shards


def test_ntd_probs_normalise(small_index, small_corpus):
    res = small_index.ntd(small_corpus[10:13], exclude_sep=False)
    assert res.probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_ntd_approximation_stays_close(small_index, small_corpus):
    """Sampled next-token counts should track exact ones on a frequent context."""
    pat = [int(small_corpus[5])]
    exact = small_index.ntd(pat, exclude_sep=False)
    approx = small_index.ntd(pat, max_support=64, exclude_sep=False)
    assert approx.counts.sum() == exact.counts.sum()
    e = {int(t): int(c) for t, c in zip(exact.token_ids, exact.counts)}
    a = {int(t): int(c) for t, c in zip(approx.token_ids, approx.counts)}
    for tok, c in a.items():
        assert abs(c - e.get(tok, 0)) <= 0.5 * exact.counts.sum() / 32 + 32


# --------------------------------------------------------------------------
# infinite-gram
# --------------------------------------------------------------------------


def test_longest_suffix_matches_oracle(small_index, small_oracle, small_corpus):
    rng = np.random.default_rng(15)
    for _ in range(60):
        L = int(rng.integers(1, 40))
        s = int(rng.integers(0, len(small_corpus) - L))
        prompt = np.asarray(small_corpus[s : s + L])
        if (prompt == 999).any():
            continue
        got, cnt = small_index.longest_suffix(prompt)
        assert got == small_oracle.longest_suffix(prompt)
        assert cnt == small_oracle.count(prompt[len(prompt) - got :] if got else [])


def test_longest_suffix_on_unseen_tokens_is_zero(small_index):
    assert small_index.longest_suffix([500, 501, 502])[0] == 0


def test_infgram_prob_matches_oracle(small_index, small_oracle, small_corpus):
    rng = np.random.default_rng(16)
    for _ in range(60):
        L = int(rng.integers(1, 30))
        s = int(rng.integers(0, len(small_corpus) - L - 1))
        prompt = np.asarray(small_corpus[s : s + L])
        if (prompt == 999).any():
            continue
        cont = int(small_corpus[s + L])
        if cont == 999:
            continue
        r = small_index.infgram_prob(prompt, cont)
        p, sl, pc, cc = small_oracle.infgram_prob(prompt, cont)
        assert (r.suffix_len, r.prompt_count, r.cont_count) == (sl, pc, cc)
        assert r.prob == pytest.approx(p)


def test_infgram_respects_max_len(small_index, small_oracle, small_corpus):
    prompt = np.asarray(small_corpus[200:240])
    if (prompt == 999).any():
        pytest.skip("prompt straddles a document boundary")
    for cap in (1, 2, 5):
        got, _ = small_index.longest_suffix(prompt, max_len=cap)
        assert got == small_oracle.longest_suffix(prompt, max_len=cap)
        assert got <= cap


def test_infgram_batch_matches_single(small_index, small_corpus):
    seq = np.asarray(small_corpus[1000:1064]).reshape(1, -1)
    probs, slens, pcs, ccs = small_index.infgram_batch(seq)
    for t in range(1, seq.shape[1]):
        if (seq[0, :t] == 999).any() or seq[0, t] == 999:
            continue
        r = small_index.infgram_prob(seq[0, :t], int(seq[0, t]))
        assert int(slens[0, t]) == r.suffix_len, t
        assert int(pcs[0, t]) == r.prompt_count
        assert int(ccs[0, t]) == r.cont_count
        assert probs[0, t] == pytest.approx(r.prob)


def test_infgram_batch_first_column_is_unigram(small_index, small_corpus):
    seq = np.asarray(small_corpus[2000:2032]).reshape(1, -1)
    probs, slens, pcs, ccs = small_index.infgram_batch(seq)
    assert int(slens[0, 0]) == 0
    assert int(pcs[0, 0]) == small_index.tok_cnt
    assert int(ccs[0, 0]) == small_index.count([int(seq[0, 0])])


def test_ntd_batch_matches_single(small_index, small_corpus):
    seq = np.asarray(small_corpus[3000:3024]).reshape(2, 12)
    indptr, ids, counts = small_index.ntd_batch(seq, max_ctx=3)
    rows, cols = seq.shape
    for r in range(rows):
        for t in range(cols):
            i = r * cols + t
            ctx = seq[r, max(0, t - 3) : t]
            if (ctx == 999).any():
                continue
            single = small_index.ntd(ctx, exclude_sep=False)
            lo, hi = int(indptr[i]), int(indptr[i + 1])
            assert list(ids[lo:hi]) == list(single.token_ids)
            assert list(counts[lo:hi]) == list(single.counts)


# --------------------------------------------------------------------------
# structural invariants
# --------------------------------------------------------------------------


def test_sharding_does_not_change_answers(
    small_index, sharded_index, small_oracle, small_corpus
):
    assert sharded_index.num_shards > 1
    assert sharded_index.tok_cnt == small_index.tok_cnt
    rng = np.random.default_rng(17)
    for pat in _sample_patterns(rng, small_corpus, small_oracle, count=150):
        assert sharded_index.count(pat) == small_oracle.count(pat), pat


def test_sharded_ntd_matches_oracle(sharded_index, small_oracle, small_corpus):
    rng = np.random.default_rng(18)
    for pat in _sample_patterns(rng, small_corpus, small_oracle, count=40, max_len=4):
        res = sharded_index.ntd(pat, exclude_sep=False)
        got = {int(t): int(c) for t, c in zip(res.token_ids, res.counts)}
        expected = small_oracle.ntd(pat)
        # A pattern occurrence at the tail of a shard has no successor within
        # that shard, so at most one count per shard may be missing.
        missing = sum(expected.values()) - sum(got.values())
        assert 0 <= missing <= sharded_index.num_shards
        for tok, c in got.items():
            assert c <= expected.get(tok, 0)


def test_accelerator_does_not_change_answers(small_index, small_corpus, small_oracle):
    """The in-RAM jump table is an optimisation; it must be answer-neutral."""
    rng = np.random.default_rng(19)
    pats = _sample_patterns(rng, small_corpus, small_oracle, count=200)
    with_accel = [small_index.count(p) for p in pats]
    small_index.set_accel(False)
    try:
        without = [small_index.count(p) for p in pats]
    finally:
        small_index.set_accel(True)
    assert with_accel == without


def test_ngrams_never_span_documents(tmp_path):
    """A pattern straddling a separator must not be found."""
    toks = np.array([1, 2, 3, 999, 4, 5, 6, 999, 1, 2, 3], dtype=np.uint16)
    path = build_from_array(str(tmp_path), toks, sep=999, shard_tokens=10**9)
    idx = InfiniGram(path, check_tokens=False)
    assert idx.count([1, 2, 3]) == 2
    assert idx.count([3, 999]) == 1  # only reachable by querying the separator
    # Nothing that skips the separator exists:
    assert idx.count([3, 4]) == 0
    assert idx.count([6, 1]) == 0


def test_query_containing_separator_is_rejected(small_index):
    with pytest.raises(ValueError, match="document separator"):
        small_index.count([1, 999, 2])


def test_suffix_array_is_sorted_and_a_permutation(small_index, small_corpus):
    """Read the packed table back and verify the fundamental invariant."""
    import os

    from ngram.format import PTR_PAD

    cfg = small_index.config
    d = cfg.shard_dir(small_index.path, 0)
    n = cfg.shards[0]["tokens"]
    width = cfg.shards[0]["ptr_size"]
    raw = np.fromfile(os.path.join(d, "table"), dtype=np.uint8)
    assert raw.size == n * width + PTR_PAD
    padded = np.zeros((n, 8), dtype=np.uint8)
    padded[:, :width] = raw[: n * width].reshape(n, width)
    sa = padded.view(np.uint64).reshape(n)

    assert np.array_equal(np.sort(sa), np.arange(n, dtype=np.uint64))

    tokens = np.fromfile(os.path.join(d, "tokenized"), dtype=cfg.dtype)
    rng = np.random.default_rng(20)
    for _ in range(300):
        i = int(rng.integers(0, n - 1))
        a = tokens[sa[i] :][:64]
        b = tokens[sa[i + 1] :][:64]
        m = min(len(a), len(b))
        diff = np.flatnonzero(a[:m] != b[:m])
        if diff.size:
            assert a[diff[0]] < b[diff[0]]
        else:
            assert len(a) <= len(b)


def test_bucket_unigram_table_matches_counts(small_index, small_corpus):
    import os

    from ngram.format import read_bucket

    d = small_index.config.shard_dir(small_index.path, 0)
    b = read_bucket(os.path.join(d, "bucket"))
    uni = b["unigram"]
    vals, counts = np.unique(small_corpus, return_counts=True)
    for v, c in zip(vals, counts):
        assert int(uni[v + 1] - uni[v]) == int(c)
    assert int(uni[-1]) == len(small_corpus)
