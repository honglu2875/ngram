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

"""The user-facing index."""

from __future__ import annotations

import os
from typing import Iterable, List, NamedTuple, Optional, Sequence, Union

import numpy as np

from ._core import Engine
from .format import IndexConfig

TokenSeq = Union[str, Sequence[int], np.ndarray]


class ProbResult(NamedTuple):
    """``count(prompt + [cont]) / count(prompt)``."""

    prob: float
    prompt_count: int
    cont_count: int


class InfgramProbResult(NamedTuple):
    """Same, but conditioned on the longest suffix of the prompt that occurs."""

    prob: float
    suffix_len: int
    prompt_count: int
    cont_count: int


class NtdResult(NamedTuple):
    """A sparse distribution over the token following a context.

    ``prompt_count`` is the total mass of the distribution, i.e. the number of
    occurrences of the context that have a successor.  That is the occurrence
    count of the context itself, minus the at most one occurrence per shard
    sitting flush against the end of the shard.  Probabilities are taken against
    this total, so they still sum to 1 -- and they keep summing to less than 1,
    correctly, when the separator token is filtered out of the support.
    """

    token_ids: np.ndarray
    counts: np.ndarray
    prompt_count: int
    suffix_len: int = -1

    @property
    def probs(self) -> np.ndarray:
        if self.prompt_count == 0:
            return np.zeros(self.counts.shape, dtype=np.float64)
        return self.counts.astype(np.float64) / float(self.prompt_count)

    def top(self, k: int = 10):
        """The ``k`` most frequent continuations, as ``(token_id, count, prob)``."""
        if self.counts.size == 0:
            return []
        order = np.argsort(self.counts)[::-1][:k]
        probs = self.probs
        return [
            (int(self.token_ids[i]), int(self.counts[i]), float(probs[i]))
            for i in order
        ]

    def dense(self, vocab_size: int) -> np.ndarray:
        """Materialise into a dense probability vector.

        Only useful for small vocabularies -- the whole point of the sparse
        representation is that you rarely want this.
        """
        out = np.zeros(vocab_size, dtype=np.float64)
        if self.token_ids.size:
            keep = self.token_ids < vocab_size
            out[self.token_ids[keep]] = self.probs[keep]
        return out


class DocOccurrence(NamedTuple):
    shard: int
    position: int
    doc_id: int
    offset_in_doc: int
    tokens: np.ndarray


class InfiniGram:
    """A memory-mapped suffix-array index over a tokenized corpus.

    Opening is instant regardless of corpus size: the token array and suffix
    array stay on disk and are paged in on demand, while only the search
    accelerator is held in RAM.

    >>> idx = InfiniGram("index/fineweb-8B")          # doctest: +SKIP
    >>> idx.count([464, 3797])                        # doctest: +SKIP
    >>> idx.infgram_prob(prompt_ids, next_id)         # doctest: +SKIP
    """

    def __init__(
        self,
        path: str,
        *,
        threads: int = 0,
        accel: bool = True,
        tokenizer=None,
        load_tokenizer: bool = False,
        check_tokens: bool = True,
    ):
        self.path = path
        self.config = IndexConfig.load(path)
        dirs = self.config.shard_dirs(path)
        missing = [d for d in dirs if not os.path.exists(os.path.join(d, "table"))]
        if missing:
            raise FileNotFoundError(
                "index at %s is incomplete -- no suffix array in %s"
                % (path, ", ".join(missing))
            )
        self._engine = Engine(dirs, self.config.token_width, threads, True)
        if not accel:
            self._engine.set_accel(False)
        self.check_tokens = check_tokens

        self.tokenizer = tokenizer
        if tokenizer is None and load_tokenizer and self.config.tokenizer:
            from .readers import load_tokenizer as _lt

            self.tokenizer = _lt(self.config.tokenizer)

    # -- introspection ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "InfiniGram(path=%r, tokens=%s, docs=%s, shards=%d, dtype=%s)"
            % (
                self.path,
                f"{self.tok_cnt:,}",
                f"{self.doc_cnt:,}",
                self.num_shards,
                self.config.token_dtype,
            )
        )

    @property
    def tok_cnt(self) -> int:
        return self._engine.tok_cnt

    @property
    def doc_cnt(self) -> int:
        return self._engine.doc_cnt

    @property
    def num_shards(self) -> int:
        return self._engine.num_shards

    @property
    def num_threads(self) -> int:
        return self._engine.num_threads

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def dtype(self) -> np.dtype:
        return self.config.dtype

    def set_accel(self, on: bool) -> None:
        """Toggle the in-RAM jump table.  Results are unchanged either way."""
        self._engine.set_accel(on)

    def warm(self) -> None:
        """Hint the kernel to page the whole index in."""
        self._engine.warm()

    def evict(self) -> None:
        """Drop the index from page cache, so the next query hits disk."""
        self._engine.evict()

    # -- tokenization -------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError(
                "no tokenizer attached; pass tokenizer=... or load_tokenizer=True "
                "(the index records %r)" % self.config.tokenizer
            )
        enc = self.tokenizer
        ids = enc.encode_ordinary(text) if hasattr(enc, "encode_ordinary") else enc.encode(text)
        return np.asarray(ids, dtype=self.dtype)

    def decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer attached")
        return self.tokenizer.decode([int(i) for i in np.asarray(ids).reshape(-1)])

    def _ids(self, seq: TokenSeq) -> np.ndarray:
        if isinstance(seq, str):
            arr = self.encode(seq)
        else:
            arr = np.ascontiguousarray(seq, dtype=self.dtype)
        if arr.ndim != 1:
            raise ValueError("expected a 1-D token sequence, got shape %s" % (arr.shape,))
        if self.check_tokens and arr.size:
            sep = self.config.doc_sep_token
            if sep is not None and np.any(arr == sep):
                raise ValueError(
                    "query contains the document separator token %d; such an "
                    "n-gram would span document boundaries" % sep
                )
        return arr

    # -- counting -----------------------------------------------------------

    def count(self, ngram: TokenSeq) -> int:
        """Occurrences of ``ngram`` in the corpus."""
        return self._engine.count(self._ids(ngram))

    def count_batch(self, ngrams: Iterable[TokenSeq]) -> np.ndarray:
        """Counts for many n-grams at once, in parallel.

        Sequences may differ in length; they are padded internally and the true
        lengths passed through.
        """
        seqs = [self._ids(s) for s in ngrams]
        if not seqs:
            return np.zeros(0, dtype=np.uint64)
        width = max(len(s) for s in seqs)
        pad = np.zeros((len(seqs), max(width, 1)), dtype=self.dtype)
        lens = np.zeros(len(seqs), dtype=np.uint32)
        for i, s in enumerate(seqs):
            pad[i, : len(s)] = s
            lens[i] = len(s)
        return self._engine.count_batch(pad, lens)

    def prob(self, prompt: TokenSeq, cont: int) -> ProbResult:
        """n-gram probability of ``cont`` following ``prompt``, for arbitrary n."""
        p, pc, cc = self._engine.prob(self._ids(prompt), int(cont))
        return ProbResult(p, pc, cc)

    def prob_batch(self, prompts: Iterable[TokenSeq], conts: Sequence[int]):
        """Returns ``(probs, prompt_counts, cont_counts)``."""
        seqs = [self._ids(s) for s in prompts]
        conts = np.ascontiguousarray(conts, dtype=self.dtype)
        if len(seqs) != conts.shape[0]:
            raise ValueError("prompts and conts must have the same length")
        if not seqs:
            z = np.zeros(0)
            return z, z.astype(np.uint64), z.astype(np.uint64)
        width = max(max(len(s) for s in seqs), 1)
        pad = np.zeros((len(seqs), width), dtype=self.dtype)
        lens = np.zeros(len(seqs), dtype=np.uint32)
        for i, s in enumerate(seqs):
            pad[i, : len(s)] = s
            lens[i] = len(s)
        pc, cc = self._engine.prob_batch(pad, conts, lens)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.where(pc > 0, cc.astype(np.float64) / np.maximum(pc, 1), 0.0)
        return probs, pc, cc

    # -- distributions ------------------------------------------------------

    def ntd(
        self,
        prompt: TokenSeq,
        *,
        max_support: int = 0,
        exclude_sep: bool = True,
    ) -> NtdResult:
        """Distribution over the token following ``prompt``.

        ``max_support`` caps the work for extremely frequent contexts by
        sampling ranks and extrapolating rather than visiting every occurrence;
        0 is exact.
        """
        ids = self._ids(prompt)
        tok, cnt = self._engine.ntd(ids, max_support)
        total = int(cnt.sum())
        tok, cnt = self._filter_sep(tok, cnt, exclude_sep)
        return NtdResult(tok, cnt, total)

    def infgram_ntd(
        self,
        prompt: TokenSeq,
        *,
        max_support: int = 0,
        max_len: int = 0,
        exclude_sep: bool = True,
    ) -> NtdResult:
        """:meth:`ntd` conditioned on the longest suffix of ``prompt`` that occurs."""
        ids = self._ids(prompt)
        tok, cnt, slen, _ = self._engine.infgram_ntd(ids, max_support, max_len)
        total = int(cnt.sum())  # same definition as ntd(); see NtdResult
        tok, cnt = self._filter_sep(tok, cnt, exclude_sep)
        return NtdResult(tok, cnt, total, slen)

    def _filter_sep(self, tok: np.ndarray, cnt: np.ndarray, exclude: bool):
        sep = self.config.doc_sep_token
        if not exclude or sep is None or tok.size == 0:
            return tok, cnt
        keep = tok != sep
        return tok[keep], cnt[keep]

    # -- infinite-gram ------------------------------------------------------

    def longest_suffix(self, prompt: TokenSeq, *, max_len: int = 0):
        """Longest suffix of ``prompt`` appearing in the corpus.

        Returns ``(suffix_len, count)``.
        """
        return self._engine.longest_suffix(self._ids(prompt), max_len)

    def infgram_prob(
        self, prompt: TokenSeq, cont: int, *, max_len: int = 0
    ) -> InfgramProbResult:
        """Infinite-gram probability: back off to the longest matching suffix."""
        p, slen, pc, cc = self._engine.infgram_prob(self._ids(prompt), int(cont), max_len)
        return InfgramProbResult(p, slen, pc, cc)

    def infgram_batch(self, tokens: np.ndarray, *, max_len: int = 0):
        """Infinite-gram probability of every token in a 2-D batch.

        Each row is scored left to right, reusing the previous position's
        matching suffix length, so cost is roughly two searches per token rather
        than a fresh backoff search each time.

        Returns ``(probs, suffix_lens, prompt_counts, cont_counts)``, each with
        the same shape as ``tokens``.  Column 0 has an empty context, so its
        probability is the unigram frequency of that token.
        """
        arr = np.ascontiguousarray(tokens, dtype=self.dtype)
        if arr.ndim == 1:
            arr = arr[None, :]
        return self._engine.infgram_batch(arr, max_len)

    def ntd_batch(
        self,
        tokens: np.ndarray,
        *,
        max_ctx: int = 0,
        max_support: int = 0,
        infinite: bool = False,
    ):
        """Next-token distributions at every position of a 2-D batch.

        Returns ``(indptr, token_ids, counts)`` in CSR layout over the flattened
        ``rows * cols`` positions: position ``i``'s support is
        ``token_ids[indptr[i]:indptr[i + 1]]``.

        This is the memory-frugal replacement for a dense
        ``(batch, seq, vocab)`` tensor -- a (8, 1024) batch over a 50k vocab
        would be 1.6 GB dense, and is typically a few MB here.
        """
        arr = np.ascontiguousarray(tokens, dtype=self.dtype)
        if arr.ndim == 1:
            arr = arr[None, :]
        return self._engine.ntd_batch(arr, max_ctx, max_support, infinite)

    # -- documents ----------------------------------------------------------

    def search_docs(
        self,
        ngram: TokenSeq,
        *,
        maxnum: int = 5,
        context: int = 64,
        seed: int = 0,
    ) -> List[DocOccurrence]:
        """Sample occurrences of ``ngram`` and return the surrounding text."""
        ids = self._ids(ngram)
        _, shards, positions = self._engine.sample_occurrences(ids, maxnum, seed)
        out = []
        for s, p in zip(shards, positions):
            toks, doc_id, off = self._engine.get_span(
                int(s), int(p), context, context + len(ids)
            )
            out.append(DocOccurrence(int(s), int(p), int(doc_id), int(off), toks))
        return out
