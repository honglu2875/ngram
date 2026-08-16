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

"""Input adapters, dtypes, persistence and the build driver."""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

from ngram import InfiniGram, build_index
from ngram.build import plan_concurrency
from ngram.format import IndexConfig
from ngram.readers import LLMC_HEADER_BYTES, LLMC_MAGIC, expand_inputs, llmc_token_count

from .conftest import Oracle, build_from_array, make_corpus


def write_llmc(path: str, tokens: np.ndarray) -> None:
    """Write an llm.c/nanoGPT style shard, as published by fineweb-scaled-gpt2."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = LLMC_MAGIC
    header[1] = 1
    header[2] = tokens.shape[0]
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.ascontiguousarray(tokens, dtype=np.uint16).tobytes())


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------


def test_llmc_roundtrip(tmp_path):
    toks = make_corpus(seed=1, n=5000, vocab=50, sep=50256)
    p = str(tmp_path / "shard.bin")
    write_llmc(p, toks)
    assert llmc_token_count(p) == 5000
    assert os.path.getsize(p) == LLMC_HEADER_BYTES + 5000 * 2

    out = str(tmp_path / "idx")
    build_index([p], out, fmt="llmc", doc_sep_token=50256, shard_tokens=10**9,
                verbose=False)
    idx = InfiniGram(out)
    assert idx.tok_cnt == 5000
    oracle = Oracle(toks, sep=50256)
    for start in (0, 17, 900, 4000):
        pat = toks[start : start + 4]
        if 50256 in pat:
            continue
        assert idx.count(pat) == oracle.count(pat)


def test_llmc_rejects_bad_magic(tmp_path):
    p = str(tmp_path / "bad.bin")
    with open(p, "wb") as f:
        f.write(struct.pack("<i", 12345) + b"\0" * (LLMC_HEADER_BYTES - 4))
        f.write(b"\0" * 100)
    with pytest.raises(ValueError, match="magic"):
        llmc_token_count(p)


def test_llmc_rejects_truncated_file(tmp_path):
    p = str(tmp_path / "trunc.bin")
    write_llmc(p, make_corpus(seed=2, n=1000, vocab=20, sep=None))
    with open(p, "r+b") as f:
        f.truncate(LLMC_HEADER_BYTES + 500)  # claim 1000 tokens, hold 250
    with pytest.raises(ValueError, match="truncated"):
        llmc_token_count(p)


def test_llmc_document_offsets_land_on_separators(tmp_path):
    toks = make_corpus(seed=3, n=8000, vocab=40, sep=50256, n_docs=60)
    p = str(tmp_path / "s.bin")
    write_llmc(p, toks)
    out = str(tmp_path / "idx")
    build_index([p], out, fmt="llmc", doc_sep_token=50256, shard_tokens=10**9,
                verbose=False)
    cfg = IndexConfig.load(out)
    offsets = np.fromfile(os.path.join(cfg.shard_dir(out, 0), "offset"), dtype=np.uint64)
    assert offsets.size == cfg.total_docs
    assert (toks[offsets.astype(np.int64)] == 50256).all()


def test_npy_input(tmp_path):
    toks = make_corpus(seed=4, n=4000, vocab=30, sep=None)
    p = str(tmp_path / "t.npy")
    np.save(p, toks)
    out = str(tmp_path / "idx")
    build_index([p], out, fmt="npy", doc_sep_token=None, shard_tokens=10**9,
                verbose=False)
    idx = InfiniGram(out)
    assert idx.tok_cnt == 4000
    assert idx.count(toks[10:14]) >= 1


def test_jsonl_input_inserts_separators(tmp_path):
    class Toy:
        """Character-code tokenizer, enough to exercise the jsonl path."""

        def encode(self, text):
            return [ord(c) % 200 for c in text]

    docs = ["hello world", "another document here", "third one"]
    p = str(tmp_path / "docs.jsonl")
    with open(p, "w") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")

    out = str(tmp_path / "idx")
    build_index([p], out, fmt="jsonl", tokenizer=Toy(), insert_sep=True,
                shard_tokens=10**9, verbose=False)
    cfg = IndexConfig.load(out)
    assert cfg.doc_sep_token == 0xFFFF
    assert cfg.sep_inserted
    assert cfg.total_docs == 3
    assert cfg.total_tokens == sum(len(d) for d in docs) + 3

    idx = InfiniGram(out, check_tokens=False)
    tok = Toy()
    assert idx.count(tok.encode("hello")) == 1
    # "world" then "another" only reads as adjacent across a boundary.
    assert idx.count(tok.encode("dan")) == 0


def test_expand_inputs_finds_files(tmp_path):
    for i in range(3):
        (tmp_path / ("f%d.bin" % i)).write_bytes(b"\0" * 8)
    got = expand_inputs([str(tmp_path)])
    assert len(got) == 3
    got = expand_inputs([str(tmp_path / "*.bin")])
    assert len(got) == 3
    with pytest.raises(FileNotFoundError):
        expand_inputs([str(tmp_path / "nope*")])


# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------


def test_u32_index_matches_oracle(tmp_path):
    """The 32-bit path exists for vocabularies beyond 65535."""
    rng = np.random.default_rng(5)
    toks = rng.integers(0, 200_000, size=20_000, dtype=np.uint32)
    oracle = Oracle(toks, sep=None)
    path = build_from_array(str(tmp_path), toks, sep=None, shard_tokens=10**9,
                            dtype="u32", vocab_size=200_001)
    idx = InfiniGram(path)
    assert idx.config.token_dtype == "u32"
    assert idx.tok_cnt == 20_000
    for start in (0, 5, 999, 15_000):
        for L in (1, 2, 3):
            pat = toks[start : start + L]
            assert idx.count(pat) == oracle.count(pat)
    # High token ids must survive the round trip intact.
    big = int(toks.max())
    assert idx.count([big]) == oracle.count([big])


def test_u32_requires_vocab_size(tmp_path):
    toks = np.arange(100, dtype=np.uint32)
    raw = str(tmp_path / "c.bin")
    toks.tofile(raw)
    with pytest.raises(ValueError, match="vocab_size"):
        build_index([raw], str(tmp_path / "idx"), fmt="raw", token_dtype="u32",
                    shard_tokens=10**9, verbose=False)


# --------------------------------------------------------------------------
# persistence and resumption
# --------------------------------------------------------------------------


def test_reopen_gives_identical_answers(tmp_path):
    toks = make_corpus(seed=6, n=12_000, vocab=32, sep=None)
    path = build_from_array(str(tmp_path), toks, sep=None, shard_tokens=5_000)
    a = InfiniGram(path)
    pats = [toks[i : i + 4] for i in range(0, 8000, 400)]
    first = [a.count(p) for p in pats]
    del a
    b = InfiniGram(path)
    assert [b.count(p) for p in pats] == first


def test_build_is_resumable(tmp_path):
    toks = make_corpus(seed=7, n=9_000, vocab=20, sep=None)
    raw = str(tmp_path / "c.bin")
    toks.tofile(raw)
    out = str(tmp_path / "idx")
    build_index([raw], out, fmt="raw", doc_sep_token=None, shard_tokens=4_000,
                verbose=False)
    cfg = IndexConfig.load(out)
    assert cfg.num_shards == 3
    stamps = {
        os.path.join(cfg.shard_dir(out, i), "table"): os.path.getmtime(
            os.path.join(cfg.shard_dir(out, i), "table")
        )
        for i in range(cfg.num_shards)
    }
    # A second run must not redo any work.
    build_index([raw], out, fmt="raw", doc_sep_token=None, shard_tokens=4_000,
                verbose=False)
    for path, mtime in stamps.items():
        assert os.path.getmtime(path) == mtime


def test_incomplete_index_reports_clearly(tmp_path):
    toks = make_corpus(seed=8, n=3000, vocab=16, sep=None)
    path = build_from_array(str(tmp_path), toks, sep=None, shard_tokens=10**9)
    os.remove(os.path.join(IndexConfig.load(path).shard_dir(path, 0), "table"))
    with pytest.raises(FileNotFoundError, match="incomplete"):
        InfiniGram(path)


def test_unknown_format_version_is_rejected(tmp_path):
    toks = make_corpus(seed=9, n=2000, vocab=16, sep=None)
    path = build_from_array(str(tmp_path), toks, sep=None, shard_tokens=10**9)
    cfgpath = os.path.join(path, "config.json")
    raw = json.load(open(cfgpath))
    raw["format"] = "ngram-sa-v99"
    json.dump(raw, open(cfgpath, "w"))
    with pytest.raises(ValueError, match="unsupported index format"):
        InfiniGram(path)


def test_vocab_size_derived_from_suffix_array(tmp_path):
    """Ingest no longer scans for the max token; libsais' frequencies supply it."""
    rng = np.random.default_rng(31)
    toks = rng.integers(0, 4000, size=6000, dtype=np.uint16)
    toks[123] = 7777  # a lone high token must still set the vocabulary
    path = build_from_array(str(tmp_path), toks, sep=None, shard_tokens=2000)
    cfg = IndexConfig.load(path)
    assert cfg.num_shards == 3
    assert cfg.vocab_size == 7778


def test_explicit_vocab_size_is_not_overridden(tmp_path):
    toks = make_corpus(seed=32, n=3000, vocab=50, sep=None)
    path = build_from_array(str(tmp_path), toks, sep=None, shard_tokens=10**9,
                            vocab_size=65000)
    assert IndexConfig.load(path).vocab_size == 65000


def test_limit_tokens_truncates(tmp_path):
    toks = make_corpus(seed=10, n=10_000, vocab=16, sep=None)
    raw = str(tmp_path / "c.bin")
    toks.tofile(raw)
    out = str(tmp_path / "idx")
    build_index([raw], out, fmt="raw", doc_sep_token=None, shard_tokens=10**9,
                limit_tokens=2500, verbose=False)
    assert InfiniGram(out).tok_cnt == 2500


# --------------------------------------------------------------------------
# planning and documents
# --------------------------------------------------------------------------


def test_plan_concurrency_respects_memory():
    # 4e9-token shards at ~15 bytes/token need ~60 GB each.
    conc, threads = plan_concurrency(4_000_000_000, cpus=30, mem_bytes=240 << 30)
    assert conc == 4 and threads == 7
    conc, threads = plan_concurrency(4_000_000_000, cpus=30, mem_bytes=32 << 30)
    assert conc == 1 and threads == 30
    conc, threads = plan_concurrency(1000, cpus=8, mem_bytes=1 << 30)
    assert conc == 8 and threads == 1
    # Widening u32 inputs to int64 for libsais costs more, so fewer fit at once.
    assert plan_concurrency(4_000_000_000, 30, 240 << 30, "u32")[0] == 2
    # Cores left over by a small shard count go to libsais' own threads.
    assert plan_concurrency(60_000_000, 30, 240 << 30, "u16", num_jobs=2) == (2, 15)
    assert plan_concurrency(60_000_000, 30, 240 << 30, "u16", num_jobs=100) == (30, 1)


def test_search_docs_returns_real_occurrences(tmp_path):
    toks = make_corpus(seed=11, n=20_000, vocab=12, sep=999, n_docs=100)
    path = build_from_array(str(tmp_path), toks, sep=999, shard_tokens=10**9)
    idx = InfiniGram(path)
    pat = np.asarray(toks[500:505])
    if (pat == 999).any():
        pytest.skip("pattern straddles a separator")
    hits = idx.search_docs(pat, maxnum=5, context=20)
    assert 0 < len(hits) <= 5
    for occ in hits:
        assert np.array_equal(toks[occ.position : occ.position + len(pat)], pat)
        assert 999 not in occ.tokens[1:]  # spans never cross into a neighbour doc


def test_search_docs_on_absent_ngram_is_empty(tmp_path):
    toks = make_corpus(seed=12, n=2000, vocab=8, sep=None)
    path = build_from_array(str(tmp_path), toks, sep=None, shard_tokens=10**9)
    idx = InfiniGram(path)
    assert idx.search_docs([100, 101, 102]) == []
