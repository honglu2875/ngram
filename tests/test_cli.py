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

"""Command line interface."""

from __future__ import annotations

import os

import numpy as np
import pytest

from ngram.cli import _parse_size, main

from .conftest import make_corpus
from .test_build import write_llmc


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1000", 1000),
        ("4e9", 4_000_000_000),
        ("4_000", 4000),
        ("4G", 4_000_000_000),
        ("500M", 500_000_000),
        ("2k", 2000),
        ("1.5G", 1_500_000_000),
    ],
)
def test_parse_size(text, expected):
    assert _parse_size(text) == expected


@pytest.fixture(scope="module")
def cli_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    toks = make_corpus(seed=21, n=20_000, vocab=64, sep=50256, n_docs=120)
    src = str(d / "shard.bin")
    write_llmc(src, toks)
    out = str(d / "idx")
    assert main(["build", src, "-o", out, "-f", "llmc", "--shard-tokens", "8k",
                 "--quiet"]) == 0
    return out, toks


def test_build_then_info(cli_index, capsys):
    out, toks = cli_index
    assert main(["info", out]) == 0
    printed = capsys.readouterr().out
    assert "20,000" in printed
    assert "shard 0000" in printed
    # 20k tokens in 8k shards, split at document boundaries.
    assert "shards     3" in printed


def test_count_with_token_ids(cli_index, capsys):
    out, toks = cli_index
    pat = [int(x) for x in toks[100:104]]
    if 50256 in pat:
        pytest.skip("pattern straddles a separator")
    assert main(["count", out, "--tokens", ",".join(map(str, pat))]) == 0
    printed = capsys.readouterr().out
    assert "occurrence" in printed


def test_ntd_command(cli_index, capsys):
    out, toks = cli_index
    pat = [int(toks[7])]
    assert main(["ntd", out, "--tokens", str(pat[0]), "--top", "3"]) == 0
    printed = capsys.readouterr().out
    assert "distinct continuations" in printed


def test_infgram_command(cli_index, capsys):
    out, toks = cli_index
    pat = [int(x) for x in toks[300:308]]
    if 50256 in pat:
        pytest.skip("pattern straddles a separator")
    assert main(["infgram", out, "--tokens", ",".join(map(str, pat))]) == 0
    printed = capsys.readouterr().out
    assert "suffix used" in printed
    assert "probability" in printed


def test_bench_runs(cli_index, capsys):
    out, _ = cli_index
    assert main(["bench", out, "--queries", "200", "--cold-queries", "50",
                 "--repeats", "1", "--threads", "2"]) == 0
    printed = capsys.readouterr().out
    assert "us/query" in printed
    assert "accel=True" in printed and "accel=False" in printed


def test_missing_index_reports_cleanly(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nope")]) == 1
    assert "not an ngram index" in capsys.readouterr().err


def test_query_without_text_is_an_error(cli_index):
    out, _ = cli_index
    with pytest.raises(SystemExit):
        main(["count", out])
