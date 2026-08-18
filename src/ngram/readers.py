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

"""Input adapters.

Every reader yields ``(tokens, doc_starts)`` blocks, where ``doc_starts`` are
block-relative indices at which a document begins.  Readers never modify the
token stream; deciding whether to insert a separator is the builder's job.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Header used by llm.c / nanoGPT style ``.bin`` token shards, as published by
#: e.g. ``quintic/fineweb-scaled-gpt2``: 1024 bytes of little-endian int32,
#: slot 0 magic, slot 1 version, slot 2 token count.
LLMC_HEADER_BYTES = 1024
LLMC_MAGIC = 20240520

Block = Tuple[np.ndarray, np.ndarray]

DEFAULT_BLOCK_TOKENS = 64 << 20  # 128 MiB of uint16


def expand_inputs(inputs: Sequence[str], pattern: Optional[str] = None) -> List[str]:
    """Resolve files, directories and globs into a sorted list of files."""
    out: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            pat = pattern or "*"
            out.extend(sorted(glob.glob(os.path.join(item, "**", pat), recursive=True)))
        else:
            hits = sorted(glob.glob(item))
            out.extend(hits if hits else [item])
    files = [p for p in out if os.path.isfile(p)]
    if not files:
        raise FileNotFoundError("no input files matched %r" % (list(inputs),))
    return files


def llmc_token_count(path: str) -> int:
    """Token count from an llm.c ``.bin`` header, validating magic and size."""
    with open(path, "rb") as f:
        header = f.read(LLMC_HEADER_BYTES)
    if len(header) < LLMC_HEADER_BYTES:
        raise ValueError("%s is too short to be an llm.c bin file" % path)
    magic, version, ntok = struct.unpack_from("<iii", header, 0)
    if magic != LLMC_MAGIC:
        raise ValueError(
            "%s has magic %d, expected %d -- is this really an llm.c .bin?"
            % (path, magic, LLMC_MAGIC)
        )
    if version != 1:
        raise ValueError("%s has unsupported llm.c version %d" % (path, version))
    on_disk = (os.path.getsize(path) - LLMC_HEADER_BYTES) // 2
    if on_disk != ntok:
        raise ValueError(
            "%s claims %d tokens but holds %d -- truncated download?"
            % (path, ntok, on_disk)
        )
    return ntok


def count_tokens(paths: Sequence[str], fmt: str, dtype: np.dtype) -> int:
    """Total tokens across inputs, without reading payloads where possible."""
    if fmt == "llmc":
        return sum(llmc_token_count(p) for p in paths)
    if fmt == "raw":
        return sum(os.path.getsize(p) // dtype.itemsize for p in paths)
    if fmt == "npy":
        return sum(int(np.load(p, mmap_mode="r").shape[0]) for p in paths)
    return 0  # jsonl needs tokenization to know


def _emit(
    tokens: np.ndarray,
    doc_sep_token: Optional[int],
    block_tokens: int,
) -> Iterator[Block]:
    """Chop a token array into blocks and locate document starts in each."""
    n = tokens.shape[0]
    for start in range(0, n, block_tokens):
        chunk = np.asarray(tokens[start : start + block_tokens])
        if doc_sep_token is None:
            starts = np.zeros(0, dtype=np.int64)
        else:
            starts = np.flatnonzero(chunk == doc_sep_token).astype(np.int64)
        yield chunk, starts


def read_llmc(
    paths: Sequence[str],
    doc_sep_token: Optional[int],
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
) -> Iterator[Block]:
    for path in paths:
        ntok = llmc_token_count(path)
        arr = np.memmap(
            path, dtype=np.uint16, mode="r", offset=LLMC_HEADER_BYTES, shape=(ntok,)
        )
        yield from _emit(arr, doc_sep_token, block_tokens)
        del arr


def read_raw(
    paths: Sequence[str],
    dtype: np.dtype,
    doc_sep_token: Optional[int],
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
) -> Iterator[Block]:
    for path in paths:
        arr = np.memmap(path, dtype=dtype, mode="r")
        yield from _emit(arr, doc_sep_token, block_tokens)
        del arr


def read_npy(
    paths: Sequence[str],
    dtype: np.dtype,
    doc_sep_token: Optional[int],
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
) -> Iterator[Block]:
    for path in paths:
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != dtype:
            raise ValueError(
                "%s has dtype %s but the index is %s" % (path, arr.dtype, dtype)
            )
        yield from _emit(arr.reshape(-1), doc_sep_token, block_tokens)
        del arr


def _open_text(path: str):
    if path.endswith(".gz"):
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".zst"):
        import io

        import zstandard as zstd

        raw = open(path, "rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, encoding="utf-8")


def read_jsonl(
    paths: Sequence[str],
    tokenizer,
    dtype: np.dtype,
    text_field: str = "text",
    block_tokens: int = DEFAULT_BLOCK_TOKENS,
    batch_docs: int = 1024,
) -> Iterator[Block]:
    """Tokenize JSONL documents into blocks.

    ``tokenizer`` needs only an ``encode`` (or ``encode_batch``) method, so both
    ``transformers`` and ``tiktoken`` objects work.
    """
    buf: List[np.ndarray] = []
    starts: List[int] = []
    pending = 0

    def flush() -> Iterator[Block]:
        nonlocal buf, starts, pending
        if not buf:
            return
        tokens = np.concatenate(buf)
        doc_starts = np.array(starts, dtype=np.int64)
        buf, starts, pending = [], [], 0
        yield tokens, doc_starts

    texts: List[str] = []

    def encode_batch(batch: List[str]) -> List[Sequence[int]]:
        if hasattr(tokenizer, "encode_batch"):
            return tokenizer.encode_batch(batch)
        if hasattr(tokenizer, "encode_ordinary_batch"):
            return tokenizer.encode_ordinary_batch(batch)
        if hasattr(tokenizer, "batch_encode_plus"):
            return tokenizer.batch_encode_plus(batch)["input_ids"]
        return [tokenizer.encode(t) for t in batch]

    for path in paths:
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if text_field not in obj:
                    raise KeyError(
                        "%s: line has no %r field (fields: %s)"
                        % (path, text_field, sorted(obj))
                    )
                texts.append(obj[text_field])
                if len(texts) >= batch_docs:
                    for ids in encode_batch(texts):
                        starts.append(pending)
                        arr = np.asarray(ids, dtype=dtype)
                        buf.append(arr)
                        pending += arr.shape[0]
                    texts = []
                    if pending >= block_tokens:
                        yield from flush()
    if texts:
        for ids in encode_batch(texts):
            starts.append(pending)
            arr = np.asarray(ids, dtype=dtype)
            buf.append(arr)
            pending += arr.shape[0]
    yield from flush()


def load_tokenizer(name: str):
    """Resolve a tokenizer name to an object with ``encode``.

    ``gpt2`` prefers ``tiktoken`` (much faster) and falls back to
    ``transformers``; anything else goes to ``transformers``.
    """
    if name in ("gpt2", "tiktoken:gpt2"):
        try:
            import tiktoken

            return tiktoken.get_encoding("gpt2")
        except ImportError:
            pass
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        name.split(":", 1)[-1], add_bos_token=False, add_eos_token=False
    )
