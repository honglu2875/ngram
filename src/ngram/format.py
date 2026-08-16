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

"""On-disk index layout.

An index is a directory::

    index/
      config.json          this file's IndexConfig, serialised
      shard.0000/
        tokenized          N * W bytes, little-endian uint16/uint32
        table              N * P bytes + 8 pad; suffix array as token indices
        offset             D * 8 bytes, uint64 token index of each doc start
        bucket             in-RAM search accelerator (header + unigram + pivots)
      shard.0001/ ...

Shards are independent suffix arrays over disjoint slices of the corpus, so a
count is the sum over shards.  That is what makes indexing a corpus larger than
RAM straightforward: each shard is built entirely in memory, and nothing ever
has to merge them.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

FORMAT = "ngram-sa-v1"

BUCKET_MAGIC = b"NGRAMBK1"
BUCKET_HEADER_BYTES = 64

#: Trailing zero bytes on ``table`` so the reader's unaligned 8-byte load of a
#: packed pointer never runs past the end of the mapping.
PTR_PAD = 8

DEFAULT_PIVOT_STRIDE = 512
DEFAULT_PIVOT_LEN = 8

DTYPES = {"u16": np.uint16, "u32": np.uint32}
WIDTHS = {"u16": 2, "u32": 4}


def ptr_size_for(n: int) -> int:
    """Bytes needed to store a token index in ``[0, n)``."""
    w = 1
    while w < 8 and (n >> (8 * w)) != 0:
        w += 1
    return w


@dataclass
class ShardInfo:
    tokens: int
    docs: int
    ptr_size: int

    @property
    def dirname(self) -> str:
        return ""  # filled in by IndexConfig.shard_dir


@dataclass
class IndexConfig:
    format: str = FORMAT
    token_dtype: str = "u16"
    vocab_size: int = 0
    #: Token that marks the start of a document.  ``None`` means the corpus is
    #: one undivided stream.  Queries must never contain it.
    doc_sep_token: Optional[int] = None
    #: True when the builder inserted the separator rather than finding it
    #: already present in the source stream.
    sep_inserted: bool = False
    total_tokens: int = 0
    total_docs: int = 0
    pivot_stride: int = DEFAULT_PIVOT_STRIDE
    pivot_len: int = DEFAULT_PIVOT_LEN
    tokenizer: Optional[str] = None
    shards: List[Dict[str, Any]] = field(default_factory=list)
    #: Names of pipeline steps already finished, for resumable builds.
    completed: List[str] = field(default_factory=list)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(DTYPES[self.token_dtype])

    @property
    def token_width(self) -> int:
        return WIDTHS[self.token_dtype]

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def shard_dir(self, root: str, i: int) -> str:
        return os.path.join(root, "shard.%04d" % i)

    def shard_dirs(self, root: str) -> List[str]:
        return [self.shard_dir(root, i) for i in range(self.num_shards)]

    def save(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        tmp = os.path.join(root, "config.json.tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
        os.replace(tmp, os.path.join(root, "config.json"))

    @classmethod
    def load(cls, root: str) -> "IndexConfig":
        path = os.path.join(root, "config.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "%s is not an ngram index (no config.json)" % root
            )
        with open(path) as f:
            raw = json.load(f)
        if raw.get("format") != FORMAT:
            raise ValueError(
                "unsupported index format %r (this build reads %r)"
                % (raw.get("format"), FORMAT)
            )
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def write_bucket(
    path: str,
    token_width: int,
    tok_cnt: int,
    unigram: np.ndarray,
    pivots: np.ndarray,
    pivot_stride: int,
) -> None:
    """Write the search accelerator for one shard.

    ``unigram`` has ``vocab_slots + 1`` uint64 entries giving the first suffix
    rank of each token id; ``pivots`` is ``(num_pivots, pivot_len)`` sampled
    suffix prefixes.
    """
    unigram = np.ascontiguousarray(unigram, dtype=np.uint64)
    pivots = np.ascontiguousarray(pivots)
    vocab_slots = unigram.shape[0] - 1
    num_pivots, pivot_len = (pivots.shape if pivots.size else (0, 0))

    header = bytearray(BUCKET_HEADER_BYTES)
    header[0:8] = BUCKET_MAGIC
    struct.pack_into("<III", header, 8, token_width, vocab_slots, pivot_len)
    struct.pack_into("<QQQ", header, 24, tok_cnt, pivot_stride, num_pivots)

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(bytes(header))
        f.write(unigram.tobytes())
        f.write(pivots.tobytes())
    os.replace(tmp, path)


def read_bucket(path: str):
    """Inverse of :func:`write_bucket`; used by tests and tooling."""
    with open(path, "rb") as f:
        header = f.read(BUCKET_HEADER_BYTES)
        if header[0:8] != BUCKET_MAGIC:
            raise ValueError("not a bucket file: %s" % path)
        token_width, vocab_slots, pivot_len = struct.unpack_from("<III", header, 8)
        tok_cnt, pivot_stride, num_pivots = struct.unpack_from("<QQQ", header, 24)
        unigram = np.frombuffer(f.read((vocab_slots + 1) * 8), dtype=np.uint64)
        dtype = np.uint16 if token_width == 2 else np.uint32
        pivots = np.frombuffer(
            f.read(num_pivots * pivot_len * token_width), dtype=dtype
        ).reshape(num_pivots, pivot_len)
    return dict(
        token_width=token_width,
        tok_cnt=tok_cnt,
        pivot_stride=pivot_stride,
        unigram=unigram,
        pivots=pivots,
    )
