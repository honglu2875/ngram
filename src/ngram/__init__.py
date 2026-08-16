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

"""Infini-gram: disk-backed suffix-array indexes over token corpora.

Build an index from tokenized files, then query n-grams of any length --
including the longest suffix that occurs at all, which is what makes the model
an "infinite-gram".

    from ngram import build_index, InfiniGram

    build_index(["data/"], "index/", fmt="llmc", doc_sep_token=50256)
    idx = InfiniGram("index/")
    idx.count([464, 3797])
"""

from ._core import Engine
from .build import build_index, build_shard_sa
from .format import IndexConfig
from .index import (
    DocOccurrence,
    InfgramProbResult,
    InfiniGram,
    NtdResult,
    ProbResult,
)

__version__ = "0.1.0"

__all__ = [
    "InfiniGram",
    "build_index",
    "build_shard_sa",
    "IndexConfig",
    "Engine",
    "NtdResult",
    "ProbResult",
    "InfgramProbResult",
    "DocOccurrence",
    "__version__",
]
