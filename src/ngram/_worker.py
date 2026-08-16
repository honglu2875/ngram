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

"""Subprocess entry point for building one shard's suffix array.

Run as ``python -m ngram._worker '<json spec>'``.

Shards are built in separate *processes*, not threads or forked children:

* libsais peaks at ~10 bytes per token, and process exit is the only way to be
  certain that memory is handed back before the next shard starts;
* ``multiprocessing`` with the ``spawn`` start method re-imports the parent's
  ``__main__``, which fails outright from a REPL, a notebook, or a script piped
  on stdin -- all normal ways to drive this library;
* ``fork`` is unsafe here because OpenMP state is undefined across ``fork()``.
"""

from __future__ import annotations

import json
import sys
import traceback


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m ngram._worker '<json spec>'", file=sys.stderr)
        return 2
    spec = json.loads(argv[0])
    try:
        from .build import build_shard_sa

        result = build_shard_sa(
            spec["shard_dir"],
            spec["token_dtype"],
            spec["alphabet"],
            spec["threads"],
            spec["pivot_stride"],
            spec["pivot_len"],
        )
    except Exception as exc:  # reported back to the driver, not swallowed
        result = {
            "shard_dir": spec.get("shard_dir"),
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }
        print("NGRAM_RESULT " + json.dumps(result), flush=True)
        return 1
    print("NGRAM_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
