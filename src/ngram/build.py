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

"""Index construction.

Two steps, both resumable:

1. **ingest** -- stream the input into per-shard ``tokenized``/``offset`` files.
2. **suffix arrays** -- for each shard, build the suffix array in RAM with
   libsais, pack it to disk, and write the search accelerator.

Step 2 is where the time goes, and it parallelises across *shards in separate
processes* rather than across threads inside one libsais call.  Measured on this
class of machine, libsais' own OpenMP scaling is about 1.25x at 30 threads
(it is memory-bandwidth bound), whereas running independent shards concurrently
scales nearly linearly until memory runs out.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import _core, readers
from .format import (
    DEFAULT_PIVOT_LEN,
    DEFAULT_PIVOT_STRIDE,
    DTYPES,
    PTR_PAD,
    IndexConfig,
    ptr_size_for,
    write_bucket,
)

#: Peak bytes per token during a shard's suffix-array build, used to decide how
#: many shards may build concurrently.
#:
#: u16: 8 for the int64 suffix array (anonymous) + 2 for the token map + ~5 for
#: the packed table, the latter two as actively-used page cache.
#: u32: libsais has no 32-bit-alphabet entry point with 64-bit indices, so the
#: input is widened to int64 first -- 8 more bytes per token.
BUILD_BYTES_PER_TOKEN = {"u16": 15, "u32": 25}

DEFAULT_SHARD_TOKENS = 4_000_000_000


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print("[ngram %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f PiB" % n


# ---------------------------------------------------------------------------
# Step 1: ingest
# ---------------------------------------------------------------------------


class _ShardWriter:
    """Streams blocks into shard files, rolling over on document boundaries."""

    def __init__(
        self,
        root: str,
        dtype: np.dtype,
        shard_tokens: int,
        sep_token: Optional[int],
        insert_sep: bool,
        verbose: bool,
    ):
        self.root = root
        self.dtype = dtype
        self.shard_tokens = shard_tokens
        self.sep_token = sep_token
        self.insert_sep = insert_sep
        self.verbose = verbose

        self.shards: List[dict] = []
        self._idx = -1
        self._ds = None
        self._doc_offsets: List[np.ndarray] = []
        self._tokens = 0
        self._open()

    def _dir(self, i: int) -> str:
        return os.path.join(self.root, "shard.%04d" % i)

    def _open(self) -> None:
        self._idx += 1
        d = self._dir(self._idx)
        os.makedirs(d, exist_ok=True)
        self._ds = open(os.path.join(d, "tokenized"), "wb")
        self._doc_offsets = []
        self._tokens = 0

    def _close_current(self) -> None:
        if self._ds is None:
            return
        self._ds.close()
        d = self._dir(self._idx)
        offsets = (
            np.concatenate(self._doc_offsets)
            if self._doc_offsets
            else np.zeros(0, dtype=np.uint64)
        )
        # A shard always has at least one document, even if the source stream
        # carried no separators; the engine needs a non-empty offset table.
        if offsets.size == 0 or offsets[0] != 0:
            offsets = np.concatenate([np.zeros(1, dtype=np.uint64), offsets])
        with open(os.path.join(d, "offset"), "wb") as f:
            f.write(np.ascontiguousarray(offsets, dtype=np.uint64).tobytes())
        self.shards.append(
            dict(tokens=int(self._tokens), docs=int(offsets.size),
                 ptr_size=ptr_size_for(int(self._tokens)))
        )
        _log(
            self.verbose,
            "  shard %04d: %s tokens, %s docs"
            % (self._idx, f"{self._tokens:,}", f"{offsets.size:,}"),
        )
        self._ds = None

    def _rollover(self) -> None:
        self._close_current()
        self._open()

    def _append(self, tokens: np.ndarray, doc_starts: np.ndarray) -> None:
        if tokens.size == 0:
            return
        self._ds.write(np.ascontiguousarray(tokens, dtype=self.dtype).tobytes())
        if doc_starts.size:
            self._doc_offsets.append((doc_starts + self._tokens).astype(np.uint64))
        self._tokens += int(tokens.size)

    def add(self, tokens: np.ndarray, doc_starts: np.ndarray) -> None:
        if tokens.dtype != self.dtype:
            tokens = tokens.astype(self.dtype)
        if self.insert_sep and doc_starts.size:
            if self.sep_token is None:
                raise ValueError("insert_sep requires a separator token")
            tokens = np.insert(tokens, doc_starts, self.dtype.type(self.sep_token))
            # Each insertion shifts every later start by one more position, and
            # the document is taken to begin at its separator.
            doc_starts = doc_starts + np.arange(doc_starts.size, dtype=np.int64)

        pos = 0
        n = int(tokens.size)
        while pos < n:
            room = self.shard_tokens - self._tokens
            if room <= 0:
                self._rollover()
                continue
            if n - pos <= room:
                cut = n
            elif doc_starts.size:
                inrange = doc_starts[(doc_starts > pos) & (doc_starts <= pos + room)]
                # No boundary in range means one document straddles the whole
                # remaining budget; documents are never split, so let this shard
                # run over rather than corrupt the stream.
                cut = int(inrange[-1]) if inrange.size else n
            else:
                cut = pos + room

            sel = doc_starts[(doc_starts >= pos) & (doc_starts < cut)] - pos
            self._append(tokens[pos:cut], sel)
            pos = cut
            if pos < n:
                self._rollover()

    def close(self) -> List[dict]:
        self._close_current()
        # Drop a trailing empty shard, which happens when the corpus size is an
        # exact multiple of the shard budget.
        if self.shards and self.shards[-1]["tokens"] == 0:
            shutil.rmtree(self._dir(len(self.shards) - 1), ignore_errors=True)
            self.shards.pop()
        return self.shards


def _blocks(
    paths: Sequence[str],
    fmt: str,
    dtype: np.dtype,
    doc_sep_token: Optional[int],
    tokenizer,
    block_tokens: int,
) -> Iterator[readers.Block]:
    if fmt == "llmc":
        return readers.read_llmc(paths, doc_sep_token, block_tokens)
    if fmt == "raw":
        return readers.read_raw(paths, dtype, doc_sep_token, block_tokens)
    if fmt == "npy":
        return readers.read_npy(paths, dtype, doc_sep_token, block_tokens)
    if fmt == "jsonl":
        if tokenizer is None:
            raise ValueError("jsonl input needs --tokenizer")
        return readers.read_jsonl(paths, tokenizer, dtype, block_tokens=block_tokens)
    raise ValueError("unknown input format %r" % fmt)


# ---------------------------------------------------------------------------
# Step 2: suffix arrays (runs in worker processes)
# ---------------------------------------------------------------------------


def build_shard_sa(
    shard_dir: str,
    token_dtype: str,
    alphabet: int,
    threads: int,
    pivot_stride: int,
    pivot_len: int,
) -> dict:
    """Build ``table`` and ``bucket`` for one already-ingested shard.

    Written to be callable standalone (and from a worker process), so a failed
    build can be resumed one shard at a time.
    """
    dtype = np.dtype(DTYPES[token_dtype])
    ds_path = os.path.join(shard_dir, "tokenized")
    n = os.path.getsize(ds_path) // dtype.itemsize
    if n == 0:
        raise ValueError("%s is empty" % ds_path)

    tokens = np.memmap(ds_path, dtype=dtype, mode="r")
    t0 = time.time()
    sa, freq = _core.build_suffix_array(tokens, threads=threads, alphabet=alphabet)
    t_sa = time.time() - t0

    width = ptr_size_for(int(n))
    tmp = os.path.join(shard_dir, "table.tmp")
    table = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(n * width + PTR_PAD,))
    _core.pack_suffix_array(sa, width, out=table)
    table.flush()
    del table
    os.replace(tmp, os.path.join(shard_dir, "table"))

    unigram = _core.unigram_table(freq)
    pivots = _core.sample_pivots(tokens, sa, pivot_stride, pivot_len)
    write_bucket(
        os.path.join(shard_dir, "bucket"),
        token_width=dtype.itemsize,
        tok_cnt=int(n),
        unigram=unigram,
        pivots=pivots,
        pivot_stride=pivot_stride,
    )

    del sa, tokens
    return dict(
        shard_dir=shard_dir,
        tokens=int(n),
        ptr_size=width,
        sa_seconds=t_sa,
        max_token=int(np.flatnonzero(freq)[-1]) if freq.any() else 0,
    )


def run_shard_jobs(specs: Sequence[dict], concurrent: int, verbose: bool) -> List[dict]:
    """Run ``build_shard_sa`` for each spec, at most ``concurrent`` at a time.

    Each shard runs in its own subprocess so that its ~10 bytes/token peak is
    returned to the OS on exit, and so the driver works from a REPL or a piped
    script (see :mod:`ngram._worker`).
    """
    results: List[dict] = [None] * len(specs)  # type: ignore[list-item]
    pending = list(enumerate(specs))
    running: List[Tuple[int, subprocess.Popen]] = []

    def launch(idx: int, spec: dict) -> None:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(spec["threads"])
        proc = subprocess.Popen(
            [sys.executable, "-m", "ngram._worker", json.dumps(spec)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        running.append((idx, proc))

    while pending or running:
        while pending and len(running) < concurrent:
            idx, spec = pending.pop(0)
            launch(idx, spec)
            _log(verbose, "  started %s" % os.path.basename(spec["shard_dir"]))

        # Reap whichever worker finishes first, not whichever started first --
        # otherwise a single large shard holds up the whole queue behind it.
        done = None
        while done is None:
            for pos, (_, p) in enumerate(running):
                if p.poll() is not None:
                    done = pos
                    break
            if done is None:
                time.sleep(0.2)
        idx, proc = running.pop(done)

        out, err = proc.communicate()
        payload = None
        for line in out.splitlines():
            if line.startswith("NGRAM_RESULT "):
                payload = json.loads(line[len("NGRAM_RESULT "):])
        if payload is None:
            payload = {
                "shard_dir": specs[idx]["shard_dir"],
                "error": "worker exited with code %d" % proc.returncode,
                "traceback": (err or "")[-4000:],
            }
        results[idx] = payload
        if "error" not in payload:
            _log(
                verbose,
                "  finished %s: %s tokens, suffix array in %.1fs"
                % (
                    os.path.basename(payload["shard_dir"]),
                    f"{payload['tokens']:,}",
                    payload["sa_seconds"],
                ),
            )
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def plan_concurrency(
    shard_tokens: int,
    cpus: int,
    mem_bytes: int,
    token_dtype: str = "u16",
    num_jobs: Optional[int] = None,
) -> Tuple[int, int]:
    """Choose (concurrent shards, OpenMP threads per shard) for the SA step.

    Concurrency is capped by memory and by how many shards there are to build;
    whatever cores are left over then go to libsais' own OpenMP.  Clamping to
    ``num_jobs`` before dividing up the cores matters: with two shards on a
    30-core box the choice is 2x15 threads, not 2x1.
    """
    per_token = BUILD_BYTES_PER_TOKEN[token_dtype]
    by_mem = max(1, int(mem_bytes // (per_token * max(shard_tokens, 1))))
    concurrent = max(1, min(cpus, by_mem))
    if num_jobs is not None:
        concurrent = max(1, min(concurrent, num_jobs))
    threads = max(1, cpus // concurrent)
    return concurrent, threads


def build_index(
    inputs: Sequence[str],
    output: str,
    *,
    fmt: str = "llmc",
    token_dtype: str = "u16",
    vocab_size: Optional[int] = None,
    doc_sep_token: Optional[int] = None,
    insert_sep: bool = False,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    cpus: Optional[int] = None,
    mem_gb: Optional[float] = None,
    pivot_stride: int = DEFAULT_PIVOT_STRIDE,
    pivot_len: int = DEFAULT_PIVOT_LEN,
    tokenizer=None,
    tokenizer_name: Optional[str] = None,
    pattern: Optional[str] = None,
    block_tokens: int = readers.DEFAULT_BLOCK_TOKENS,
    limit_tokens: Optional[int] = None,
    resume: bool = True,
    verbose: bool = True,
) -> IndexConfig:
    """Build an index directory from tokenized inputs.

    Returns the saved :class:`~ngram.format.IndexConfig`.
    """
    if token_dtype not in DTYPES:
        raise ValueError("token_dtype must be one of %s" % sorted(DTYPES))
    dtype = np.dtype(DTYPES[token_dtype])
    cpus = cpus or (os.cpu_count() or 1)
    mem_bytes = int((mem_gb if mem_gb else _default_mem_gb()) * (1 << 30))

    if token_dtype == "u16":
        alphabet = 65536
        default_sep = 0xFFFF
    else:
        if vocab_size is None:
            raise ValueError("u32 indexes need an explicit vocab_size")
        alphabet = vocab_size + 1
        default_sep = vocab_size
    if insert_sep and doc_sep_token is None:
        doc_sep_token = default_sep

    os.makedirs(output, exist_ok=True)
    cfg = None
    if resume:
        try:
            cfg = IndexConfig.load(output)
        except (FileNotFoundError, ValueError):
            cfg = None

    # -- step 1: ingest -----------------------------------------------------
    if cfg is not None and "ingest" in cfg.completed:
        _log(verbose, "step 1/2 ingest: already done (%s tokens, %s shards)"
             % (f"{cfg.total_tokens:,}", cfg.num_shards))
    else:
        paths = readers.expand_inputs(inputs, pattern)
        _log(verbose, "step 1/2 ingest: %d input file(s), format=%s" % (len(paths), fmt))
        known = readers.count_tokens(paths, fmt, dtype)
        if known:
            _log(verbose, "  source holds %s tokens (%s)"
                 % (f"{known:,}", _fmt_bytes(known * dtype.itemsize)))

        # When the source already carries a separator we only need to *find* it;
        # inserting one is for sources that have no document structure of their
        # own.  Either way, an n-gram can only cross a boundary if it contains
        # the separator token, and queries containing it are rejected.
        scan_for = doc_sep_token if not insert_sep else None
        if fmt == "jsonl":
            scan_for = None

        writer = _ShardWriter(output, dtype, shard_tokens, doc_sep_token, insert_sep, verbose)
        t0 = time.time()
        total = 0
        for tokens, doc_starts in _blocks(paths, fmt, dtype, scan_for, tokenizer, block_tokens):
            if limit_tokens is not None and total + tokens.size > limit_tokens:
                keep = limit_tokens - total
                if keep <= 0:
                    break
                tokens = tokens[:keep]
                doc_starts = doc_starts[doc_starts < keep]
            writer.add(tokens, doc_starts)
            total += int(tokens.size)
            if limit_tokens is not None and total >= limit_tokens:
                break
        shards = writer.close()
        dt = time.time() - t0
        ingested = sum(s["tokens"] for s in shards)
        _log(verbose, "step 1/2 ingest: %s tokens into %d shard(s) in %.1fs (%.1f M tok/s)"
             % (f"{ingested:,}", len(shards), dt, ingested / max(dt, 1e-9) / 1e6))

        cfg = IndexConfig(
            token_dtype=token_dtype,
            # Filled in after step 2 from libsais' symbol frequencies, which
            # saves a full pass over the corpus here.
            vocab_size=vocab_size or 0,
            doc_sep_token=doc_sep_token,
            sep_inserted=insert_sep,
            total_tokens=ingested,
            total_docs=sum(s["docs"] for s in shards),
            pivot_stride=pivot_stride,
            pivot_len=pivot_len,
            tokenizer=tokenizer_name,
            shards=shards,
            completed=["ingest"],
        )
        cfg.save(output)

    # -- step 2: suffix arrays ---------------------------------------------
    todo = []
    for i in range(cfg.num_shards):
        d = cfg.shard_dir(output, i)
        done = "sa.%04d" % i
        if resume and done in cfg.completed and os.path.exists(os.path.join(d, "table")):
            continue
        todo.append((i, d))

    if not todo:
        _log(verbose, "step 2/2 suffix arrays: already done")
    else:
        concurrent, threads = plan_concurrency(
            shard_tokens, cpus, mem_bytes, cfg.token_dtype, len(todo)
        )
        _log(
            verbose,
            "step 2/2 suffix arrays: %d shard(s), %d concurrent x %d thread(s), "
            "memory budget %s" % (len(todo), concurrent, threads, _fmt_bytes(mem_bytes)),
        )
        specs = [
            dict(
                shard_dir=d,
                token_dtype=cfg.token_dtype,
                alphabet=alphabet,
                threads=threads,
                pivot_stride=cfg.pivot_stride,
                pivot_len=cfg.pivot_len,
            )
            for _, d in todo
        ]
        t0 = time.time()
        results = run_shard_jobs(specs, concurrent, verbose)

        errors = [r for r in results if "error" in r]
        if errors:
            raise RuntimeError(
                "suffix array build failed for %d shard(s):\n%s"
                % (
                    len(errors),
                    "\n".join(
                        "  %s: %s\n%s"
                        % (r["shard_dir"], r["error"], r.get("traceback", ""))
                        for r in errors
                    ),
                )
            )
        dt = time.time() - t0
        built = sum(r["tokens"] for r in results)
        for (i, _), r in zip(todo, results):
            cfg.shards[i]["ptr_size"] = r["ptr_size"]
            cfg.completed.append("sa.%04d" % i)
        if not vocab_size:
            cfg.vocab_size = max(cfg.vocab_size,
                                 max(r["max_token"] for r in results) + 1)
        cfg.save(output)
        _log(verbose, "step 2/2 suffix arrays: %s tokens in %.1fs (%.1f M tok/s aggregate)"
             % (f"{built:,}", dt, built / max(dt, 1e-9) / 1e6))

    size = _dir_size(output)
    _log(verbose, "index ready at %s -- %s on disk, %.2f bytes/token"
         % (output, _fmt_bytes(size), size / max(cfg.total_tokens, 1)))
    return cfg


def _default_mem_gb() -> float:
    """Assume 80% of physical RAM is available, as the reference indexer does."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return 0.8 * pages * page / (1 << 30)
    except (ValueError, OSError, AttributeError):
        return 8.0


def _dir_size(root: str) -> int:
    total = 0
    for dirpath, _, names in os.walk(root):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total
