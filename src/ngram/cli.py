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

"""``ngram`` command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f PiB" % n


def _human_int(x) -> str:
    return f"{int(x):,}"


def _parse_size(s: str) -> int:
    """Accept ``4e9``, ``4_000_000_000``, ``4G``, ``500M``."""
    s = str(s).strip().replace("_", "")
    mult = 1
    if s and s[-1].upper() in "KMGT":
        mult = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}[s[-1].upper()]
        s = s[:-1]
    return int(float(s) * mult)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def cmd_build(args) -> int:
    from .build import build_index
    from .readers import load_tokenizer

    tokenizer = None
    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)

    doc_sep = args.doc_sep_token
    insert = args.insert_sep
    if args.format == "llmc" and doc_sep is None and not insert:
        # llm.c shards already prefix every document with the GPT-2 end-of-text
        # token, so the boundaries are there to be found rather than added.
        doc_sep = 50256
    if args.format == "jsonl" and not insert and doc_sep is None:
        insert = True

    build_index(
        args.input,
        args.output,
        fmt=args.format,
        token_dtype=args.token_dtype,
        vocab_size=args.vocab_size,
        doc_sep_token=doc_sep,
        insert_sep=insert,
        shard_tokens=_parse_size(args.shard_tokens),
        cpus=args.cpus,
        mem_gb=args.mem,
        pivot_stride=args.pivot_stride,
        pivot_len=args.pivot_len,
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
        pattern=args.pattern,
        limit_tokens=_parse_size(args.limit) if args.limit else None,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )
    return 0


# ---------------------------------------------------------------------------
# query helpers
# ---------------------------------------------------------------------------


def _open(args, threads: Optional[int] = None):
    from .index import InfiniGram

    return InfiniGram(
        args.index,
        threads=threads if threads is not None else getattr(args, "threads", 0),
        load_tokenizer=True,
    )


def _to_ids(idx, args) -> np.ndarray:
    if getattr(args, "tokens", None):
        return np.array(
            [int(x) for x in args.tokens.replace(",", " ").split()], dtype=idx.dtype
        )
    text = " ".join(args.text) if isinstance(args.text, list) else args.text
    if not text:
        raise SystemExit("give either --tokens or some text")
    return idx.encode(text)


def _show(idx, ids: np.ndarray) -> str:
    try:
        return repr(idx.decode(ids))
    except Exception:
        return str(list(int(i) for i in ids))


def cmd_info(args) -> int:
    idx = _open(args)
    cfg = idx.config
    print("index      %s" % idx.path)
    print("tokens     %s" % _human_int(idx.tok_cnt))
    print("documents  %s" % _human_int(idx.doc_cnt))
    print("shards     %d" % idx.num_shards)
    print("dtype      %s (vocab %s)" % (cfg.token_dtype, _human_int(cfg.vocab_size)))
    print("separator  %s%s" % (cfg.doc_sep_token,
                               " (inserted)" if cfg.sep_inserted else " (from source)"))
    print("tokenizer  %s" % (cfg.tokenizer or "-"))
    print("pivots     stride %d, %d tokens" % (cfg.pivot_stride, cfg.pivot_len))
    size = 0
    for dp, _, fs in os.walk(idx.path):
        for f in fs:
            size += os.path.getsize(os.path.join(dp, f))
    print("on disk    %s (%.2f bytes/token)" % (_fmt_bytes(size), size / max(idx.tok_cnt, 1)))
    for i, s in enumerate(cfg.shards):
        print("  shard %04d  %14s tokens  %10s docs  %d-byte pointers"
              % (i, _human_int(s["tokens"]), _human_int(s["docs"]), s["ptr_size"]))
    return 0


def cmd_count(args) -> int:
    idx = _open(args)
    ids = _to_ids(idx, args)
    n = idx.count(ids)
    print("%s  ->  %s occurrence(s) in %s tokens" % (_show(idx, ids), _human_int(n),
                                                     _human_int(idx.tok_cnt)))
    if n and args.docs:
        for occ in idx.search_docs(ids, maxnum=args.docs, context=args.context):
            print("-" * 72)
            print("shard %d, doc %d, offset %d" % (occ.shard, occ.doc_id, occ.offset_in_doc))
            print(_show(idx, occ.tokens))
    return 0


def cmd_ntd(args) -> int:
    idx = _open(args)
    ids = _to_ids(idx, args)
    res = (idx.infgram_ntd(ids, max_support=args.max_support)
           if args.infinite else idx.ntd(ids, max_support=args.max_support))
    if res.suffix_len >= 0:
        print("longest matching suffix: %d of %d tokens" % (res.suffix_len, len(ids)))
    print("context occurs %s time(s); %d distinct continuations"
          % (_human_int(res.prompt_count), len(res.token_ids)))
    for tok, cnt, p in res.top(args.top):
        try:
            label = repr(idx.decode([tok]))
        except Exception:
            label = str(tok)
        print("  %8d  %-24s %10s  %.6f" % (tok, label, _human_int(cnt), p))
    return 0


def cmd_infgram(args) -> int:
    idx = _open(args)
    ids = _to_ids(idx, args)
    if len(ids) < 2:
        raise SystemExit("need at least two tokens: a prompt and a continuation")
    prompt, cont = ids[:-1], int(ids[-1])
    r = idx.infgram_prob(prompt, cont)
    print("prompt      %s" % _show(idx, prompt))
    print("continuation %s" % _show(idx, [cont]))
    print("suffix used %d of %d tokens" % (r.suffix_len, len(prompt)))
    print("counts      %s / %s" % (_human_int(r.cont_count), _human_int(r.prompt_count)))
    print("probability %.6g" % r.prob)
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def _probe_tokens(idx, probe: Optional[str]) -> np.ndarray:
    """The token stream that benchmark queries are drawn from.

    Default is the indexed corpus, so every pattern is guaranteed to occur --
    the right choice for measuring search cost, since a miss can exit early.
    Pass ``--probe`` with a held-out file (for instance the validation shard
    shipped alongside a training split) to measure what queries from unseen
    text actually cost, which is what infinite-gram latency depends on.
    """
    if probe:
        from .readers import LLMC_HEADER_BYTES, llmc_token_count

        ntok = llmc_token_count(probe)
        return np.memmap(probe, dtype=np.uint16, mode="r",
                         offset=LLMC_HEADER_BYTES, shape=(ntok,))
    d = idx.config.shard_dir(idx.path, 0)
    return np.memmap(os.path.join(d, "tokenized"), dtype=idx.config.dtype, mode="r")


def _random_patterns(tokens, n: int, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(len(tokens) - length - 1, 1), size=n)
    return np.stack([np.asarray(tokens[s : s + length]) for s in starts])


def cmd_bench(args) -> int:
    idx = _open(args, threads=args.threads)
    probe = _probe_tokens(idx, args.probe)
    pats = _random_patterns(probe, args.queries, args.length, args.seed)
    n = pats.shape[0]

    def timed(label: str) -> None:
        idx._engine.count_batch(pats[: min(n, 2000)])  # warm the code paths
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            idx._engine.count_batch(pats)
            times.append(time.perf_counter() - t0)
        dt = min(times)
        print("  %-28s %8.2f us/query  %12s queries/s"
              % (label, dt / n * 1e6, _human_int(n / dt)))

    print("index %s: %s tokens, %d shard(s), %d thread(s)"
          % (idx.path, _human_int(idx.tok_cnt), idx.num_shards, idx.num_threads))
    print("queries: %d patterns of %d tokens, drawn from %s\n"
          % (n, args.length, args.probe if args.probe else "the indexed corpus"))

    if not args.cold_only:
        print("warm (index resident in page cache):")
        t0 = time.perf_counter()
        idx.warm()
        print("  loaded in %.1fs" % (time.perf_counter() - t0))
        for accel in (True, False):
            idx.set_accel(accel)
            timed("accel=%s" % accel)

    if not args.warm_only:
        print("\ncold (page cache dropped before each run):")
        cold_n = min(n, args.cold_queries)
        for accel in (True, False):
            idx.set_accel(accel)
            idx.evict()
            t0 = time.perf_counter()
            idx._engine.count_batch(pats[:cold_n])
            dt = time.perf_counter() - t0
            print("  %-28s %8.2f us/query  %12s queries/s"
                  % ("accel=%s" % accel, dt / cold_n * 1e6, _human_int(cold_n / dt)))
        idx.set_accel(True)

    if args.infgram:
        print("\ninfinite-gram scoring (%s):"
              % ("held-out text" if args.probe else "text taken from the corpus"))
        rows, cols = args.infgram_rows, args.infgram_cols
        batch = _random_patterns(probe, rows, cols, args.seed + 1)
        idx.warm()
        t0 = time.perf_counter()
        probs, slens, pc, _ = idx.infgram_batch(batch, max_len=args.max_len)
        dt = time.perf_counter() - t0
        total = rows * cols
        print("  %s positions in %.2fs -> %.2f us/token, %s tokens/s"
              % (_human_int(total), dt, dt / total * 1e6, _human_int(total / dt)))
        print("  matching suffix length: mean %.1f, median %d, max %d"
              % (slens.mean(), int(np.median(slens)), slens.max()))
        print("  mean probability of the observed token: %.4f" % probs.mean())
    return 0


# ---------------------------------------------------------------------------
# repl
# ---------------------------------------------------------------------------


def cmd_query(args) -> int:
    idx = _open(args)
    print(idx)
    print("Type text to count it. Prefix with '?' for the next-token distribution,")
    print("'!' for the infinite-gram probability of the last token. Ctrl-D to exit.")
    while True:
        try:
            line = input("ngram> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        try:
            if line.startswith("?"):
                ids = idx.encode(line[1:].strip())
                res = idx.infgram_ntd(ids)
                print("  suffix %d, %s occurrences" % (res.suffix_len,
                                                       _human_int(res.prompt_count)))
                for tok, cnt, p in res.top(10):
                    print("    %-24s %10s %.6f" % (repr(idx.decode([tok])), _human_int(cnt), p))
            elif line.startswith("!"):
                ids = idx.encode(line[1:].strip())
                if len(ids) < 2:
                    print("  need at least two tokens")
                    continue
                r = idx.infgram_prob(ids[:-1], int(ids[-1]))
                print("  suffix %d, p = %.6g (%s/%s)"
                      % (r.suffix_len, r.prob, _human_int(r.cont_count),
                         _human_int(r.prompt_count)))
            else:
                ids = idx.encode(line)
                print("  %s occurrence(s)" % _human_int(idx.count(ids)))
        except Exception as exc:
            print("  error: %s" % exc)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ngram", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build an index from tokenized files")
    b.add_argument("input", nargs="+", help="files, directories or globs")
    b.add_argument("-o", "--output", required=True, help="index directory to create")
    b.add_argument("-f", "--format", default="llmc",
                   choices=["llmc", "raw", "npy", "jsonl"],
                   help="llmc: 1024-byte header + uint16 payload (default)")
    b.add_argument("--pattern", help="glob applied inside input directories")
    b.add_argument("--token-dtype", default="u16", choices=["u16", "u32"])
    b.add_argument("--vocab-size", type=int, help="required for u32 indexes")
    b.add_argument("--doc-sep-token", type=int,
                   help="token that already marks document starts in the source")
    b.add_argument("--insert-sep", action="store_true",
                   help="insert a separator before each document instead")
    b.add_argument("--shard-tokens", default="4G",
                   help="max tokens per shard (default 4G)")
    b.add_argument("--limit", help="stop after this many tokens")
    b.add_argument("--tokenizer", help="tokenizer name, required for jsonl input")
    b.add_argument("--cpus", type=int, help="cores available (default: all)")
    b.add_argument("--mem", type=float, help="memory budget in GiB (default: 80%% of RAM)")
    b.add_argument("--pivot-stride", type=int, default=512,
                   help="sample one suffix prefix per this many ranks (default 512)")
    b.add_argument("--pivot-len", type=int, default=8,
                   help="tokens kept per sampled prefix (default 8)")
    b.add_argument("--no-resume", action="store_true", help="rebuild from scratch")
    b.add_argument("-q", "--quiet", action="store_true")
    b.set_defaults(func=cmd_build)

    def q(name, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("index")
        s.add_argument("text", nargs="*", help="text to tokenize and query")
        s.add_argument("--tokens", help="token ids instead of text, e.g. '464,3797'")
        s.add_argument("--threads", type=int, default=0)
        return s

    i = sub.add_parser("info", help="show index statistics")
    i.add_argument("index")
    i.add_argument("--threads", type=int, default=0)
    i.set_defaults(func=cmd_info)

    c = q("count", "count occurrences of an n-gram")
    c.add_argument("--docs", type=int, default=0, help="also show this many occurrences")
    c.add_argument("--context", type=int, default=48, help="tokens of context per occurrence")
    c.set_defaults(func=cmd_count)

    n = q("ntd", "next-token distribution after a context")
    n.add_argument("--top", type=int, default=15)
    n.add_argument("--max-support", type=int, default=0,
                   help="approximate above this many occurrences (0 = exact)")
    n.add_argument("--infinite", action="store_true",
                   help="back off to the longest matching suffix")
    n.set_defaults(func=cmd_ntd)

    g = q("infgram", "infinite-gram probability of the final token")
    g.set_defaults(func=cmd_infgram)

    r = sub.add_parser("query", help="interactive prompt")
    r.add_argument("index")
    r.add_argument("--threads", type=int, default=0)
    r.set_defaults(func=cmd_query)

    m = sub.add_parser("bench", help="measure query latency and throughput")
    m.add_argument("index")
    m.add_argument("--queries", type=int, default=20000)
    m.add_argument("--cold-queries", type=int, default=2000)
    m.add_argument("--length", type=int, default=8, help="tokens per query pattern")
    m.add_argument("--threads", type=int, default=0)
    m.add_argument("--repeats", type=int, default=3)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--warm-only", action="store_true")
    m.add_argument("--cold-only", action="store_true")
    m.add_argument("--probe", help="draw query patterns from this held-out llm.c "
                                   ".bin instead of the indexed corpus")
    m.add_argument("--infgram", action="store_true", help="also time infinite-gram scoring")
    m.add_argument("--infgram-rows", type=int, default=64)
    m.add_argument("--infgram-cols", type=int, default=512)
    m.add_argument("--max-len", type=int, default=0,
                   help="cap the matching suffix length (0 = unbounded)")
    m.set_defaults(func=cmd_bench)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print("ngram: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
