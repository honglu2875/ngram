# ngram

Disk-backed **infini-gram** indexes over tokenized corpora: count any n-gram,
get next-token distributions, and query the *longest suffix that occurs at all*
— on corpora much larger than RAM.

Implements the index and query engine of
[Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)
as a memory-mapped suffix array, built with
[libsais](https://github.com/IlyaGrebnov/libsais) and queried from C++ through
Cython.

```python
from ngram import InfiniGram

idx = InfiniGram("index/fineweb-8B", load_tokenizer=True)
idx.count(idx.encode(" the United States"))        # 663_558
idx.ntd(idx.encode(" the capital of")).top(3)      # [(262, ...), (3999, ...), ...]
idx.infgram_prob(prompt_ids, next_id)              # backs off to the longest match
```

## Why a suffix array

The previous version of this library was a pointer trie: one heap node per
n-gram prefix, each holding an `unordered_map`, an atomic counter and an OpenMP
lock. That is 150–250 bytes per node and up to five nodes per token, so roughly
**1 KB of RAM per corpus token** — a billion tokens would have wanted a terabyte.
It also fixed `n` at construction time, capped the vocabulary at 65 535, and
never touched disk.

A suffix array stores the same information in **~6 bytes per token, on disk**,
answers queries for *any* `n` without rebuilding, and needs only a few hundred
megabytes resident regardless of corpus size.

## Install

```bash
pip install -e .                     # needs a C++17 compiler and OpenMP
pip install -e ".[dev,tokenizers]"   # plus pytest and tiktoken/transformers
pytest
```

## Build an index

The fast path is a directory of pre-tokenized `.bin` shards in llm.c / nanoGPT
layout (1024-byte header, magic `20240520`, little-endian `uint16` payload) —
for example [`quintic/fineweb-scaled-gpt2`](https://huggingface.co/datasets/quintic/fineweb-scaled-gpt2):

```bash
hf download quintic/fineweb-scaled-gpt2 --repo-type dataset \
    --include '8B/*' --local-dir ./fineweb

ngram build ./fineweb/8B -o index/fineweb-8B -f llmc \
    --pattern 'fineweb_train_*.bin' --tokenizer gpt2 --shard-tokens 4G
```

Other inputs: `-f npy`, `-f raw` (headerless token array), and `-f jsonl` with
`--tokenizer gpt2` to tokenize `.jsonl`/`.gz`/`.zst` documents on the way in.

Builds are **resumable** — re-running skips every finished step.

```bash
ngram info  index/fineweb-8B
ngram count index/fineweb-8B " the United States" --docs 3
ngram ntd   index/fineweb-8B " the capital of" --infinite
ngram bench index/fineweb-8B --infgram
ngram query index/fineweb-8B          # interactive
```

## Python API

```python
idx = InfiniGram("index/fineweb-8B", threads=0)   # 0 = one thread per core

idx.count([464, 3797])                 # occurrences of an n-gram, any length
idx.prob(prompt_ids, cont_id)          # -> (prob, prompt_count, cont_count)
idx.ntd(prompt_ids)                    # sparse next-token distribution
idx.longest_suffix(prompt_ids)         # -> (length, count)
idx.infgram_prob(prompt_ids, cont_id)  # -> (prob, suffix_len, ...)
idx.infgram_ntd(prompt_ids)
idx.search_docs(ids, maxnum=5)         # sampled occurrences with context
```

Batched entry points release the GIL and run on a persistent thread pool:

```python
counts = idx.count_batch([ids_a, ids_b, ...])            # ragged input is fine
probs, slens, pc, cc = idx.infgram_batch(tokens_2d)      # per-position scoring
indptr, token_ids, counts = idx.ntd_batch(tokens_2d)     # CSR, not dense
```

`ntd_batch` returns a ragged CSR structure rather than a
`(batch, seq, vocab)` tensor. A single `(8, 1024)` batch over GPT-2's vocabulary
would be 1.6 GB dense and is typically a few MB here.

## How it works

A shard is a token array plus its suffix array. Every occurrence of an n-gram
occupies one contiguous run of suffix-array ranks, so a count is two binary
searches, and extending an n-gram searches only inside its prefix's range.
Shards are independent and counts add, which is what lets a corpus larger than
RAM be indexed one in-memory piece at a time.

Three things make it fast:

**A two-level jump table** (`bucket`, held in RAM). Level one maps each token id
straight to its rank range, which comes out of suffix-array construction for
free. Level two samples the first 8 tokens of every 512th suffix — `N/32` bytes,
about 250 MB for 8 B tokens — so the top of every binary search runs in RAM and
only ~18 probes ever touch the mapped files. Measured: **2.6x** faster warm,
**3.8x** faster with page cache dropped, and the gap grows with corpus size.
It is strictly an optimisation; the tests assert answers are identical with it
switched off.

**Range reuse.** `prob(prompt, cont)` searches for the continuation inside the
prompt's own range instead of starting over. `infgram_batch` goes further: the
longest matching suffix can grow by at most one per position, so each position
starts from the previous answer and walks down only on failure — amortised
O(1) searches per token instead of O(log n) backoff searches.

**Batch-level parallelism.** One persistent pool, no thread creation on the
query path. Query throughput scales close to linearly with cores.

For the on-disk layout, see [`docs/format.md`](docs/format.md).

### Differences from the reference implementation

- Suffixes are sorted in **numeric token order** (libsais runs on the `uint16`
  array directly) rather than byte-wise over little-endian tokens. Next-token
  distributions come out already sorted and the unigram table is a plain index
  by token id. `table` files are therefore not interchangeable with published
  infini-gram indexes.
- **Shards are built in parallel processes**, each calling libsais, instead of
  an external merge sort. libsais' own OpenMP scaling is about 1.25x at 30
  threads because it is memory-bandwidth bound, whereas independent shards
  scale nearly linearly.
- The next-token divide-and-conquer is **iterative**. The reference spawns a
  `std::thread` at every recursion node — O(distinct · log range) thread
  creations per call.
- The in-RAM jump table has no counterpart in the reference, which begins every
  query with a full binary search over the whole suffix array.

## Measured

Indexing the 7.9 B-token FineWeb-edu prefix of `quintic/fineweb-scaled-gpt2`
(GPT-2 tokens, two 4 B-token shards) on a 30-core Xeon 8358 that was also
running other work:

**Build — 6.2 minutes end to end, 44.5 GiB, 6.04 bytes/token**

| step | time | rate |
|---|---|---|
| ingest 79 files | 25 s | 315 M tokens/s |
| suffix arrays (2 shards in parallel) | 343 s | 23 M tokens/s |

Peak memory was ~40 GB per shard process, and the two ran concurrently — so the
same machine would index roughly 30 B tokens per shard pass, and any corpus at
all given more passes.

**Query** — 20 000 8-token patterns drawn from the held-out validation shard,
30 threads:

| | with jump table | without | speedup |
|---|---|---|---|
| warm (page cache resident) | **0.48 µs** / 2.09 M queries/s | 1.13 µs / 0.88 M/s | 2.4x |
| cold (page cache dropped) | **397 µs** / 2 520 queries/s | 1 413 µs / 707/s | 3.6x |

**∞-gram scoring** of held-out text: **1.64 µs/token**, 611 k tokens/s. The
matching suffix averaged 5.6 tokens (median 4, max 203) and the mean probability
assigned to the token actually observed was 0.211.

Reproduce with `ngram bench <index> --infgram --probe <held-out .bin>`. Without
`--probe`, patterns are drawn from the indexed corpus itself, which measures
search cost on guaranteed hits but makes ∞-gram numbers meaningless — every
sequence matches almost end to end.

**Correctness** was checked against brute force at both scales: exhaustive
`Counter`-based oracles for counts, distributions and ∞-gram on small corpora,
and on the 7.9 B index, all 50 100 distinct token counts against a `bincount`
over the stored corpus plus exact full-corpus n-gram counts.

## Limits

- Vocabulary up to 65 535 with `u16`; use `--token-dtype u32 --vocab-size N`
  beyond that, at a higher build-time memory cost (libsais has no 32-bit-alphabet
  entry point with 64-bit indices, so the input is widened to `int64`).
- A shard may hold at most 2^40 tokens; corpora beyond that need more shards.
- Queries may not contain the document separator token, since such an n-gram
  would span documents. This is rejected with a clear error.
- Shards are searched serially within one query, so query latency grows with
  shard count. Prefer the largest `--shard-tokens` your build-time memory allows.

## Licence

Apache 2.0. Vendors [libsais](https://github.com/IlyaGrebnov/libsais) 2.10.4
(Apache 2.0) under `csrc/libsais/`.
