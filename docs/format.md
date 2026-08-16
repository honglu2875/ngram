# Index format `ngram-sa-v1`

An index is a directory. Everything in it is little-endian, and everything
except `config.json` is read by `mmap` at query time.

```
index/
  config.json
  shard.0000/
    tokenized      N * W bytes
    table          N * P bytes + 8
    offset         D * 8 bytes
    bucket         64-byte header + unigram table + sampled prefixes
  shard.0001/
  ...
```

`W` is the token width (2 for `u16`, 4 for `u32`), `N` the shard's token count,
`D` its document count, and `P` its pointer size.

## Sharding

A shard is an independent suffix array over a disjoint slice of the corpus.
Occurrence counts are additive across shards, so there is no merge step: a
corpus far larger than RAM is indexed by building one in-memory shard at a time.
The cost is that a query runs once per shard, so fewer, larger shards are better
for latency and more, smaller shards are better for build-time memory. Shard
boundaries always fall on document boundaries.

## `tokenized`

The raw token stream, `N` values of `W` bytes. Nothing is escaped or
transformed; if the source already marked documents (llm.c shards prefix each
document with GPT-2's token 50256), that stream is stored verbatim.

`config.json` records `doc_sep_token`. Any token sequence spanning two documents
necessarily contains it, so an n-gram that does not contain the separator can
never cross a boundary. The Python layer rejects queries that do contain it.

For sources with no document structure of their own, `--insert-sep` writes a
separator before each document — `0xFFFF` for `u16` (not a valid GPT-2 token) or
`vocab_size` for `u32`.

## `table` — the suffix array

`N` entries, each the **token index** of one suffix, packed little-endian into
`P` bytes. Entries are sorted by the suffix they point at, in numeric token
order.

`P = ceil(bits_needed(N) / 8)`: 4 bytes up to 4.29 B tokens per shard, 5 up to
1.1 T. Storing token indices rather than byte offsets saves a byte at every
scale that matters and removes a multiply from the innermost comparison loop.

The file carries **8 trailing zero bytes**. Reading an entry is one unaligned
8-byte load followed by a mask, and the padding keeps the load for the last
entry inside the mapping.

Numeric token order is a deliberate difference from the reference infini-gram
implementation, which sorts byte-wise over little-endian tokens. That order is
self-consistent but not numeric; sorting on tokens directly makes the next-token
distribution come out already sorted by token id and makes the unigram table a
plain index by token id. The consequence is that `table` files are not
interchangeable with published infini-gram indexes.

## `offset`

`D` uint64 token indices, one per document, giving the position where each
document starts (at its separator, when there is one). Used to clip retrieved
spans to their document and to report document ids.

## `bucket` — the search accelerator

Loaded into anonymous RAM at open time rather than mapped, so it does not
compete with page cache for the large files.

```
offset  size              field
0       8                 magic "NGRAMBK1"
8       4  uint32         token_width (2 or 4)
12      4  uint32         vocab_slots
16      4  uint32         pivot_len
20      4                 reserved
24      8  uint64         tok_cnt
32      8  uint64         pivot_stride
40      8  uint64         num_pivots
48      16                reserved
64      (vocab_slots+1)*8 uint64 unigram[]
...     num_pivots*pivot_len*W   token pivots[]
```

**`unigram`** is the prefix sum of libsais' symbol frequency table:
`[unigram[t], unigram[t+1])` is the rank range of every suffix beginning with
token `t`. It costs 512 KiB for a 16-bit alphabet and it comes out of suffix
array construction for free.

**`pivots`** holds the first `pivot_len` tokens of every `pivot_stride`-th
suffix, `num_pivots = ceil(N / pivot_stride)` of them. Suffixes shorter than
`pivot_len` are padded with token 0.

Zero-padding preserves the ordering. If a padded key compares greater than the
pattern at position `j`, then `j` must be inside the real suffix, because `0` is
never greater than a token; and if it compares less at `j`, the suffix is either
genuinely smaller at `j` or a proper prefix of the pattern, which is smaller
too. So pruning against padded keys is conservative, never wrong.

At the defaults (`stride=512`, `pivot_len=8`) this is `N/32` bytes: about
250 MB for 8 B tokens.

### What the accelerator buys

Without it, a search binary-searches the whole suffix array — roughly
`2 * log2(N)` probes, each a random read into `table` followed by a random read
into `tokenized`. At 4 B tokens that is about 64 probe pairs.

With it, the unigram table supplies the starting range for free, the pivot
search runs entirely in RAM down to a window of one stride, and only
`2 * log2(stride)` = 18 probes ever touch the mapped files — and those land
inside a ~2 KB window of `table`.

Measured on a 400 M-token index (see `ngram bench`): **2.6x** faster warm,
**3.8x** faster with page cache dropped. The gap widens with corpus size, since
the pivot phase absorbs the extra `log2(N)` levels.

The accelerator is strictly an optimisation. `set_accel(False)` disables it, and
the test suite asserts that answers are identical either way.

## `config.json`

```json
{
  "format": "ngram-sa-v1",
  "token_dtype": "u16",
  "vocab_size": 50257,
  "doc_sep_token": 50256,
  "sep_inserted": false,
  "total_tokens": 7900000000,
  "total_docs": 11320662,
  "pivot_stride": 512,
  "pivot_len": 8,
  "tokenizer": "gpt2",
  "shards": [{"tokens": 4000000000, "docs": 5736269, "ptr_size": 4}, ...],
  "completed": ["ingest", "sa.0000", "sa.0001"]
}
```

`completed` is what makes builds resumable: a step whose name is listed and
whose output exists is skipped on a re-run.

## Space

| file | bytes per token |
|---|---|
| `tokenized` | 2 (`u16`) |
| `table` | 4 up to 4.29 B tokens/shard, else 5 |
| `bucket` | ~0.03 |
| `offset` | 8 per document |

About **6 bytes per token** in total for a `u16` corpus in 4 B-token shards.
