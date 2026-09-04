# Persistent Vector Index: Qdrant snapshots stored in XNS

The other four recipes in this repository build their index in memory and
lose it when the process exits. Re-running means re-embedding the whole
corpus, and nothing carries to a second machine.
[`multimodal-rag`](../multimodal-rag/) says so about itself.

This recipe removes that ceiling. Qdrant holds the vectors. After ingest,
the collection is snapshotted and the snapshot file is stored in your XNS
bucket alongside the documents it was built from. A Qdrant that has never
seen the collection — new container, empty volume, different host — pulls
that snapshot back and serves queries without re-embedding the corpus.

```yaml
recipe:
  schema_version: 1
  name: persistent-index
  purpose: Keep a vector index across processes, containers and machines
  engine: Qdrant 1.19.0 (one container)
  entrypoint: index.py
  commands:
    - ingest      # load, chunk, embed, upsert, snapshot to the bucket
    - query       # ask; restores first if the collection is missing
    - cold-start  # delete the collection, restore it, then ask
    - status      # what Qdrant holds, and what the bucket holds
  inputs:
    - .txt and .md objects in the bucket
  artifacts:
    - key: index/<collection>-<id>-<timestamp>.snapshot
      producer: ingest
      consumer: query, cold-start
      format: Qdrant collection snapshot
    - key: index/latest.json
      producer: ingest
      consumer: query, cold-start
      format: JSON manifest — snapshot name, size, sha256, point count
  required_environment:
    - XNS_ENDPOINT
    - XNS_ACCESS_KEY_ID
    - XNS_SECRET_ACCESS_KEY
    - OPENAI_API_KEY   # default embedding path only; not needed with XNS_EMBEDDINGS=hashing
  optional_environment:
    - QDRANT_URL        # default http://localhost:6333
    - XNS_COLLECTION    # default corpus
    - XNS_EMBEDDINGS    # openai (default) | hashing
  contract:
    - The bucket, not the Qdrant volume, is what the index survives on
    - A snapshot is verified against a sha256 in the manifest before it is restored
  concurrency:
    model: single writer
    rule: One ingest at a time; the manifest names one current snapshot
    conflict_behavior: Last ingest wins — a concurrent one overwrites the manifest
  non_goals:
    - a managed or highly available Qdrant deployment
    - incremental or streaming index updates
    - multi-tenant collections
    - benchmarking, or any cost estimate
```

## Architecture

```
   documents in XNS
   ┌──────────────────┐
   │ docs/*.md        │
   └────────┬─────────┘
            │  XNSBlobLoader
            ▼
        chunk ──▶ embed ──▶ ┌────────────────────────┐
                            │  Qdrant collection     │
                            │  (container, on disk)  │
                            └───────────┬────────────┘
                                        │ create_snapshot
                                        │ + GET the file out
                                        ▼
                            ┌────────────────────────┐
                            │  XNS bucket            │
                            │  index/*.snapshot      │
                            │  index/latest.json     │
                            └───────────┬────────────┘
                                        │ POST /snapshots/upload
                                        ▼
                            ┌────────────────────────┐
                            │  a cold Qdrant         │
                            │  — new container       │
                            │  — empty volume        │
                            │  — different host      │
                            └────────────────────────┘
```

The Qdrant volume is disposable. The bucket is not. That inversion is the
whole recipe: `docker compose down -v` throws the index away and the next
query rebuilds it from object storage rather than from the embedding
provider.

Storage reads carry no per-read charge under the product model this
repository assumes, which is what makes "restore on every cold start"
a reasonable default rather than something to avoid. Embedding and
generation are billed by your model provider as usual.

## Prerequisites

1. A running XNS Relayer with `~/.xns/credentials` written — see the
   [repo-level prerequisites](../README.md#prerequisites).
2. Docker, for the one Qdrant container.
3. `pip install -r requirements.txt`.
4. Some `.txt` or `.md` objects in a bucket.
5. An embedding provider. The default is OpenAI's
   `text-embedding-3-small` and reads `OPENAI_API_KEY`. See
   [Running without an embedding account](#running-without-an-embedding-account)
   if you want to exercise the storage path first.

## Quickstart

```bash
cd persistent-index
pip install -r requirements.txt
docker compose up -d --wait          # Qdrant, ~10s including the pull

export XNS_ENDPOINT=http://localhost:9000
export XNS_ACCESS_KEY_ID=...  XNS_SECRET_ACCESS_KEY=...
export OPENAI_API_KEY=sk-...

python index.py ingest my-bucket
python index.py cold-start my-bucket "what does the corpus say about X?"
```

`cold-start` deletes the local collection before it answers. That is
deliberate: without it, a warm Qdrant answers from memory and the restore
path is never exercised, so the recipe would demonstrate nothing.

**Runtime.** This recipe does not meet the under-60-seconds rule the other
four hold to, and it should not pretend to. A container pull plus a
snapshot round trip does not fit in a minute. The Qdrant image pull is the
long pole on a first run; after that, ingest is dominated by the embedding
call and the snapshot round trip is well under a second for a small corpus.
Real numbers below.

## Verified run

Against a pre-release gateway and Qdrant 1.19.0 from the compose file
above, three small Markdown documents, using the local hashing embedder so
the numbers are storage and nothing else:

```
$ docker compose up -d --wait
 Container persistent-index-qdrant-1  Healthy

$ python index.py ingest cookbook-index-check
embeddings: hashing (local, deterministic, not semantic)
3 document(s) from s3://cookbook-index-check
3 chunks
embedded in 0.0s
upserted 3 points into 'corpus'
snapshot corpus-7467990003043846-2026-09-02-13-21-54.snapshot -> s3://cookbook-index-check/index/corpus-7467990003043846-2026-09-02-13-21-54.snapshot (155,648 bytes, 3 points, 0.4s)

$ python index.py cold-start cookbook-index-check "what makes the index survive losing the disk?"
deleted local collection 'corpus' -- Qdrant is now cold
restored corpus-7467990003043846-2026-09-02-13-21-54.snapshot from s3://cookbook-index-check/index/corpus-7467990003043846-2026-09-02-13-21-54.snapshot -> 3 points (0.4s)
embeddings: hashing (local, deterministic, not semantic)

Q: what makes the index survive losing the disk?

retrieved 3 chunks:
  [0.545] docs/snapshots.md: # Snapshots  A Qdrant snapshot is a point-in-time copy of a collection, written to the node's own disk. Downlo…
  [0.371] docs/cold-start.md: # Cold start  A cold start is a process, container, or machine that has never seen the index. It restores the …
  [0.307] docs/embeddings.md: # Embeddings and cost  Embedding a corpus is the expensive step in a retrieval pipeline. Every re-run that reb…
```

The stronger check, run separately on a second host: the container was
destroyed and a new one started with an empty volume, and a plain `query`
— not `cold-start` — restored and answered:

```
collection 'corpus' not present -- restoring from the bucket
restored corpus-…snapshot from s3://cookbook-index-check/index/… -> 3 points (0.8s)
```

The same cycle on the default embedding path, `text-embedding-3-small`,
including the generated answer:

```
$ python index.py ingest cookbook-index-check
embeddings: text-embedding-3-small
3 document(s) from s3://cookbook-index-check
3 chunks
embedded in 2.7s
upserted 3 points into 'corpus'
snapshot corpus-7467990003043846-2026-09-02-13-23-47.snapshot -> s3://…/index/corpus-7467990003043846-2026-09-02-13-23-47.snapshot (176,128 bytes, 3 points, 0.3s)

$ python index.py cold-start cookbook-index-check "what makes the index survive losing the disk?"
deleted local collection 'corpus' -- Qdrant is now cold
restored corpus-7467990003043846-2026-09-02-13-23-47.snapshot from s3://…/index/corpus-… -> 3 points (0.3s)
embeddings: text-embedding-3-small

Q: what makes the index survive losing the disk?

retrieved 3 chunks:
  [0.411] docs/snapshots.md: # Snapshots  A Qdrant snapshot is a point-in-time copy of a collection, written to the node's own disk. Downlo…
  [0.315] docs/cold-start.md: # Cold start  A cold start is a process, container, or machine that has never seen the index. It restores the …
  [0.284] docs/embeddings.md: # Embeddings and cost  Embedding a corpus is the expensive step in a retrieval pipeline. Every re-run that reb…

A: Downloading a Qdrant snapshot and storing the file elsewhere makes the index survive the loss of that disk.
```

The 2.7 seconds of embedding against 0.3 seconds of snapshot round trip
is the ratio the recipe exists for, and it is the ratio that grows with
the corpus: embedding scales with how much text you have, while the
restore scales with the index and happens once per cold start rather than
once per run.

Two things in that output are worth reading carefully. The snapshot is
roughly 150 KB for 3 points: a Qdrant snapshot carries the collection's
segment structure, not just its vectors, so it has a floor that has
nothing to do with corpus size — and it varied between 100 KB and 176 KB
across runs of the same three documents. Do not extrapolate per-point
storage from a small run, and do not treat the size as stable.

## Running without an embedding account

```bash
XNS_EMBEDDINGS=hashing python index.py ingest my-bucket
XNS_EMBEDDINGS=hashing python index.py cold-start my-bucket "a question"
```

`hashing` is a deterministic bag-of-words embedder, local, no network, no
account. It exists so you can prove the part this recipe is actually about
— upsert, snapshot, store, restore, retrieve — against your own gateway
before spending anything.

**It is a test fixture, not a model.** It retrieves on term overlap and
understands nothing; a query that shares no words with a relevant chunk
will not find it. The script refuses to generate an answer on this path,
because an answer built on term-overlap retrieval would misrepresent what
the recipe does. Never ingest a real corpus with it — and if you switch
embedders, re-run `ingest`, since the vector dimensions and the meaning of
the space both change.

## How the round trip works

**Snapshot out.** `create_snapshot` writes the file to Qdrant's own disk,
which is the disk this recipe assumes you can lose. The script then GETs
it back over the REST API, stores the bytes in the bucket, and deletes
Qdrant's copy — leaving it would grow that disk on every ingest for no
benefit. A manifest at `index/latest.json` records the snapshot name,
size, sha256, point count and which embedder produced it.

**Snapshot back.** The bytes are POSTed to
`/collections/{name}/snapshots/upload` rather than handed to Qdrant as a
URL. `recover_snapshot(location=...)` would make Qdrant fetch the file
itself, which requires the Qdrant container to reach your gateway — true
on a laptop, often false in a cluster where the vector database sits on a
network the object store does not. Uploading the bytes works in both.

**Before restoring**, the downloaded blob is checked against the sha256 in
the manifest. Recovery overwrites the collection, so a truncated or
corrupted download would replace a working index with a broken one. On a
mismatch the script stops and restores nothing.

**Behavior worth knowing before you build on it:**

- `ingest` deletes and recreates the collection. A re-ingest is a rebuild;
  stale points from a previous corpus are the subtler failure.
- The manifest names exactly one current snapshot. Older snapshot objects
  are left in the bucket and nothing prunes them — that retention decision
  is yours.
- `query` restores only if the collection is *missing*. It does not
  compare a present collection against the manifest, so a Qdrant holding a
  stale collection will answer from it. `status` will show you the
  disagreement; `cold-start` forces the restore.
- Two concurrent ingests will both write `index/latest.json`. Last write
  wins, and the loser's snapshot object is orphaned in the bucket.

## Configuration

Credentials resolve from `XNS_ENDPOINT` / `XNS_ACCESS_KEY_ID` /
`XNS_SECRET_ACCESS_KEY`, falling back to `~/.xns/credentials` — the same
zero-config read every other recipe here uses, via `langchain-xns`.

| Variable | Default | What it does |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Where Qdrant is |
| `XNS_COLLECTION` | `corpus` | Collection name, and the snapshot prefix |
| `XNS_EMBEDDINGS` | `openai` | `openai` or `hashing` |

The compose file pins `qdrant/qdrant:v1.19.0` and `requirements.txt` pins
`qdrant-client>=1.19,<2`. Keep them on the same minor version — the client
warns and may refuse when the server drifts more than one minor away.

Qdrant is started here with no API key and no TLS, reachable on
`localhost`. That is a local development posture. Anything beyond your own
machine needs an API key, TLS, and a network policy, none of which this
compose file sets up for you.

## Why Qdrant

One container, and snapshotting is a first-class collection operation
rather than a filesystem copy you have to reason about. If you already run
PostgreSQL, `pgvector` is the better trade — one fewer datastore, at the
cost of the native snapshot path this recipe leans on, since you would be
persisting through your existing database backups instead.

## Limitations

- **Single node, single writer.** No replication, no sharding, no
  high-availability story. One ingest at a time.
- **Snapshot granularity is the whole collection.** There is no
  incremental update path here: changing one document means re-ingesting
  and re-snapshotting the corpus.
- **The restore overwrites.** `priority=snapshot` means the uploaded
  snapshot is the source of truth and anything in the collection is
  replaced.
- **A query still embeds the query.** The corpus is not re-embedded, which
  is the saving. One embedding call per question remains, and that is
  worth being precise about rather than rounding to "no API calls".
- **No pruning.** Snapshot objects accumulate in the bucket.
- **Local development posture.** No API key, no TLS, no network policy.
- **The hashing embedder is a fixture.** It is for exercising the storage
  path, not for retrieval quality.
- **Nothing here is a benchmark.** The numbers above are one run, one
  corpus, one machine, one gateway.

**Compatibility note (2026-09-02):** the snapshot download, the multipart
snapshot upload, and the cold-container restore were exercised against a
current pre-release gateway build and Qdrant 1.19.0 during development.
Treat these as behaviors confirmed on those builds rather than as
guaranteed properties; re-check against the versions you run.

## Tests

```bash
python test_index.py     # offline: no Qdrant, no gateway, no provider
```

Covers the embedder's determinism and normalization and the corruption
guard. It does **not** cover the round trip — that needs a real Qdrant and
a real bucket, and it is what the [Verified run](#verified-run) section
records.

## What to try next

- **Run ingest on one machine and query from another** against the same
  bucket. Nothing in the script changes; only `QDRANT_URL` does.
- **Point the multimodal-rag or agentic-doc-parsing output at this index**
  instead of rebuilding FAISS in memory each run.
- **Keep snapshots per corpus version** by setting `XNS_COLLECTION` per
  build, so a bad ingest is one restore away from being undone.
- **Add a staleness check** to `query` that compares the live collection
  against the manifest, rather than only restoring when it is absent.
