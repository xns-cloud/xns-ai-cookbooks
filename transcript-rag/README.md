# Transcript RAG — Cached Speech-to-Text Retrieval for Audio & Video

Transcribe the audio track of media files stored in XNS, embed the
transcripts, and query them with natural language — with both transcripts and
embeddings cached in the bucket so re-runs only pay for what changed.

**What this is not:** video understanding. The pipeline extracts speech and
retrieves over text. It cannot answer "what was on the slide at 14:32" —
there is no frame sampling, OCR, or visual embedding here. If you need
visual retrieval, this recipe is the audio half of that system, not the
whole of it.

This is a starter recipe: single process, in-memory index, happy path. The
[Limitations](#limitations) section says exactly where that stops being
enough.

## Architecture

```
    ┌──────────────────┐
    │    XNS Bucket    │  your media files
    │  (S3 API)        │  no per-read charge
    └────────┬─────────┘
             │  XNSBlobLoader
             ▼
    ┌──────────────────┐      ┌────────────────────────┐
    │  Transcription   │─────▶│  Transcript cache      │
    │  gpt-4o-mini-    │      │  cache/transcripts/    │
    │  transcribe      │      │  key: ETag + model     │
    └────────┬─────────┘      └────────────────────────┘
             │                  unchanged file + same
             ▼                  model = no API call
    ┌──────────────────┐
    │  Text Splitter   │  500-char chunks, 50 overlap
    │  (LangChain)     │  (tutorial default — tune it)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐      ┌────────────────────────┐
    │  Embeddings      │─────▶│  Embedding cache       │
    │  text-embedding- │      │  cache/embeddings/     │
    │  3-small         │      │  key: chunk hash+model │
    └────────┬─────────┘      └────────────────────────┘
             │                  unchanged chunk = no
             ▼                  API call
    ┌──────────────────┐
    │  FAISS Index     │  in-memory; rebuilt from
    │  (local)         │  cached vectors each run
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  gpt-4.1-mini    │  answers from retrieved
    │  (ChatOpenAI)    │  context; says so when the
    └──────────────────┘  context doesn't cover it
```

### What the caches actually save

Be precise about this, because the caches save different things:

| You change… | Re-transcribe? | Re-embed? | Storage cost |
|---|---|---|---|
| Nothing (plain re-run) | No | No | $0 |
| Chunking strategy | No (transcript cached) | Yes — new chunks are new cache keys | $0 |
| Embedding model | No | Yes — new vector space, full re-embed | $0 |
| Transcription model | Yes — full re-transcribe | Yes | $0 |
| A media file (new ETag) | That file only | That file's chunks only | $0 |

"$0" is the storage side: XNS has no per-read charge, so cache reads and
media re-reads are free. The OpenAI columns are real money — transcription
is billed per audio minute, embeddings and chat per token. The caches exist
to make you pay those *once per change*, not once per run.

The FAISS index is rebuilt in memory each run from cached vectors. That is
seconds for a small corpus; persist it locally (`index.save_local`) when
rebuild time starts to matter.

## Prerequisites

1. **A running XNS Relayer** with `~/.xns/credentials` written.
   See the [repo-level prerequisites](../README.md#prerequisites).

2. **An OpenAI API key** — transcription, embeddings, and the chat model
   all use it.

3. **Media in a bucket** — upload some `.mp3`, `.wav`, `.m4a`, or `.mp4`
   files. Any short audio clip works for a first run:

   ```bash
   # AWS CLI pointed at your Relayer
   aws s3 mb s3://media --endpoint-url http://localhost:9000
   aws s3 cp meeting-recording.mp3 s3://media/ --endpoint-url http://localhost:9000
   ```

## Run it

```bash
cd transcript-rag
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python pipeline.py media "What decisions were made in the meeting?"
```

First run: transcribes, embeds, caches both, answers.
Re-run: transcript and embedding cache hits skip those API calls; the query
embedding and the answer call still happen — those are per-question costs.

## Data flow — what leaves your infrastructure

Raw media bytes, transcript text, chunk text, and your questions are all
sent to OpenAI's API. If your recordings contain material you cannot send
to a third-party API — customer data, legal content, anything under a
recording-consent regime — swap the OpenAI calls for a locally hosted
transcription and embedding stack before using this pipeline. The XNS side
(media, caches) stays on your own Relayer.

Deleting a media file does not delete its cached transcript, cached
embeddings, or index entries. If deletion needs to propagate, you own that
workflow — the cache keys are derivable from the object's ETag.

## MCP config

To let an AI assistant manage your Relayer — create buckets, upload files —
add this to your Claude Desktop or Cursor MCP config:

```json
{
  "mcpServers": {
    "xns-relayer": {
      "command": "npx",
      "args": ["@xns-cloud/relayer-mcp@latest"]
    }
  }
}
```

The MCP server authenticates via OIDC and writes `~/.xns/credentials`
during setup, which `pipeline.py` reads automatically.

## Limitations

Starter-recipe boundaries, stated so you don't discover them in production:

- **No visual analysis.** Speech only. Silent video transcribes to nothing.
- **No timestamps or speaker labels in retrieval.** The answer can't cite
  who said something or when. `gpt-4o-transcribe-diarize` returns
  timestamped, speaker-labeled segments — but aligning and chunking those
  is a real design problem (segment boundaries, overlap handling), not a
  drop-in swap.
- **Character-based chunking.** 500/50 is a tutorial default. It will split
  speaker turns and topics at arbitrary boundaries. Transcript-aware
  segmentation retrieves better; measure against your own queries.
- **Single-process, in-memory index.** No concurrency, no restart recovery,
  no multi-user story. Fine for a workstation and a few hundred recordings;
  a service needs a real vector store.
- **Happy path only.** No handling for oversized files, unsupported codecs,
  API rate limits, or partial failures.

## What to try next

- **Persist the FAISS index** (`save_local`/`load_local`) so a fresh process
  skips the rebuild.
- **Try diarization.** Run `gpt-4o-transcribe-diarize` on the same media —
  another free read from XNS — and look at the segment structure before
  deciding how to chunk it.
- **Re-chunk and measure.** Change the splitter settings and re-run: the
  transcript cache makes the experiment cost embeddings only. Keep a small
  set of known-answer questions and compare retrieval before/after.
