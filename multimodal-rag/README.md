# Multimodal RAG — Speech + Visual Retrieval for Audio & Video

Query media files stored in XNS by what was **said** and what was **shown**.

Both tracks of every video are processed: the audio track is transcribed,
and frames sampled with ffmpeg are captioned by a vision model — on-screen
text, slide titles, chart labels read out into the captions. Transcript and
captions are timestamp-labeled, embedded into one index, and retrieved
together. "What was on the slide when they made the call" and "what did
they decide" are both answerable, with the file and timestamp in the
retrieved context.

Transcripts, captions, and embeddings are all cached back into the bucket,
keyed so a re-run only pays for what changed. This is a starter recipe:
single process, in-memory index, happy path. The
[Limitations](#limitations) section says exactly where that stops being
enough.

## Architecture

```
    ┌──────────────────────────────────────────┐
    │               XNS Bucket                 │
    │        media files  (S3 API,             │
    │         no per-read charge)              │
    └──────┬───────────────────────┬───────────┘
           │ audio track           │ video track
           ▼                       ▼
    ┌──────────────┐        ┌──────────────┐
    │ Transcription│        │ ffmpeg       │  1 frame / 10 s
    │ gpt-4o-mini- │        │ frame        │
    │ transcribe   │        │ sampling     │
    └──────┬───────┘        └──────┬───────┘
           │                       ▼
           │                ┌──────────────┐
           │                │ Vision       │  reads slides, charts,
           │                │ captions     │  UI state off each frame
           │                │ gpt-4.1-mini │
           │                └──────┬───────┘
           ▼                       ▼
    ┌─────────────────────────────────────────┐     ┌───────────────────┐
    │  Timestamp-labeled text                 │────▶│  XNS caches       │
    │  [file spoken] …   [file @ 1:40         │     │  cache/transcripts│
    │  on screen] …                           │     │  cache/captions   │
    └──────────────────┬──────────────────────┘     │  cache/embeddings │
                       ▼                            └───────────────────┘
    ┌─────────────────────────────────────────┐      unchanged input =
    │  Splitter → text-embedding-3-small      │      no repeat API call
    │  (cached) → FAISS (in-memory, local)    │
    └──────────────────┬──────────────────────┘
                       ▼
    ┌─────────────────────────────────────────┐
    │  gpt-4.1-mini answers from retrieved    │
    │  context; says so when it can't         │
    └─────────────────────────────────────────┘
```

### What the caches actually save

The caches save different things — be precise about which re-run costs what:

| You change… | Re-transcribe? | Re-caption frames? | Re-embed? | Storage cost |
|---|---|---|---|---|
| Nothing (plain re-run) | No | No | No | $0 |
| Chunking strategy | No | No | Yes — new chunks | $0 |
| Embedding model | No | No | Yes — full re-embed | $0 |
| Vision model or frame interval | No | Yes | Yes (captions changed) | $0 |
| Transcription model | Yes | No | Yes (transcripts changed) | $0 |
| A media file (new ETag) | That file | That file | That file's chunks | $0 |

"$0" is the storage side: XNS has no per-read charge, so cache reads and
media re-reads are free — including pulling the same multi-GB video again
to re-sample frames at a finer interval. The OpenAI columns are real money:
transcription per audio minute, captions per frame, embeddings and chat per
token. The caches make you pay those once per change, not once per run.

The FAISS index is rebuilt in memory each run from cached vectors — seconds
for a small corpus; persist it locally (`index.save_local`) when rebuild
time starts to matter.

## Prerequisites

1. **A running XNS Relayer** with `~/.xns/credentials` written.
   See the [repo-level prerequisites](../README.md#prerequisites).

2. **ffmpeg on PATH** — `apt install ffmpeg` / `brew install ffmpeg`. The
   de facto frame sampler; nothing else in the ecosystem is close.

3. **An OpenAI API key** — transcription, captions, embeddings, and the
   chat model all use it.

4. **Media in a bucket** — `.mp4`/`.webm`/`.mov`/`.mkv` get both tracks
   processed; audio-only formats get transcription only:

   ```bash
   # AWS CLI pointed at your Relayer
   aws s3 mb s3://media --endpoint-url http://localhost:9000
   aws s3 cp demo-recording.mp4 s3://media/ --endpoint-url http://localhost:9000
   ```

## Run it

```bash
cd multimodal-rag
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python pipeline.py media "What was on the slide when they discussed pricing?"
```

First run: transcribes, samples and captions frames, embeds, caches all
three, answers. Re-run: cache hits skip the transcription, caption, and
document-embedding calls; the query embedding and answer call still happen —
those are per-question costs.

## Data flow — what leaves your infrastructure

Raw media bytes, sampled frames, transcript text, captions, chunk text, and
your questions are all sent to OpenAI's API. If your recordings contain
material you cannot send to a third-party API — customer data, legal
content, anything under a recording-consent regime — swap the OpenAI calls
for a locally hosted stack (an open VLM such as Qwen-VL for captions, local
Whisper for transcription) before using this pipeline. The XNS side —
media and all three caches — stays on your own Relayer.

Deleting a media file does not delete its cached transcript, captions,
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

- **Text-mediated visual retrieval.** Frames are retrieved by their
  captions, not by joint image-text embeddings. A visual detail the
  captioner didn't mention is not findable. Cross-modal embedding models
  (SigLIP, voyage-multimodal) retrieve what captions miss — at the cost of
  a second index and a heavier stack.
- **Fixed-interval sampling.** One frame per 10 seconds misses anything
  shorter, and wastes captions on static scenes. Scene-change detection
  (`ffmpeg -vf select='gt(scene,0.3)'`) is the next step up.
- **Timestamps are labels, not citations.** They ride in the chunk text, so
  answers can quote them, but nothing verifies the model cited the right
  one. A production system carries them as structured metadata.
- **Character-based chunking.** 500/50 is a tutorial default; it will split
  speaker turns and can separate a caption from related speech. Measure
  against your own queries.
- **Single-process, in-memory index.** No concurrency, no restart recovery,
  no multi-user story. Fine for a workstation; a service needs a real
  vector store.
- **Happy path only.** Audio over the transcription API's 25 MB cap is
  extracted and compressed with ffmpeg automatically (about 14 MB per hour
  at mono 32 kbps), but multi-hour recordings that still exceed the cap
  need segmenting, and there is no handling for unsupported codecs, API
  rate limits, or partial failures. Whole objects are also held in memory
  during processing — a 3 GB video needs 3 GB of RAM.

## What to try next

- **Tighten the frame interval** on content that changes fast — the media
  re-read is free, and the caption cache keys include the interval, so the
  old set is not clobbered.
- **Scene-change sampling.** Swap the `fps=` filter for a `select=` scene
  filter and compare caption coverage on the same video — another free
  re-read.
- **Speaker labels.** `gpt-4o-transcribe-diarize` returns timestamped,
  speaker-labeled segments; merging them with the caption timeline gives
  "who said what while what was on screen."
- **Persist the FAISS index** (`save_local`/`load_local`) so a fresh
  process skips the rebuild.
