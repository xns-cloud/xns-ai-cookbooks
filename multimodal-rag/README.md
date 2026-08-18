# Multimodal RAG — Audio & Video Transcription Pipeline

Transcribe media files stored in XNS, embed the transcriptions, and query them
with natural language.

The pipeline reads audio and video from an XNS bucket, transcribes them
with OpenAI, chunks the text, embeds it, and caches the embeddings back
in XNS.  Every read from the bucket — the original media and the cached
vectors — is unmetered.  So you can re-run the pipeline with a different
chunking strategy, a new embedding model, or against a growing corpus, without
paying to move the data again.

## Architecture

```
    ┌──────────────────┐
    │    XNS Bucket    │  your audio/video files
    │  (S3 on :9000)   │  reads are unmetered
    └────────┬─────────┘
             │  XNSBlobLoader
             │  streams each object
             ▼
    ┌──────────────────┐
    │  Transcription   │  gpt-4o-mini-transcribe
    │   (OpenAI)       │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Text Splitter   │  500-char chunks
    │  (LangChain)     │  with 50-char overlap
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐      ┌──────────────────┐
    │  Embeddings      │─────▶│  XNS Cache       │
    │  text-embed-3    │      │  (XNSByteStore)  │
    │  (OpenAI)        │      │  cache/embeddings/│
    └────────┬─────────┘      └──────────────────┘
             │                  re-run skips the
             │                  embedding API call;
             │                  reading the cache
             ▼                  from XNS is free
    ┌──────────────────┐
    │  FAISS Index     │  in-memory similarity
    │  (local)         │  search
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  gpt-4.1-mini    │  answers from the
    │  (ChatOpenAI)    │  retrieved context
    └──────────────────┘
```

### Why this matters at scale

A typical multimodal RAG pipeline reads the same raw media multiple times:
once for transcription, once for speaker diarization, once when you retune
your chunking, again when you upgrade the embedding model.  On egress-billed
storage, each pass is a line item.  On XNS, those reads are zero — the only
costs are compute (transcription, embeddings, LLM) and storage capacity.

## Prerequisites

1. **A running XNS Relayer** with `~/.xns/credentials` written.
   See the [repo-level prerequisites](../README.md#prerequisites).

2. **An OpenAI API key** — transcription, embeddings, and the chat model all use it.

3. **Media in a bucket** — upload some `.mp3`, `.wav`, `.m4a`, or `.mp4` files.
   For a quick test, any short audio clip works:

   ```bash
   # Using the AWS CLI pointed at your Relayer
   aws s3 mb s3://media --endpoint-url http://localhost:9000
   aws s3 cp meeting-recording.mp3 s3://media/ --endpoint-url http://localhost:9000
   ```

   Or use `langchain-xns` in Python:

   ```python
   import boto3
   from langchain_xns._config import resolve_config

   cfg = resolve_config()
   s3 = boto3.client("s3",
       endpoint_url=cfg.endpoint,
       aws_access_key_id=cfg.access_key_id,
       aws_secret_access_key=cfg.secret_access_key,
       region_name=cfg.region,
   )
   s3.upload_file("meeting-recording.mp3", "media", "meeting-recording.mp3")
   ```

## Run it

```bash
cd multimodal-rag
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python pipeline.py media "What decisions were made in the meeting?"
```

First run: transcribes, embeds, caches, answers.
Second run: reads cached embeddings from XNS (free), skips the embedding API,
answers instantly.

## MCP config

To let an AI assistant manage your Relayer — install it, create buckets, upload
files — add this to your Claude Desktop or Cursor MCP config:

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

The MCP server handles its own auth via OIDC.  It also writes
`~/.xns/credentials` during setup, which `pipeline.py` reads automatically.

## What to try next

- **Swap the embedding model.** Change `text-embedding-3-small` to
  `text-embedding-3-large` and re-run.  The old cache keys won't collide
  (they're namespaced), and re-reading the media from XNS is free.

- **Add speaker diarization.** Use `gpt-4o-transcribe-diarize` on the same
  audio, then join the speaker labels with the transcript before chunking.
  Same source files, another free read.

- **Scale to a corpus.** Point the script at a bucket with hundreds of
  recordings.  `XNSBlobLoader` streams and never loads the full listing
  into memory.

- **Persist the FAISS index.** Save it to disk between runs so you only
  embed new files.  The embedding cache still prevents redundant API calls
  for files already processed.
