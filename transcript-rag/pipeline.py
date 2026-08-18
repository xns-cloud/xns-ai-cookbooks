#!/usr/bin/env python3
"""
Transcript RAG on XNS storage — audio/video speech-to-text, then text RAG.

Transcribes the audio track of media files in an XNS bucket, embeds the
transcript text, and answers questions against it.  Both the transcripts
and the embeddings are cached back into the bucket, keyed so a re-run
only pays for work that actually changed:

  - transcript cache key = object ETag + transcription model
    (new upload or model change -> re-transcribe; otherwise cached)
  - embedding cache key   = chunk content hash + embedding model
    (new chunking strategy -> re-embed changed chunks only)

Reads from XNS carry no per-read charge, so cache hits cost nothing on
the storage side.  OpenAI API calls (transcription, embedding, chat) are
billed by OpenAI as usual — caching exists to avoid repeating them.

This pipeline does NOT analyze video frames.  A question about what was
on screen cannot be answered — only what was said.

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    python pipeline.py <bucket> ["your question here"]

Credentials resolve from ~/.xns/credentials automatically.
"""

import sys
from pathlib import Path

from langchain_xns import XNSBlobLoader, XNSByteStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAI

MEDIA_EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".flac")
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
EMBED_MODEL = "text-embedding-3-small"


def transcribe_all(bucket: str) -> list[str]:
    """Return one transcript per media object, reading from cache when the
    object (by ETag) was already transcribed with the current model."""
    cache = XNSByteStore(bucket, prefix="cache/transcripts/")
    client = OpenAI()
    transcripts = []
    for blob in XNSBlobLoader(bucket, suffixes=MEDIA_EXTENSIONS).yield_blobs():
        cache_key = f"{blob.metadata['etag']}.{TRANSCRIBE_MODEL}"
        (cached,) = cache.mget([cache_key])
        if cached is not None:
            transcripts.append(cached.decode("utf-8"))
            print(f"  {blob.metadata['key']}: transcript cache hit")
            continue
        result = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=(Path(blob.path).name, blob.as_bytes()),
        )
        cache.mset([(cache_key, result.text.encode("utf-8"))])
        transcripts.append(result.text)
        print(f"  {blob.metadata['key']}: transcribed ({len(result.text)} chars)")
    return transcripts


def main():
    bucket = sys.argv[1] if len(sys.argv) > 1 else "media"
    question = sys.argv[2] if len(sys.argv) > 2 else "Summarize the key points."

    transcripts = transcribe_all(bucket)
    if not transcripts:
        print(f"No media files in '{bucket}'. Upload some audio first, then re-run.")
        sys.exit(1)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).create_documents(transcripts)
    print(f"{len(chunks)} chunks")

    # Embedding cache: keyed by chunk content hash + model namespace, stored
    # in the same bucket.  A cache hit skips the OpenAI embedding call.
    embedder = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model=EMBED_MODEL),
        XNSByteStore(bucket, prefix="cache/embeddings/"),
        namespace=EMBED_MODEL,
        key_encoder="sha256",
    )

    # The FAISS index is in-memory and rebuilt each run from cached vectors.
    # Persist it locally (index.save_local) if rebuild time starts to matter.
    index = FAISS.from_documents(chunks, embedder)

    docs = index.similarity_search(question, k=3)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    answer = ChatOpenAI(model="gpt-4.1-mini").invoke(
        f"Answer based only on this transcript context. If the context does "
        f"not contain the answer, say so.\n\n{context}\n\nQuestion: {question}"
    )
    print(f"\nQ: {question}")
    print(f"A: {answer.content}")


if __name__ == "__main__":
    main()
