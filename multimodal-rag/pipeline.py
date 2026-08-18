#!/usr/bin/env python3
"""
Multimodal RAG pipeline on XNS storage.

Transcribes audio/video from an XNS bucket, embeds the text with caching,
and answers questions against it.  Re-running costs nothing in storage
fees — reads from XNS are free, and cached embeddings skip the OpenAI call.

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    python pipeline.py <bucket> ["your question here"]

The script reads ~/.xns/credentials automatically.  If you set up XNS
through the MCP server or relayer-quickstart, you already have that file.
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

def main():
    bucket = sys.argv[1] if len(sys.argv) > 1 else "media"
    question = sys.argv[2] if len(sys.argv) > 2 else "Summarize the key points."

    # 1. Pull media from XNS — every read is free

    blobs = list(XNSBlobLoader(bucket, suffixes=MEDIA_EXTENSIONS).yield_blobs())
    print(f"{len(blobs)} media file(s) in '{bucket}'")
    if not blobs:
        print("Upload some audio or video first, then re-run.")
        sys.exit(1)

    # 2. Transcribe

    client = OpenAI()
    transcripts = []
    for blob in blobs:
        result = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=(Path(blob.path).name, blob.as_bytes()),
        )
        transcripts.append(result.text)
        print(f"  {blob.path}: {len(result.text)} chars")

    # 3. Chunk the transcriptions

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).create_documents(transcripts)
    print(f"{len(chunks)} chunks")

    # 4. Embed with caching in XNS
    #
    # First run: computes embeddings via OpenAI, writes them to the bucket.
    # Second run: reads cached vectors from XNS (free) and skips the API.

    embedder = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model="text-embedding-3-small"),
        XNSByteStore(bucket, prefix="cache/embeddings/"),
        namespace="text-embedding-3-small",
        key_encoder="sha256",
    )
    index = FAISS.from_documents(chunks, embedder)

    # 5. Retrieve and answer

    docs = index.similarity_search(question, k=3)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    answer = ChatOpenAI(model="gpt-4.1-mini").invoke(
        f"Answer based only on this transcription:\n\n{context}\n\nQuestion: {question}"
    )
    print(f"\nQ: {question}")
    print(f"A: {answer.content}")


if __name__ == "__main__":
    main()
