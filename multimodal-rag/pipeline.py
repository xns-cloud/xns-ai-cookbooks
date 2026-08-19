#!/usr/bin/env python3
"""
Multimodal RAG on XNS storage — speech AND visuals from audio/video.

Processes both tracks of media files stored in an XNS bucket:

  audio track -> gpt-4o-mini-transcribe -> transcript text
  video track -> ffmpeg frame sampling -> gpt-4.1-mini vision captions

Transcript and captions are timestamp-labeled, embedded together, and
retrieved together — so "what was on the slide when they made the call"
and "what did they decide" are both answerable.

Everything expensive is cached back into the bucket, keyed so a re-run
only pays for what changed:

  - transcript cache key = object ETag + transcription model
  - caption cache key    = object ETag + frame interval + vision model
  - embedding cache key  = chunk content hash + embedding model

Reads from XNS carry no per-read charge, so cache hits cost nothing on
the storage side.  OpenAI API calls are billed by OpenAI as usual —
caching exists to avoid repeating them.

Usage:
    pip install -r requirements.txt      # plus: ffmpeg on PATH
    export OPENAI_API_KEY=sk-...
    python pipeline.py <bucket> ["your question here"]

Credentials resolve from ~/.xns/credentials automatically.
"""

import base64
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_xns import XNSBlobLoader, XNSByteStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAI

MEDIA_EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".flac")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv")
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
VISION_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
FRAME_INTERVAL = 10  # seconds between sampled frames

CAPTION_PROMPT = (
    "Describe what is visible in this video frame in one or two sentences. "
    "Read out any on-screen text, slide titles, chart labels, or UI state."
)


TRANSCRIBE_UPLOAD_LIMIT = 25 * 1024 * 1024  # OpenAI audio endpoint cap


def transcribe(blob, cache, client) -> str:
    """Speech-to-text for one media object, cached by ETag + model.

    The transcription endpoint caps uploads at 25 MB, so anything larger —
    every real video — has its audio track extracted and compressed with
    ffmpeg (mono, 32 kbps MP3: about 14 MB per hour) before upload."""
    key = f"{blob.metadata['etag']}.{TRANSCRIBE_MODEL}"
    (cached,) = cache.mget([key])
    if cached is not None:
        print(f"  {blob.metadata['key']}: transcript cache hit")
        return cached.decode("utf-8")

    name = Path(blob.metadata["key"]).name
    payload = blob.as_bytes()
    if len(payload) > TRANSCRIBE_UPLOAD_LIMIT:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / name
            src.write_bytes(payload)
            audio = Path(tmp) / "audio.mp3"
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", str(src),
                 "-vn", "-ac", "1", "-b:a", "32k", str(audio)],
                check=True,
            )
            payload = audio.read_bytes()
            name = audio.name
        if len(payload) > TRANSCRIBE_UPLOAD_LIMIT:
            raise ValueError(
                f"{blob.metadata['key']}: audio track is still over 25 MB "
                f"after compression — split it into segments first."
            )

    result = client.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=(name, payload),
    )
    cache.mset([(key, result.text.encode("utf-8"))])
    print(f"  {blob.metadata['key']}: transcribed ({len(result.text)} chars)")
    return result.text


def caption_frames(blob, cache, client) -> list[str]:
    """Sample frames with ffmpeg and caption each with the vision model.
    The whole caption set for an object is cached by ETag + interval + model."""
    key = f"{blob.metadata['etag']}.{FRAME_INTERVAL}s.{VISION_MODEL}"
    (cached,) = cache.mget([key])
    if cached is not None:
        print(f"  {blob.metadata['key']}: caption cache hit")
        return cached.decode("utf-8").split("\n")

    name = blob.metadata["key"]
    captions = []
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / Path(name).name
        src.write_bytes(blob.as_bytes())
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(src),
             "-vf", f"fps=1/{FRAME_INTERVAL}", str(Path(tmp) / "f_%04d.jpg")],
            check=True,
        )
        for i, frame in enumerate(sorted(Path(tmp).glob("f_*.jpg"))):
            b64 = base64.b64encode(frame.read_bytes()).decode()
            response = client.responses.create(
                model=VISION_MODEL,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": CAPTION_PROMPT},
                        {"type": "input_image",
                         "image_url": f"data:image/jpeg;base64,{b64}",
                         "detail": "low"},
                    ],
                }],
            )
            timestamp = i * FRAME_INTERVAL
            captions.append(
                f"[{name} @ {timestamp // 60}:{timestamp % 60:02d} on screen] "
                f"{response.output_text.strip()}"
            )
    cache.mset([(key, "\n".join(captions).encode("utf-8"))])
    print(f"  {name}: captioned {len(captions)} frames")
    return captions


def main():
    bucket = sys.argv[1] if len(sys.argv) > 1 else "media"
    question = sys.argv[2] if len(sys.argv) > 2 else "Summarize the key points."

    transcript_cache = XNSByteStore(bucket, prefix="cache/transcripts/")
    caption_cache = XNSByteStore(bucket, prefix="cache/captions/")
    client = OpenAI()

    texts = []
    for blob in XNSBlobLoader(bucket, suffixes=MEDIA_EXTENSIONS).yield_blobs():
        name = blob.metadata["key"]
        texts.append(f"[{name} spoken] {transcribe(blob, transcript_cache, client)}")
        if name.lower().endswith(VIDEO_EXTENSIONS):
            texts.extend(caption_frames(blob, caption_cache, client))

    if not texts:
        print(f"No media files in '{bucket}'. Upload some audio or video first.")
        sys.exit(1)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).create_documents(texts)
    print(f"{len(chunks)} chunks")

    # Embedding cache: keyed by chunk content hash + model namespace.
    embedder = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model=EMBED_MODEL),
        XNSByteStore(bucket, prefix="cache/embeddings/"),
        namespace=EMBED_MODEL,
        key_encoder="sha256",
    )

    # The FAISS index is in-memory and rebuilt from cached vectors each run.
    index = FAISS.from_documents(chunks, embedder)

    docs = index.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    answer = ChatOpenAI(model=VISION_MODEL).invoke(
        "Answer based only on this context. Entries marked [… spoken] are "
        "from the audio; entries marked [… on screen] describe video frames "
        "at that timestamp. If the context does not contain the answer, "
        f"say so.\n\n{context}\n\nQuestion: {question}"
    )
    print(f"\nQ: {question}")
    print(f"A: {answer.content}")


if __name__ == "__main__":
    main()
