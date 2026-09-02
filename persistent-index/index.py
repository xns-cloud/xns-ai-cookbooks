#!/usr/bin/env python3
"""
A vector index that survives the process, the container, and the machine.

The other recipes in this repository build their index in memory and lose
it at exit. Re-running means re-embedding the whole corpus, and nothing
carries to a second machine. That is the ceiling this recipe removes.

Qdrant holds the vectors. After ingest, its collection is snapshotted and
the snapshot is stored in your XNS bucket. A cold Qdrant -- fresh
container, empty volume, different host -- pulls that snapshot back and
answers queries without re-embedding anything.

  documents in XNS ──▶ chunk ──▶ embed ──▶ Qdrant collection
                                              │
                                       snapshot│
                                              ▼
                                   XNS: index/<collection>/…
                                              │
        cold Qdrant  ◀── upload/recover ──────┘

Commands:

    python index.py ingest <bucket>
        Load .txt/.md objects, chunk, embed, upsert, snapshot to XNS.

    python index.py query <bucket> "your question"
        Query. Restores from the snapshot first if the collection is
        missing, so this works against a Qdrant that has never seen it.

    python index.py cold-start <bucket> "your question"
        The demonstration: DELETE the local collection, restore it from
        the bucket, then query. Proves the restore did the work instead
        of a warm index quietly answering.

    python index.py status <bucket>
        What is in Qdrant, and what snapshots are in the bucket.

Credentials resolve from ~/.xns/credentials, or from XNS_ENDPOINT /
XNS_ACCESS_KEY_ID / XNS_SECRET_ACCESS_KEY -- the same zero-config read
every other recipe here uses.

Embeddings:

    XNS_EMBEDDINGS=openai   (default) text-embedding-3-small, needs OPENAI_API_KEY
    XNS_EMBEDDINGS=hashing            deterministic, local, no API account

The hashing embedder exists so you can prove the storage path -- upsert,
snapshot, upload, restore, retrieve -- against your own gateway before
spending anything. Its retrieval quality is term-overlap, not semantic;
it is a test fixture, not a model. Never ingest a real corpus with it.
"""

import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_xns import XNSBlobLoader, XNSByteStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("XNS_COLLECTION", "corpus")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
SUFFIXES = (".txt", ".md")
SNAPSHOT_PREFIX = "index"
MANIFEST_KEY = f"{SNAPSHOT_PREFIX}/latest.json"


# ── embeddings ──────────────────────────────────────────────────────────

class HashingEmbeddings:
    """Deterministic bag-of-words vectors. No network, no account.

    Present so the storage path can be exercised end to end without an
    embedding provider -- the part this recipe is actually about. It
    retrieves on term overlap and understands nothing. Do not ship it.
    """

    dimension = 512

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in text.lower().split():
            token = token.strip(".,:;!?()[]\"'")
            if not token:
                continue
            slot = int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
            ) % self.dimension
            vec[slot] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(bucket: str):
    """The OpenAI path caches vectors in the bucket, exactly as
    multimodal-rag does, so a re-ingest of unchanged text is free on the
    model side as well as the storage side."""
    choice = os.environ.get("XNS_EMBEDDINGS", "openai").lower()
    if choice == "hashing":
        print("embeddings: hashing (local, deterministic, not semantic)")
        return HashingEmbeddings(), HashingEmbeddings.dimension

    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_openai import OpenAIEmbeddings

    print(f"embeddings: {EMBED_MODEL}")
    embedder = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model=EMBED_MODEL),
        XNSByteStore(bucket, prefix="cache/embeddings/"),
        namespace=EMBED_MODEL,
        key_encoder="sha256",
    )
    return embedder, 1536


# ── the bucket side ─────────────────────────────────────────────────────

class SnapshotStore:
    """Snapshots and their manifest, in the same bucket as the corpus.

    Uses XNSByteStore rather than a hand-rolled boto3 client so the
    credential resolution is the one the rest of the repository already
    settled on.
    """

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.store = XNSByteStore(bucket, prefix=f"{SNAPSHOT_PREFIX}/")
        self.manifest_store = XNSByteStore(bucket, prefix="")

    def put(self, name: str, blob: bytes) -> str:
        self.store.mset([(name, blob)])
        return f"{SNAPSHOT_PREFIX}/{name}"

    def get(self, name: str) -> bytes | None:
        (found,) = self.store.mget([name])
        return found

    def write_manifest(self, manifest: dict) -> None:
        self.manifest_store.mset(
            [(MANIFEST_KEY, json.dumps(manifest, indent=2).encode("utf-8"))]
        )

    def read_manifest(self) -> dict | None:
        (found,) = self.manifest_store.mget([MANIFEST_KEY])
        return json.loads(found) if found else None


# ── the Qdrant side ─────────────────────────────────────────────────────

def qdrant() -> QdrantClient:
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=120)
        client.get_collections()
        return client
    except Exception as exc:
        sys.exit(
            f"Cannot reach Qdrant at {QDRANT_URL}: {exc}\n"
            "Start it with:  docker compose up -d --wait"
        )


def snapshot_to_bucket(client: QdrantClient, snapshots: SnapshotStore) -> dict:
    """Snapshot the collection, pull the file out of Qdrant, store it.

    Qdrant keeps its snapshots on its own disk, which is the disk this
    recipe assumes you can lose. The download-then-store round trip is
    the whole point -- what lands in the bucket is what survives.
    """
    started = time.perf_counter()
    description = client.create_snapshot(collection_name=COLLECTION)
    if description is None:
        sys.exit("Qdrant declined to create a snapshot.")

    url = f"{QDRANT_URL}/collections/{COLLECTION}/snapshots/{description.name}"
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    blob = response.content

    key = snapshots.put(description.name, blob)
    # Qdrant's copy has served its purpose; leaving it grows its disk
    # every ingest for no benefit.
    client.delete_snapshot(collection_name=COLLECTION, snapshot_name=description.name)

    manifest = {
        "collection": COLLECTION,
        "snapshot": description.name,
        "key": key,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "points": client.count(COLLECTION, exact=True).count,
        "embeddings": os.environ.get("XNS_EMBEDDINGS", "openai").lower(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    snapshots.write_manifest(manifest)
    print(
        f"snapshot {description.name} -> s3://{snapshots.bucket}/{key} "
        f"({len(blob):,} bytes, {manifest['points']} points, "
        f"{time.perf_counter() - started:.1f}s)"
    )
    return manifest


def restore_from_bucket(client: QdrantClient, snapshots: SnapshotStore) -> dict:
    """Recover the collection from the snapshot in the bucket.

    The bytes are POSTed to Qdrant rather than handed to it as a URL.
    `recover_snapshot(location=...)` would make Qdrant fetch the file
    itself, which requires the container to reach your gateway -- true on
    a laptop, often false in a cluster. Uploading works either way.
    """
    manifest = snapshots.read_manifest()
    if manifest is None:
        sys.exit(
            f"No snapshot manifest at s3://{snapshots.bucket}/{MANIFEST_KEY}. "
            "Run `python index.py ingest <bucket>` first."
        )

    blob = snapshots.get(manifest["snapshot"])
    if blob is None:
        sys.exit(
            f"Manifest names {manifest['key']}, but that object is not in the "
            "bucket. The manifest and the snapshots have diverged."
        )

    digest = hashlib.sha256(blob).hexdigest()
    if digest != manifest["sha256"]:
        sys.exit(
            "Snapshot checksum does not match the manifest -- refusing to "
            f"restore.\n  manifest {manifest['sha256']}\n  actual   {digest}"
        )

    started = time.perf_counter()
    response = requests.post(
        f"{QDRANT_URL}/collections/{manifest['collection']}/snapshots/upload",
        params={"priority": "snapshot", "wait": "true"},
        files={"snapshot": (manifest["snapshot"], io.BytesIO(blob),
                            "application/octet-stream")},
        timeout=600,
    )
    response.raise_for_status()

    count = client.count(manifest["collection"], exact=True).count
    print(
        f"restored {manifest['snapshot']} from s3://{snapshots.bucket}/"
        f"{manifest['key']} -> {count} points "
        f"({time.perf_counter() - started:.1f}s)"
    )
    if count != manifest["points"]:
        print(
            f"  WARNING: manifest recorded {manifest['points']} points. "
            "The bucket and the collection disagree."
        )
    return manifest


def ensure_collection(client: QdrantClient, snapshots: SnapshotStore) -> None:
    if client.collection_exists(COLLECTION):
        return
    print(f"collection '{COLLECTION}' not present -- restoring from the bucket")
    restore_from_bucket(client, snapshots)


# ── commands ────────────────────────────────────────────────────────────

def ingest(bucket: str) -> None:
    embedder, dimension = build_embedder(bucket)

    texts, sources = [], []
    for blob in XNSBlobLoader(bucket, suffixes=SUFFIXES).yield_blobs():
        texts.append(blob.as_bytes().decode("utf-8", errors="replace"))
        sources.append(blob.metadata["key"])
    if not texts:
        sys.exit(
            f"No {' or '.join(SUFFIXES)} objects in '{bucket}'. "
            "Upload some documents first."
        )
    print(f"{len(texts)} document(s) from s3://{bucket}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks, chunk_sources = [], []
    for text, source in zip(texts, sources):
        pieces = splitter.split_text(text)
        chunks.extend(pieces)
        chunk_sources.extend([source] * len(pieces))
    print(f"{len(chunks)} chunks")

    started = time.perf_counter()
    vectors = embedder.embed_documents(chunks)
    print(f"embedded in {time.perf_counter() - started:.1f}s")

    client = qdrant()
    # Recreated deliberately: a re-ingest is a rebuild, and leaving stale
    # points behind from a previous corpus is the subtler failure.
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    client.upsert(
        COLLECTION,
        points=[
            PointStruct(id=i, vector=v, payload={"text": c, "source": s})
            for i, (v, c, s) in enumerate(zip(vectors, chunks, chunk_sources))
        ],
    )
    print(f"upserted {len(chunks)} points into '{COLLECTION}'")

    snapshot_to_bucket(client, SnapshotStore(bucket))


def query(bucket: str, question: str, cold: bool = False) -> None:
    client = qdrant()
    snapshots = SnapshotStore(bucket)

    if cold:
        # The demonstration. Without this, a warm collection answers and
        # the restore path is never exercised.
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
            print(f"deleted local collection '{COLLECTION}' -- Qdrant is now cold")
        restore_from_bucket(client, snapshots)
    else:
        ensure_collection(client, snapshots)

    embedder, _ = build_embedder(bucket)
    # One embedding call, for the question. The corpus is not re-embedded
    # -- that is the saving, and it is worth being precise about.
    hits = client.query_points(
        COLLECTION, query=embedder.embed_query(question), limit=4, with_payload=True
    ).points

    print(f"\nQ: {question}")
    if not hits:
        print("A: nothing retrieved -- the collection is empty.")
        return

    print(f"\nretrieved {len(hits)} chunks:")
    for hit in hits:
        text = hit.payload["text"].replace("\n", " ")
        print(f"  [{hit.score:.3f}] {hit.payload['source']}: {text[:110]}…")

    if os.environ.get("XNS_EMBEDDINGS", "openai").lower() == "hashing":
        print(
            "\nNo answer generated: the hashing embedder is a storage-path "
            "fixture, and an answer built on term-overlap retrieval would "
            "misrepresent what this recipe does."
        )
        return

    from langchain_openai import ChatOpenAI

    context = "\n\n---\n\n".join(h.payload["text"] for h in hits)
    answer = ChatOpenAI(model=CHAT_MODEL).invoke(
        "Answer using only this context. If it does not contain the answer, "
        f"say so.\n\n{context}\n\nQuestion: {question}"
    )
    print(f"\nA: {answer.content}")


def status(bucket: str) -> None:
    client = qdrant()
    exists = client.collection_exists(COLLECTION)
    points = client.count(COLLECTION, exact=True).count if exists else 0
    print(f"qdrant {QDRANT_URL}")
    print(f"  collection '{COLLECTION}': {'present' if exists else 'absent'}"
          f"{f', {points} points' if exists else ''}")

    manifest = SnapshotStore(bucket).read_manifest()
    print(f"bucket s3://{bucket}")
    if manifest is None:
        print(f"  {MANIFEST_KEY}: absent -- nothing has been ingested yet")
        return
    print(f"  {manifest['key']}")
    print(f"    {manifest['bytes']:,} bytes, {manifest['points']} points, "
          f"{manifest['embeddings']} embeddings, created {manifest['created']}")
    if exists and points != manifest["points"]:
        print("  NOTE: Qdrant and the manifest disagree on the point count.")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    command, rest = args[0], args[1:]
    if command in ("ingest", "status"):
        if not rest:
            sys.exit(f"usage: python index.py {command} <bucket>")
        (ingest if command == "ingest" else status)(rest[0])
    elif command in ("query", "cold-start"):
        if len(rest) < 2:
            sys.exit(f'usage: python index.py {command} <bucket> "your question"')
        query(rest[0], rest[1], cold=(command == "cold-start"))
    else:
        sys.exit(f"unknown command '{command}'. Try --help.")


if __name__ == "__main__":
    main()
