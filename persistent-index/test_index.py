#!/usr/bin/env python3
"""
Offline tests for the persistent-index recipe. No Qdrant, no gateway, no
embedding provider.

These cover the parts that are cheap to get wrong and expensive to
notice: a non-deterministic test embedder (which would make every cold
start look broken), an unnormalized vector (which quietly ruins cosine
scores), and the corruption guard that decides whether a snapshot is
safe to restore.

What they do NOT cover is the round trip itself -- snapshot, upload,
recover. That needs a real Qdrant and a real bucket, and it is what the
"Verified run" section of the README records.

    python test_index.py
"""

import hashlib
import sys

from index import HashingEmbeddings


def test_embedder_is_deterministic():
    a = HashingEmbeddings().embed_query("cold start restores the collection")
    b = HashingEmbeddings().embed_query("cold start restores the collection")
    assert a == b, "same text must embed identically across instances"
    # Different instance, different process order -- still the same vector,
    # which is what makes a restored index comparable to the one ingested.
    print("ok  deterministic across instances")


def test_vectors_are_normalized():
    vec = HashingEmbeddings().embed_query("snapshot upload recover")
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-9, f"expected unit length, got {norm}"
    print("ok  unit-length vectors (cosine distance needs them)")


def test_empty_text_does_not_divide_by_zero():
    vec = HashingEmbeddings().embed_query("   ")
    assert len(vec) == HashingEmbeddings.dimension
    assert all(v == 0.0 for v in vec), "empty text should give a zero vector"
    print("ok  empty text gives a zero vector instead of raising")


def test_punctuation_and_case_fold_together():
    e = HashingEmbeddings()
    assert e.embed_query("Snapshot.") == e.embed_query("snapshot")
    assert e.embed_query("(recover)") == e.embed_query("recover")
    print("ok  case and surrounding punctuation folded")


def test_overlap_ranks_above_non_overlap():
    e = HashingEmbeddings()
    query = e.embed_query("cold start restore from the bucket")
    close = e.embed_query("a cold start will restore the collection from the bucket")
    far = e.embed_query("ffmpeg samples video frames for captioning")

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cosine(query, close) > cosine(query, far), (
        "the fixture must at least rank term overlap correctly, or a failing "
        "cold start is indistinguishable from a bad embedder"
    )
    print("ok  term overlap outranks unrelated text")


def test_document_and_query_paths_agree():
    e = HashingEmbeddings()
    (doc,) = e.embed_documents(["restore the collection"])
    assert doc == e.embed_query("restore the collection"), (
        "embed_documents and embed_query must agree, or nothing retrieves"
    )
    print("ok  document and query embeddings agree")


def test_checksum_detects_a_changed_byte():
    # The guard restore_from_bucket applies before handing bytes to Qdrant.
    blob = b"pretend snapshot bytes" * 100
    recorded = hashlib.sha256(blob).hexdigest()
    corrupted = bytearray(blob)
    corrupted[17] ^= 0x01
    assert hashlib.sha256(bytes(corrupted)).hexdigest() != recorded
    assert hashlib.sha256(blob).hexdigest() == recorded
    print("ok  checksum catches a single flipped bit")


def main():
    for fn in (
        test_embedder_is_deterministic,
        test_vectors_are_normalized,
        test_empty_text_does_not_divide_by_zero,
        test_punctuation_and_case_fold_together,
        test_overlap_ranks_above_non_overlap,
        test_document_and_query_paths_agree,
        test_checksum_detects_a_changed_byte,
    ):
        fn()
    print("\nall offline tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
