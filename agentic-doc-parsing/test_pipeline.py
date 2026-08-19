#!/usr/bin/env python3
"""Offline tests for the parts of pipeline.py that decide correctness.

No network, no API key, no Docling install — Docling is stubbed, because
what is worth testing here is cache-key derivation, section splitting,
merging, artifact naming, and retry behavior.  Extraction quality is not
unit-testable; it is the reason the artifact carries provenance.

    python test_pipeline.py
"""

import sys
import types
from urllib.parse import unquote

for name in (
    "docling",
    "docling.datamodel",
    "docling.datamodel.base_models",
    "docling.document_converter",
):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["docling.datamodel.base_models"].DocumentStream = object
sys.modules["docling.document_converter"].DocumentConverter = object

import pipeline as p  # noqa: E402


def test_fingerprint_identity():
    a = p.fingerprint("docs/a.pdf", "etag1", "docling", "2.120.3", "default")
    same = p.fingerprint("docs/a.pdf", "etag1", "docling", "2.120.3", "default")
    changed_etag = p.fingerprint("docs/a.pdf", "etag2", "docling", "2.120.3", "default")
    assert a == same, "same inputs must produce the same key"
    assert a != changed_etag, "a new ETag must miss the cache"
    # Concatenating instead of hashing with a separator would collide here.
    assert p.fingerprint("a", "bc") != p.fingerprint("ab", "c")


def test_schema_hash_moves_with_the_schema():
    before = p.schema_hash()
    original, p.SCHEMA_VERSION = p.SCHEMA_VERSION, "999"
    try:
        assert p.schema_hash() != before, "schema version must reach the key"
    finally:
        p.SCHEMA_VERSION = original


def test_split_sections():
    short = "x" * 100
    assert p.split_sections(short) == [short], "short input must not be split"

    big = "\n\n".join(["para " + "y" * 1000 for _ in range(200)])
    sections = p.split_sections(big)
    assert len(sections) > 1
    assert all(len(s) <= p.MAX_SECTION_CHARS for s in sections), "section over budget"
    lost = len(big) - sum(len(s) for s in sections)
    assert lost <= 2 * len(sections), "splitting dropped content"


def test_merge():
    fact = p.Fact(statement="f1", source_ref="p1")
    first = p.Extraction(
        title="Master Agreement",
        doc_type="contract",
        summary="s1",
        key_facts=[fact],
        entities=["Acme"],
        dates=[p.DateRef(as_written="Jan 1", normalized="2026-01-01", source_ref="p1")],
        amounts=[p.Amount(as_written="5.00", currency="USD", source_ref="p1")],
        warnings=["ambiguous party name"],
    )
    second = p.Extraction(
        title="",
        doc_type="",
        summary="s2",
        key_facts=[p.Fact(statement="f2", source_ref="p9")],
        entities=["Acme", "Beta"],
        dates=[],
        amounts=[],
        warnings=[],
    )
    merged = p.merge([first, second], 2)
    assert merged.title == "Master Agreement", "first non-empty title wins"
    assert merged.doc_type == "contract"
    assert merged.summary == "s1 s2"
    assert len(merged.key_facts) == 2
    assert merged.entities == ["Acme", "Beta"], "entities deduped and sorted"
    assert "2 sections" in merged.warnings[-1], "sectioning must be disclosed"


def test_artifact_name_is_flat_and_reversible():
    for key in ["a.pdf", "contracts/2026/q1 report.pdf", "odd/#name?.xlsx"]:
        name = p.artifact_name(key)
        assert "/" not in name, "artifact names must not create prefixes"
        assert unquote(name.removesuffix(".json")) == key, "source key must survive"


def test_with_retries():
    import time

    import httpx
    from openai import RateLimitError

    def transient():
        return RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
            body=None,
        )

    slept, time.sleep = time.sleep, lambda _s: None
    try:
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise transient()
            return "ok"

        assert p.with_retries(flaky, "flaky") == "ok"
        assert attempts["n"] == 3, "should have retried twice then succeeded"

        def always_fails():
            raise transient()

        try:
            p.with_retries(always_fails, "doomed", attempts=2)
            raise AssertionError("must re-raise once the bound is reached")
        except RateLimitError:
            pass
    finally:
        time.sleep = slept


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
