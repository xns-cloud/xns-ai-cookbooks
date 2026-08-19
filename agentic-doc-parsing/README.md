# Agentic Document Parsing — PDFs and Spreadsheets to JSON

Turn a bucket of documents into structured records an agent can query,
and re-run it every night without paying to read the corpus again.

Each document is parsed locally by [Docling](https://github.com/docling-project/docling)
into Markdown, then turned into a JSON record by an OpenAI model using
structured outputs. **Raw document bytes never leave the host** — only the parsed
text and the extraction prompt are sent to the model API. Every stage is
cached in the bucket under a fingerprint of everything that can change
its output, so an unchanged corpus makes no model calls at all.

This is a starter recipe: single process, one schema, no work queue. The
[Limitations](#limitations) section says exactly where that stops being
enough.

## Architecture

```
    ┌──────────────────────────────────────────┐
    │               XNS Bucket                 │
    │    contracts/*.pdf   finance/*.xlsx      │
    │       (S3 API, no per-read charge)       │
    └──────┬───────────────────────────────────┘
           │ XNSBlobLoader — filters on the listing,
           │ so non-matching keys are never fetched
           ▼
    ┌──────────────────────┐                ┌─────────────────────┐
    │  Docling  (LOCAL)    │───────────────▶│  XNS caches         │
    │  PDF / XLSX          │◀───────────────│  cache/parsed/      │
    │  → Markdown          │  key: source,  │  cache/extracted/   │
    └──────┬───────────────┘  version, and  └─────────────────────┘
           │                  parser/config   cache hit = no work
           │                  fingerprint
           ▼
    ┌──────────────────────┐
    │  gpt-4.1-mini        │  structured outputs, Pydantic schema
    │  responses.parse()   │  key: parse fp + schema + prompt + model
    └──────┬───────────────┘
           ▼
    ┌──────────────────────────────────────────┐
    │  extracted/<source-key>.json             │
    │  record + provenance — the durable        │
    │  output agents read                       │
    └──────────────────────────────────────────┘
```

Three things worth naming in that diagram:

- **Parsing is local.** Docling runs on your box. The model never sees a
  PDF, only text you could inspect first.
- **Both caches live in the bucket**, via `XNSByteStore`. Nothing depends
  on local disk surviving between runs.
- **`extracted/` is not a cache.** It is the output contract, described
  below. `cache/` is internal and may change shape between versions.

### The output contract

For every source object the pipeline writes one durable JSON artifact:

```
extracted/<percent-encoded-source-key>.json
```

Its envelope carries the record plus the provenance needed to trust it:
`source_key`, `source_etag`, `parse_fingerprint`, `extract_fingerprint`,
`parser` (Docling version and configuration), `extraction_model`,
`schema_version`, `prompt_version`, `created`, and `pipeline_warnings`.

A later run skips the write only when those fingerprints still match. The
pipeline writes a complete artifact in one PUT rather than streaming an
update into place; overwrite-then-read behaved correctly on the gateway
used during development, but verify overwrite-read semantics for your own
deployment before depending on it.

Downstream agents should read `extracted/`, check `pipeline_warnings` and
the record's own `warnings`, and use `source_key` plus each item's
`source_ref` to go back to the original when a value matters.

A real artifact, trimmed, from a spreadsheet of invoices:

```json
{
  "source_key": "q2-invoices.xlsx",
  "source_etag": "25c4d57a118d2e3c731b453c3859d1fc",
  "parse_fingerprint": "9fb9aa63102fc316d8207902c3c9d0ce",
  "extract_fingerprint": "3967778cbc6fe2310da0ab55631f444d",
  "parser": "docling 2.120.3 (default-converter)",
  "extraction_model": "gpt-4.1-mini",
  "schema_version": "1",
  "prompt_version": "1",
  "created": "2026-08-19T12:52:46.622279+00:00",
  "pipeline_warnings": [],
  "extraction": {
    "title": "Invoice Summary and Related Notes",
    "doc_type": "Invoice Table and Notes",
    "key_facts": [
      {
        "statement": "Invoice INV-2026-043 from Fabrikam GmbH is disputed and was escalated to legal on 2026-06-14.",
        "source_ref": "Note below table"
      }
    ],
    "entities": ["Northwind Robotics", "Contoso Analytics", "Fabrikam GmbH"],
    "amounts": [
      {"as_written": "18400", "currency": "USD", "source_ref": "Invoice Table"},
      {"as_written": "31900", "currency": "EUR", "source_ref": "Invoice Table"}
    ],
    "warnings": []
  }
}
```

Those `source_ref` values are what let an agent — or a person — go back and
check a number instead of trusting it.

### What the caches actually save

| You change… | Re-parse (local CPU)? | Re-extract (model API)? | Storage |
|---|---|---|---|
| Nothing (nightly re-run) | No | No | $0 |
| The extraction schema or prompt | No | Yes — every document | $0 |
| The extraction model | No | Yes — every document | $0 |
| Docling version or parser config | Yes | Yes — parsed text changed | $0 |
| One document (new ETag) | That file | That file | $0 |
| Add 50 documents | Those files | Those files | $0 |

The storage column reflects XNS's no-per-read-charge model for this access
pattern; compute, capacity, and model API usage are still yours.

Note the asymmetry with a transcription pipeline: here the parse stage is
local CPU, not a metered API. The parse cache buys wall-clock, not money.
The extraction cache is the one that avoids spend.

## Prerequisites

1. **A running XNS Relayer** with `~/.xns/credentials` written.
   See the [repo-level prerequisites](../README.md#prerequisites).

2. **An OpenAI API key** — the extraction stage uses it. Parsing does not.

3. **Documents in a bucket** — `.pdf` and `.xlsx`. Docling reads more
   formats than that; these are the two this recipe documents and tests:

   ```bash
   # AWS CLI pointed at your Relayer
   aws s3 mb s3://documents --endpoint-url http://localhost:9000
   aws s3 cp contract.pdf s3://documents/ --endpoint-url http://localhost:9000
   ```

The credentials this script needs are narrow: list and read on the source
prefixes, write on `cache/`, `extracted/`, and `runs/`. It never deletes.

## Run it

```bash
cd agentic-doc-parsing
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python pipeline.py documents
```

First run parses, extracts, caches, and writes one artifact per document.
Re-run and every stage reports a cache hit.

You can prove no model calls happen on a cached re-run: set
`OPENAI_API_KEY` to a deliberately invalid value and run it again. It
completes normally, because nothing asks the API for anything.

The pieces that decide correctness — cache-key derivation, section
splitting, merging, artifact naming, retry bounds — have offline tests
that need no API key, no bucket, and no Docling install:

```bash
python test_pipeline.py
```

**First execution is slower than steady state.** Docling downloads its
layout and table-recognition models on first use. Runtime after that
depends on document complexity, page count, your hardware, and model API
latency — measure it on your own corpus rather than trusting a number
here.

### The overnight loop

Because every stage is keyed by a fingerprint of its inputs, the script is
idempotent: run it as often as you like and it only spends model tokens on
documents that actually changed.

Linux cron example — `flock` is Linux-specific; on another scheduler use
whatever single-instance lock it provides:

```cron
0 2 * * * cd /srv/docs && /usr/bin/flock -n /tmp/docparse.lock \
          python pipeline.py documents >> /var/log/docparse.log 2>&1
```

The `flock` matters. Two overlapping runs will duplicate model calls and
race each other writing the same artifact. This is one process by design —
if you need many workers, you need a real queue, and that is a different
program.

Each run writes a machine-readable summary to `runs/<timestamp>.json`
listing every document as written, unchanged, or failed with its error. A
failed document is logged and skipped; it does not end the run.

## Data flow — what leaves your infrastructure

Parsed Markdown and the extraction prompt go to OpenAI. The raw PDF and
XLSX bytes do not — Docling runs locally. That is a materially tighter
boundary than a hosted parsing service, where the document itself is
uploaded.

It is not the same as nothing leaving. Extracted text can carry everything
confidential the document contained, so treat the model provider as a
recipient of that text. Keep credentials out of logs, don't log document
contents, and route anything you can't send to a third party to a local
OpenAI-compatible endpoint (vLLM, Ollama) instead — structured-output
support varies across local servers, so verify it before relying on it.

Document text is untrusted input. The extraction prompt states that the
document is data rather than instructions, and downstream agents should
treat extracted fields the same way — never as commands to execute.

Deleting a source document does not delete its cached parse, its cached
extraction, or its artifact. The cache keys are fingerprints, not
reversible names, so if deletion must propagate, drive it from the
artifact's `source_key` field and own that workflow.

## MCP config

To let an AI assistant set up and manage your Relayer, add this to your
Claude Desktop or Cursor MCP config:

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

Two different jobs, worth separating clearly:

| Server | What it does |
|---|---|
| `@xns-cloud/relayer-mcp` | Sets up and manages the Relayer itself — prerequisites, registration, install, health, S3 credential provisioning. It writes `~/.xns/credentials`, which this script reads. It does **not** expose object-level tools. |
| A generic S3 MCP server | Lists, reads, and writes bucket objects, for an agent that wants to inspect `extracted/` directly. Any server supporting a custom `endpoint_url` with path-style addressing works. |

An agent with shell access can equally just run this script on a schedule
and read the run summary.

## Limitations

- **Schema conformance is not accuracy.** Structured outputs constrain a
  *successful* response to the declared schema. They do not make the
  contents true. The script still handles refusals, incomplete responses,
  and API errors, and it records them in `pipeline_warnings` — but a
  well-formed, confidently wrong field looks identical to a right one.
  Verify high-impact values against the `source_ref` they cite.
- **One generic schema.** A single shape across contracts and financial
  workbooks flattens what makes each interesting. Real deployments write a
  schema per document type and route on `doc_type`.
- **Digital-native PDFs.** Docling has an OCR path; this configuration
  does not enable it. Add and benchmark an explicit OCR setup before
  pointing this at scans, and expect it to be slower.
- **Workbooks become text.** Docling gives you a textual representation
  for a model to read — not a calculation engine. Formulas, named ranges,
  hidden sheets, merged cells, and cross-sheet references should be
  validated against your own workbooks before you trust extractions from
  them.
- **Long documents are sectioned.** Markdown over the input budget is
  split at paragraph boundaries, extracted per section, and merged, with a
  warning recorded on the record. A fact spanning two sections can be
  missed.
- **Single process, in memory.** Each object is held in memory while it is
  parsed. There is no work queue, no distributed execution, no
  exactly-once guarantee, and no dead-letter path — retries are bounded
  and then the document is marked failed.
- **No retrieval index.** This recipe ends at structured JSON. Searching
  across extractions is the [Multimodal RAG](../multimodal-rag/) recipe's
  territory, or a downstream store.

## What to try next

- **A schema per document type.** Classify first, then extract with the
  right shape, writing each type to its own prefix.
- **Feed `extracted/` into the RAG recipe.** The two compose on one
  bucket: extract structure here, embed and query it there.
- **Docling's JSON export** instead of Markdown, when table cell
  positions matter downstream.
- **Tighten the credentials.** Give the extraction job write access only
  to `cache/`, `extracted/`, and `runs/`, and read-only access to sources.
