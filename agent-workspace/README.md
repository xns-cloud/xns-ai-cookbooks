# Agent Workspace — share artifacts across CrewAI workers

This recipe uses an XNS S3-compatible bucket as a shared workspace for a
small CrewAI pipeline. One agent writes a research brief to
`work/brief.md`; a second agent is given that key, retrieves the artifact,
and writes a final report to `outputs/report.md`.

Use this pattern when agents run in separate processes, containers, or
machines and need to exchange artifacts without putting the full artifact
into every handoff prompt. This is an object-store workspace, not a shared
POSIX filesystem and not a workflow queue.

The included demo runs sequentially in one process for clarity. The same
object-key contract works for workers deployed separately, provided they
can reach the endpoint and hold appropriate credentials.

```yaml
recipe:
  name: agent-workspace
  purpose: Exchange artifacts between agents through an S3-compatible bucket
  framework: CrewAI 1.15.16
  entrypoint: crew.py
  inputs:
    - topic: Text supplied as a command-line argument
  artifacts:
    - key: work/brief.md
      producer: researcher
      consumer: reporter
      format: Markdown
    - key: outputs/report.md
      producer: reporter
      consumer: human_or_downstream_worker
      format: Markdown
  required_environment:
    - XNS_ENDPOINT
    - XNS_ACCESS_KEY_ID
    - XNS_SECRET_ACCESS_KEY
    - OPENAI_API_KEY
  guarantees:
    - Artifact handoff uses object keys rather than embedding artifacts in prompts
  non_goals:
    - POSIX filesystem semantics
    - locking or atomic coordination
    - event-driven orchestration
    - anonymous public hosting
```

## Architecture

```
   worker A (process, container, or machine)   worker B — same endpoint and keys
   ┌─────────────────────────┐                 ┌─────────────────────────┐
   │  Researcher agent       │                 │  Reporter agent         │
   │  writes work/brief.md   │                 │  reads  work/brief.md   │
   │                         │                 │  writes outputs/report  │
   └───────────┬─────────────┘                 └───────────┬─────────────┘
               │  workspace tools (boto3 + endpoint_url)   │
               ▼                                           ▼
   ┌───────────────────────────────────────────────────────────┐
   │                      XNS bucket                           │
   │  inputs/    material the crew starts from                 │
   │  work/      agent-to-agent artifacts                      │
   │  outputs/   final deliverables                            │
   └───────────────────────────────────────────────────────────┘
                               ▲
              humans and downstream jobs retrieve outputs/
```

The bucket is the artifact handoff point. It is not a task queue, an event
bus, a lock manager, or a workflow-state store.

Each prefix has exactly one writer. That convention is the concurrency
model — there is nothing else enforcing it.

| Handoff pattern | Artifact through the LLM context | Survives the worker's container |
|---|---|---|
| Paste the artifact into the next prompt | Full text, every hop | Yes, but long artifacts get truncated |
| Local-file tool | Key only | No — files stay on that worker's disk |
| Shared bucket | Key only | Yes |

Local-file tools are fine for single-worker runs, but their files are not
automatically available to workers running elsewhere and can disappear
with an ephemeral container. XNS does not charge per read under the
product model this recipe assumes, which matters here because agent loops
re-read: a reviewing agent fetches the same draft on every iteration.
LLM inference and context use remain separate from storage transfer.

## Prerequisites

1. A running XNS Relayer with `~/.xns/credentials` written — see the
   [repo-level prerequisites](../README.md#prerequisites).
2. `pip install -r requirements.txt` — CrewAI and boto3, pinned to the
   versions this recipe was tested against.
3. An LLM API key. The example defaults to a model you configure via
   `CREW_MODEL`; choose one appropriate for your latency, quality, and
   data-handling requirements.

## Quickstart

```bash
cd agent-workspace
pip install -r requirements.txt

export XNS_ENDPOINT=http://localhost:9000
export XNS_ACCESS_KEY_ID=...  XNS_SECRET_ACCESS_KEY=...
export OPENAI_API_KEY=sk-...

python crew.py --selftest                    # workspace only, no LLM calls
python crew.py "your topic here"             # the two-agent pipeline
```

`--selftest` exercises write, read, list, missing-key handling, and a
presigned download against your gateway without spending a token. Run it
first; if it fails, the problem is configuration, not the crew.

Retrieve the result:

```bash
aws s3 cp s3://crew-workspace/outputs/report.md - --endpoint-url $XNS_ENDPOINT
```

The bucket is created if it does not exist. Override its name with
`XNS_WORKSPACE_BUCKET`.

## How the handoff works

The researcher's task says: write the brief, store it under
`work/brief.md`, and reply with only that key. The reporter's task says: a
brief is stored at `work/brief.md`, fetch it with `read_artifact`. The
prompt carries an object key and the expected format; the receiving agent
retrieves the artifact itself.

Observed in a run against the live gateway: the researcher stored 1,782
bytes at `work/brief.md`, the reporter called `read_artifact` on that key,
and wrote 3,087 bytes to `outputs/report.md`. The brief's text never
appeared in the reporter's prompt.

**Behavior worth knowing before you build on it:**

- `work/brief.md` is overwritten on each run. Add a run ID to the prefix
  if you need history.
- `read_artifact` on a missing key returns a message beginning `ERROR:`
  rather than raising, so an agent can recover instead of failing the
  task. It does not retry.
- Bad credentials or an unreachable endpoint fail immediately at startup,
  before any agent runs.
- Nothing detects two workers writing the same key. Last write wins.

### Why these tools instead of the stock ones

`crewai_tools` ships `S3WriterTool` and `S3ReaderTool`, but both construct
their boto3 client without an `endpoint_url` (verified by reading the
installed source in crewai-tools 1.15.16), so they can only address AWS.
The workspace tools here are about thirty lines and take an endpoint,
which is all any S3-compatible gateway needs.

**Framework portability:** the storage pattern is framework-independent —
write an artifact to a known object key, pass the key plus its expected
format to the next worker, and let that worker fetch the object. This
recipe implements that contract with CrewAI. In a LangChain or LangGraph
node, [`langchain-xns`](https://pypi.org/project/langchain-xns/) provides
the same operations as a loader and a byte store.

## Configuration and security

Credentials resolve from `XNS_ENDPOINT` / `XNS_ACCESS_KEY_ID` /
`XNS_SECRET_ACCESS_KEY`, and fall back to `~/.xns/credentials`.
Environment wins. That is the whole precedence rule; the script does not
consult anything else.

The client sets path-style addressing and an explicit SigV4 signer:

```python
boto3.client("s3", endpoint_url=..., region_name="us-east-1",
             config=Config(signature_version="s3v4",
                           s3={"addressing_style": "path"}))
```

**The signer is not optional for presigned URLs.** botocore still presigns
with SigV2 by default, and this gateway is SigV4-only, so a presigned URL
from a default client returns **403 AccessDenied**. The error names access,
so it reads like a credential problem when it is a signing-version
problem. Setting `signature_version="s3v4"` fixes it. Ordinary `get_object`
and `put_object` calls already sign SigV4 and are unaffected.

Presigned links are useful for handing an artifact to something that has
no credentials — including a scoped upload URL for an agent you do not
want to hold keys. Treat them as bearer credentials: anyone holding the
link can use it until it expires, so keep the TTL short.

This recipe stores artifacts in the XNS endpoint you configure. Review
that deployment's storage, encryption, access-control, and retention
settings before putting sensitive material through it.

### What leaves your infrastructure

Task descriptions, agent reasoning, and any artifact text an agent reads
back are sent to the configured LLM provider. The artifacts themselves go
to the endpoint you configure. If the material cannot go to a hosted
model, point `CREW_MODEL` at a local one.

Deleting an input does not delete artifacts derived from it under `work/`
or `outputs/`. That propagation is yours to define.

## Running workers on separate machines

The storage interface and object-key convention are the same when workers
run apart: each process needs network reach to the endpoint, valid
credentials, and the prefix ownership convention. Deployment, networking,
and TLS still need configuring — splitting the crew is an operational
change, not a code change.

## MCP config

To let an assistant set up the Relayer and inspect the workspace:

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

It authenticates via OIDC and writes `~/.xns/credentials`. MCP-capable
clients can use the configured server where its tool contract and their
client support are compatible.

## Limitations

- **No coordination primitive.** Workers must not write the same key
  concurrently. Give each worker an owned prefix or a unique run ID. There
  is no locking and no compare-and-swap in this recipe.
- **Polling only.** Artifacts are discovered by listing or reading known
  keys. No notifications, queues, or event triggers.
- **No transactional workflow state.** An object upload proves bytes
  landed. It does not prove the work that produced them was correct or
  complete.
- **The context window still binds.** An agent must read an artifact into
  tokens to reason about it. The workspace removes repeated handoff
  duplication, not per-read context use.
- **Deployment is yours.** Separate workers need endpoint reachability,
  credentials, and appropriate TLS and network configuration.
- **Presigned URLs are bearer credentials.** Limit permissions and expiry,
  and use the SigV4 client configuration shown above.
- **Sequential demo.** The crew runs `Process.sequential`. Parallel crews
  sharing prefixes need the ownership convention enforced by you.

**Compatibility note (2026-08-19):** presigned GET and PUT, and
read-after-write visibility for the handoff, were exercised against the
current pre-release gateway build during development. Treat these as
behaviors confirmed on that build rather than as guaranteed properties;
re-check against the release you run.

## What to try next

- **Split the crew across two machines** and run the researcher and
  reporter separately against the same bucket.
- **Add a reviewing agent** that loops over drafts in `outputs/` — the
  re-read-heavy case this pattern suits best.
- **Add a run ID to the prefixes** so each execution keeps its own
  history instead of overwriting.
- **Point the MCP config at the same bucket** and let an assistant inspect
  the workspace the crew is using.
