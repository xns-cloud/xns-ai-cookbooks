# Resume Fine-Tuning After Preemption with XNS Checkpoints

A GPU worker can disappear halfway through a fine-tune. This recipe saves
each completed checkpoint locally, uploads it to an XNS bucket, and lets a
new worker download the newest **complete** checkpoint before calling
`Trainer.train(resume_from_checkpoint=...)`.

The same pattern works for small LoRA or QLoRA adapters and for large
full-training checkpoints that include model and optimizer state. The
storage path is identical; only the amount of data and the checkpoint
cadence change.

This is a single-writer, single-process starter recipe. It demonstrates
reliable checkpoint handoff between machines, not distributed or sharded
checkpointing.

> **Cost boundary:** re-reading from XNS has no storage-side per-read
> charge. GPU time, stored capacity, and transfer time still apply. See
> [xns.tech/pricing](https://xns.tech/pricing) for rates.

```yaml
recipe:
  name: finetune-checkpointing
  mode: single-writer-single-process
  durable_store: xns_s3_bucket
  checkpoint_candidate_when: MANIFEST.json exists
  checkpoint_usable_when: all manifest-listed files download and pass verification
  resume_selection: highest numeric usable step
  verification:
    per_object: S3 checksum on upload
    per_checkpoint: manifest whole-file SHA-256 after download
  do_not:
    - write credentials to logs, images, or source control
    - treat a multipart checksum as a whole-file SHA-256
    - use one run prefix for multiple concurrent writers
    - resume from a prefix without MANIFEST.json
```

## What this recipe does

Checkpoints a training run to a bucket every N steps and recovers it on
any other machine — a replacement spot instance, a rented GPU from a
marketplace, or your own workstation. Treat training machines as
disposable workers; the bucket is the durable recovery location.

**Supported:** one trainer, one run ID, checkpoints written in increasing
step order.

**Not covered:** multiple concurrent writers to one run prefix, and
distributed sharded checkpointing (FSDP multi-node). Both need a different
design — see [Scale](#scale).

## Architecture

```
   ┌────────────────────────────────────────────────┐
   │                XNS Bucket                      │
   │  checkpoints/<run-id>/step-000500/             │
   │    the files your trainer wrote  ·             │
   │    MANIFEST.json  (uploaded last)              │
   │  (S3 API — no XNS storage-side per-read charge)│
   └────────┬──────────────────────────▲────────────┘
            │ pull newest complete      │ multipart push after
            │ checkpoint on start       │ each local save
            ▼                           │
   ┌────────────────────────────────────────────────┐
   │  GPU worker (spot / rented / local)            │
   │  Trainer → output_dir/checkpoint-<step>/       │
   │  resume_from_checkpoint=<verified local dir>   │
   └────────────────────────────────────────────────┘
            ▲
            └── preempted? a new worker reads the same
                prefix and continues from step N
```

| Situation | What gets pulled | What still costs |
|---|---|---|
| Preempted mid-run | Newest complete checkpoint | GPU time, stored capacity, and transfer time; no XNS storage-side per-read charge |
| Moving to a bigger GPU | Same | Same |
| Re-running eval on an old checkpoint | Selected step prefix | GPU time and transfer time; no XNS storage-side per-read charge |
| Sharing weights with a teammate | Selected prefix | Their compute and transfer time; no XNS storage-side per-read charge |

## Quickstart

**Prerequisites**

1. A running XNS Relayer with `~/.xns/credentials` written — see the
   [repo-level prerequisites](../README.md#prerequisites).
2. `pip install -r requirements.txt` — boto3, numpy, safetensors. No
   torch, no GPU. The demo simulates training so the transfer and
   recovery path can be exercised on a laptop.

**Runnable demo**

```bash
cd finetune-checkpointing
pip install -r requirements.txt

python checkpoint.py demo
```

It creates the bucket if needed, then narrates the whole cycle: save,
upload, manifest, a simulated preemption that deletes the local disk, a
second worker locating the newest complete checkpoint, download with
verification, and training continuing to a final step.

Checkpoint size is a parameter (`--size-mb`, default 48, large enough that
multipart upload actually engages). Completion time depends on your disk
and the network path to your Relayer; on a local gateway the default run
takes well under a minute.

Two other commands:

```bash
python checkpoint.py list --run-id <id>              # what is recoverable
python checkpoint.py pull --run-id <id> --dest ./resume
```

## Recovery contract

This is the part worth reading carefully — everything else is transport.

**Key layout**

```
checkpoints/<run-id>/step-<zero-padded-6>/<trainer files>
checkpoints/<run-id>/step-<zero-padded-6>/MANIFEST.json
```

Steps are zero-padded so lexicographic listing order matches numeric
order.

**A checkpoint becomes a recovery candidate when `MANIFEST.json` exists.
It is usable only after every file named in that manifest downloads and
passes verification.** The
manifest is uploaded *last*, after every other file in the prefix. An
upload interrupted partway leaves a prefix with no manifest, and resume
ignores it. Listing a prefix is not sufficient evidence that a checkpoint
is usable.

**Resume selects the highest-numbered step whose prefix has a manifest**,
then downloads every file the manifest lists and verifies each one against
the manifest's whole-file SHA-256. A mismatch aborts the resume rather
than training on bytes that are not the ones checkpointed.

| Situation | Action |
|---|---|
| New worker starts | Find the newest prefix containing `MANIFEST.json`; download and verify before resuming |
| Upload was interrupted | Ignore that prefix — it has no manifest |
| Local SHA-256 differs from the S3 checksum | Compare against `MANIFEST.json`, not against a multipart composite checksum |
| Multiple workers writing one run | Stop; give each worker its own run ID or add coordination |
| FSDP / multi-node run | Use a distributed checkpoint design instead |

### Two kinds of integrity check

Both are used here, and they answer different questions.

**Per object,** files upload with `ChecksumAlgorithm="SHA256"`, so the
gateway validates each object in transit. On a multipart upload the value
S3 returns is a *composite* — a checksum of the part checksums, with a
`-N` suffix. A multipart composite checksum will not equal a local
whole-file `sha256sum` of the same file. That mismatch is not corruption.
(Also don't depend on `head_object` reporting `PartsCount` —
it can come back empty; the part count is legible from the `-N` suffix.)

**Per checkpoint,** `MANIFEST.json` records each file's size and
whole-file SHA-256. Those are the hashes a local `sha256sum` can be
checked against, and the ones `pull` verifies after download. The
manifest's other job is marking the directory complete.

Verified against the current gateway build, which is pre-release — treat
these as behaviors we have confirmed, not as guarantees for every release.

## Integrate with Hugging Face Trainer

**Integration sketch** — illustrative, not run by the demo. `Trainer`
writes checkpoints to `output_dir/checkpoint-<step>/`; a callback mirrors
each one to the bucket after it is written.

```python
from pathlib import Path

from transformers import TrainerCallback

from checkpoint import client, upload_checkpoint

class XNSCheckpointCallback(TrainerCallback):
    def __init__(self, bucket, run_id):
        self.s3, self.bucket, self.run_id = client(), bucket, run_id

    def on_save(self, args, state, control, **kwargs):
        local = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        upload_checkpoint(self.s3, self.bucket, self.run_id, local, state.global_step)

trainer = Trainer(..., callbacks=[XNSCheckpointCallback("checkpoints", run_id)])
```

In this starter implementation, an upload failure fails the save callback
rather than silently continuing with an unprotected checkpoint. Production
jobs should add retry policy, timeout handling, and alerting appropriate to
their preemption risk.

The upload is synchronous, so it extends the time each save takes. On
multi-GB checkpoints over a slow link that is significant — measure it
before deciding your `save_steps`. A background uploader can reduce save
latency, but it needs explicit queueing, retry behavior, and the same
manifest-last completion rule: nothing may treat an upload as complete
until the manifest has been written. Treat it as a production extension,
not a drop-in change to this starter recipe.

To start a worker, pull before training. `pull` prints the directory it
verified:

```bash
python checkpoint.py pull --run-id "$RUN_ID" --dest ./resume
# prints: resume_from_checkpoint=resume/step-001500
```

```python
resume_dir = "resume/step-001500"  # the path printed by `checkpoint.py pull`
trainer.train(resume_from_checkpoint=resume_dir)
```

**Optional fast path** — for cluster boot scripts, `s5cmd` moves bulk data
with more parallelism than a single-threaded copy. Verified working
against the XNS gateway (upload, wildcard download, byte-identical
round-trip) with s5cmd 2.3.0:

`s5cmd` does not read `~/.xns/credentials`; it uses the standard AWS
credential chain, so it needs `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
and `AWS_REGION` supplied by your platform's secret manager.

```bash
# Load the scoped credentials using your environment's secret-injection
# method. Do not paste credentials into scripts or commit them to source
# control.

s5cmd --endpoint-url http://localhost:9000 \
      cp "s3://checkpoints/$RUN_ID/step-001500/*" ./resume/
```

Note that this path copies bytes without the manifest verification `pull`
performs; verify against `MANIFEST.json` before resuming from it.

Benchmark it against plain boto3 in your own environment before assuming
it is faster for your file sizes and link.

## Data flow — what leaves your infrastructure

This recipe does not send weights to a model-provider API; data flows
between your workers and your Relayer. There is no model provider in this
pipeline at all.

The exposure that does exist is the worker. On a rented marketplace GPU,
your weights and your XNS credentials sit on hardware you do not control.
Concretely:

- Use a credential scoped to the checkpoints bucket, not an account-wide
  one, where your setup allows it.
- Pass credentials through the environment. Never bake them into scripts,
  container images, or logs.
- On job completion, delete the local checkpoint copies and the
  credentials from the machine, and rotate anything that was used there.

Nothing garbage-collects old checkpoints in the bucket. `save_total_limit`
prunes the local directory only; retention on the remote side is yours to
define.

## MCP config

To let an AI assistant authenticate through the XNS Relayer MCP
integration and inspect checkpoint runs:

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

It authenticates via OIDC and writes `~/.xns/credentials`, which
`checkpoint.py` reads. The assistant should create or inspect only the
bucket and prefixes authorized by the credentials available to the Relayer.

## Limitations

### Correctness

- **Single writer.** One trainer per run ID. Two writers interleaving
  steps under one prefix will produce a history that the manifest-last
  convention cannot make coherent. There is no locking.
- **"Newest complete" is a convention, not a transaction.** A reader can
  list while a writer is mid-upload. The manifest gate covers the common
  case; it is not an atomic commit across objects.
- **Multipart checksums are composite.** Covered above — the manifest's
  whole-file hashes are the local-comparable check.
- **Overwrites have no undo.** Bucket versioning is supported by the
  gateway but off by default, and this recipe does not enable it.
  Checkpoints are immutable by convention: a new step is a new prefix.

### Scale

- **Whole-file, single-process transfers.** No sharding. At FSDP
  multi-node scale you want `torch.distributed.checkpoint` with an S3
  storage writer, which is a different program.
- **Optimizer state dominates full checkpoints.** Saving often on a large
  full fine-tune writes a lot of bytes each time.

### Operations

- **Checkpoint cadence is a tradeoff you own.** Choose it from how much
  lost work you can accept, how long a save takes, and how likely
  preemption is. Frequent saves cost throughput; rare saves cost work.
- **Upload is synchronous** in the callback sketch, so it adds to save
  time.
- **No retention policy.** Old prefixes accumulate until you remove them.

## What to try next

### Safe next steps

- **Wire the callback into a real LoRA run.** Adapters are small enough to
  push on every save and keep every step, then compare old adapters
  straight from the bucket.
- **Seed the base model into the bucket once** and have the cluster boot
  script pull base weights and the newest checkpoint together.
- **Scope and rotate a per-run credential** as part of your job template.

### Advanced patterns

- **A `latest.json` pointer** updated after each push, if listing gets
  slow at thousands of steps. It introduces a second thing that can be
  stale — the manifest gate still decides correctness.
- **Bucket versioning** on the checkpoints bucket, so an overwritten
  pointer object stays recoverable. Supported by the gateway; think
  through retention before enabling it.
- **Multi-provider training.** Workers that can reach the Relayer and
  authenticate to the bucket can recover from the same checkpoint prefix,
  which is where the absence of a storage-side per-read charge stops being
  an accounting detail. Transfer time still depends on bandwidth.
- **Distributed checkpointing** for sharded multi-node runs.
