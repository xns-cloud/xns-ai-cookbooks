#!/usr/bin/env python3
"""
Checkpoint a fine-tune to XNS and resume it on a different machine.

The recovery protocol, which is the whole point:

  save locally -> upload every file -> upload MANIFEST.json LAST

A checkpoint prefix counts as complete only once its manifest exists. An
upload interrupted halfway leaves a prefix with no manifest, and resume
skips it. Resume takes the highest-numbered complete checkpoint, downloads
it, and verifies every file against the whole-file SHA-256 in the manifest
before training continues.

Key layout:

  checkpoints/<run-id>/step-000500/model.safetensors
  checkpoints/<run-id>/step-000500/optimizer.safetensors
  checkpoints/<run-id>/step-000500/trainer_state.json
  checkpoints/<run-id>/step-000500/MANIFEST.json     <- written last

Steps are zero-padded so lexicographic listing order matches numeric order.

Two notes on integrity, because the mismatch looks like corruption when it
is not:

  - S3 checksums are per object. On a multipart upload the returned
    checksum is a checksum-of-checksums with a "-N" suffix. It will never
    equal a local sha256sum of the same file. Don't compare them.
  - MANIFEST.json carries whole-file SHA-256 hashes. Those are the ones a
    local sha256sum can be checked against, and the ones this script
    verifies after download.

This demo needs no GPU and no torch. It writes synthetic safetensors with
numpy, so the transfer and recovery path is real while the training is
simulated. Wiring the same uploader into a real Trainer is a callback; the
README shows it.

Usage:
    pip install -r requirements.txt
    python checkpoint.py demo                    # full push/preempt/resume cycle
    python checkpoint.py demo --size-mb 128      # bigger checkpoints
    python checkpoint.py list --run-id <id>      # what is recoverable
    python checkpoint.py pull --run-id <id> --dest ./resume

Credentials resolve from ~/.xns/credentials, or from XNS_ENDPOINT /
XNS_ACCESS_KEY_ID / XNS_SECRET_ACCESS_KEY.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from safetensors.numpy import save_file

BUCKET = "checkpoints"
MANIFEST = "MANIFEST.json"

# A starting point, not gospel. Larger chunks and more concurrency help on
# fast links; tune against your own bandwidth and memory.
TRANSFER = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,
    multipart_chunksize=16 * 1024 * 1024,
    max_concurrency=8,
)


def resolve_config() -> dict:
    """Endpoint and credentials: environment first, then ~/.xns/credentials."""
    cfg = {
        "endpoint": os.environ.get("XNS_ENDPOINT"),
        "access_key_id": os.environ.get("XNS_ACCESS_KEY_ID"),
        "secret_access_key": os.environ.get("XNS_SECRET_ACCESS_KEY"),
        "region": os.environ.get("XNS_REGION", "us-east-1"),
    }
    if all(cfg[k] for k in ("endpoint", "access_key_id", "secret_access_key")):
        return cfg

    path = Path(os.environ.get("XNS_CREDENTIALS_FILE", "~/.xns/credentials")).expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        profile = raw["profiles"][os.environ.get("XNS_PROFILE") or raw.get("active_profile", "default")]
    except (OSError, ValueError, KeyError):
        sys.exit(
            "No XNS configuration found. Either export XNS_ENDPOINT, "
            "XNS_ACCESS_KEY_ID and XNS_SECRET_ACCESS_KEY, or let the XNS MCP "
            f"server write {path}."
        )
    for field in ("endpoint", "access_key_id", "secret_access_key"):
        cfg[field] = cfg[field] or profile.get(field)
    cfg["region"] = profile.get("region") or cfg["region"]
    return cfg


def client():
    """A plain boto3 S3 client. The gateway is SigV4-only and path-style.

    signature_version is set explicitly because botocore still presigns
    URLs with SigV2 by default, which this gateway rejects. Ordinary calls
    are unaffected, but a client built this way works for both.
    """
    cfg = resolve_config()
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        region_name=cfg["region"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_for(run_id: str, step: int) -> str:
    return f"{run_id}/step-{step:06d}/"


def save_local(directory: Path, step: int, size_mb: int) -> Path:
    """Write a synthetic checkpoint — stand-in for what a Trainer saves.

    A real checkpoint directory holds whatever your trainer produces:
    model weights, optimizer state, scheduler state, RNG state, and
    trainer metadata. The transfer path does not care which files those
    are, only that the manifest lists them.
    """
    out = directory / f"step-{step:06d}"
    out.mkdir(parents=True, exist_ok=True)

    floats = (size_mb * 1024 * 1024) // 4
    save_file({"weight": np.zeros(floats, dtype=np.float32)}, str(out / "model.safetensors"))
    save_file({"exp_avg": np.zeros(floats // 4, dtype=np.float32)}, str(out / "optimizer.safetensors"))
    (out / "trainer_state.json").write_text(json.dumps({"step": step}, indent=2))
    return out


def upload_checkpoint(s3, bucket: str, run_id: str, directory: Path, step: int) -> None:
    """Upload every file, then the manifest. Order is the correctness story."""
    prefix = prefix_for(run_id, step)
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != MANIFEST)

    manifest = {"run_id": run_id, "step": step, "files": {}}
    for path in files:
        s3.upload_file(
            str(path), bucket, prefix + path.name,
            Config=TRANSFER,
            ExtraArgs={"ChecksumAlgorithm": "SHA256"},
        )
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"    uploaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    # Last. Until this object exists the checkpoint is not recoverable,
    # which is exactly what we want an interrupted upload to look like.
    s3.put_object(
        Bucket=bucket, Key=prefix + MANIFEST,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ChecksumAlgorithm="SHA256",
    )
    print(f"    manifest written — step {step} is now recoverable")


def complete_checkpoints(s3, bucket: str, run_id: str) -> list[int]:
    """Steps that have a manifest, oldest first. Others are ignored."""
    steps = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{run_id}/", Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            name = entry["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if not name.startswith("step-"):
                continue
            try:
                s3.head_object(Bucket=bucket, Key=entry["Prefix"] + MANIFEST)
            except ClientError:
                print(f"    {name}: no manifest, skipping (incomplete upload)")
                continue
            steps.append(int(name.removeprefix("step-")))
    return sorted(steps)


def download_checkpoint(s3, bucket: str, run_id: str, step: int, dest: Path) -> Path:
    """Download a complete checkpoint and verify it against its manifest.

    Verification uses the manifest's whole-file SHA-256, not the object's
    S3 checksum — on a multipart upload those are composite values and
    would not match a local hash.
    """
    prefix = prefix_for(run_id, step)
    manifest = json.loads(
        s3.get_object(Bucket=bucket, Key=prefix + MANIFEST)["Body"].read()
    )

    out = dest / f"step-{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        target = out / name
        s3.download_file(bucket, prefix + name, str(target), Config=TRANSFER)
        actual = sha256(target)
        if actual != expected["sha256"]:
            raise RuntimeError(
                f"{name} failed verification: manifest {expected['sha256'][:16]}…, "
                f"downloaded {actual[:16]}…"
            )
        print(f"    {name}: {target.stat().st_size / 1e6:.1f} MB, sha256 verified")
    return out


def demo(args) -> None:
    s3 = client()
    run_id = args.run_id or f"demo-{int(time.time())}"
    try:
        s3.create_bucket(Bucket=args.bucket)
    except ClientError:
        pass  # already exists, which is the normal case

    print(f"run {run_id} — checkpoints every {args.save_steps} steps, "
          f"{args.size_mb} MB each\n")
    workdir = Path(tempfile.mkdtemp(prefix="xns-ckpt-"))
    try:
        # --- worker one: trains, checkpoints, then gets preempted ---
        for step in (args.save_steps, args.save_steps * 2):
            print(f"  step {step}: saving locally")
            local = save_local(workdir / "worker-one", step, args.size_mb)
            upload_checkpoint(s3, args.bucket, run_id, local, step)

        print("\n  *** preempted — worker one is gone, local disk with it ***\n")
        shutil.rmtree(workdir / "worker-one")

        # --- worker two: a different machine, recovering from the bucket ---
        print("  worker two starting, looking for a checkpoint to resume from")
        steps = complete_checkpoints(s3, args.bucket, run_id)
        if not steps:
            sys.exit("  no complete checkpoint found — nothing to resume")
        latest = steps[-1]
        print(f"  newest complete checkpoint: step {latest} "
              f"(of {len(steps)} complete)")

        resumed = download_checkpoint(s3, args.bucket, run_id, latest, workdir / "worker-two")
        state = json.loads((resumed / "trainer_state.json").read_text())
        print(f"\n  resuming from step {state['step']} -> "
              f"trainer.train(resume_from_checkpoint='{resumed}')")

        final = state["step"] + args.save_steps
        print(f"  step {final}: saving locally")
        local = save_local(workdir / "worker-two", final, args.size_mb)
        upload_checkpoint(s3, args.bucket, run_id, local, final)
        print(f"\nrun {run_id} finished on worker two at step {final}")
        print(f"inspect it:  python checkpoint.py list --run-id {run_id}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def list_run(args) -> None:
    s3 = client()
    steps = complete_checkpoints(s3, args.bucket, args.run_id)
    if not steps:
        print(f"no complete checkpoints under {args.bucket}/{args.run_id}/")
        return
    print(f"complete checkpoints for {args.run_id}: "
          f"{', '.join(str(s) for s in steps)}")
    print(f"resume point: step {steps[-1]}")


def pull(args) -> None:
    s3 = client()
    steps = complete_checkpoints(s3, args.bucket, args.run_id)
    if not steps:
        sys.exit(f"no complete checkpoint under {args.bucket}/{args.run_id}/")
    step = args.step or steps[-1]
    if step not in steps:
        sys.exit(f"step {step} has no manifest; complete steps are {steps}")
    try:
        out = download_checkpoint(s3, args.bucket, args.run_id, step, Path(args.dest))
    except RuntimeError as exc:
        # A hash mismatch means the bytes on disk are not the bytes that
        # were checkpointed. Resuming from them is worse than not resuming.
        sys.exit(f"\nrefusing to resume: {exc}")
    print(f"\nresume_from_checkpoint={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    # --bucket is shared, and accepted after the subcommand where a reader
    # would naturally type it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bucket", default=BUCKET)

    run = sub.add_parser("demo", parents=[common],
                         help="push, preempt, resume, continue")
    run.add_argument("--run-id")
    run.add_argument("--size-mb", type=int, default=48,
                     help="synthetic checkpoint size (default 48)")
    run.add_argument("--save-steps", type=int, default=500)
    run.set_defaults(func=demo)

    show = sub.add_parser("list", parents=[common],
                          help="show recoverable checkpoints")
    show.add_argument("--run-id", required=True)
    show.set_defaults(func=list_run)

    get = sub.add_parser("pull", parents=[common],
                         help="download a complete checkpoint")
    get.add_argument("--run-id", required=True)
    get.add_argument("--step", type=int)
    get.add_argument("--dest", default="./resume")
    get.set_defaults(func=pull)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
