#!/usr/bin/env python3
"""
A shared workspace for CrewAI agents, backed by an XNS bucket.

Two agents exchange an artifact without passing its contents through the
LLM context. The researcher writes `work/brief.md`; the reporter is told
the key, fetches it, and writes `outputs/report.md`. The prompt carries a
key; the bucket carries the payload.

The workspace tools exist because `crewai_tools`' stock S3WriterTool and
S3ReaderTool construct their boto3 client without an `endpoint_url`
(verified in crewai-tools 1.15.16), so they can only reach AWS. These
tools take an endpoint, which is all a third-party S3 gateway needs.

Key layout — each prefix has one owner:

  inputs/   material the crew starts from   (you write, agents read)
  work/     agent-to-agent artifacts        (researcher writes)
  outputs/  final deliverables              (reporter writes)

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    python crew.py "your topic here"

    python crew.py --selftest      # workspace tools only, no LLM calls

Credentials resolve from ~/.xns/credentials, or from XNS_ENDPOINT /
XNS_ACCESS_KEY_ID / XNS_SECRET_ACCESS_KEY.
"""

import json
import os
import sys
from pathlib import Path
from typing import Type

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

BUCKET = os.environ.get("XNS_WORKSPACE_BUCKET", "crew-workspace")
MODEL = os.environ.get("CREW_MODEL", "gpt-4.1-mini")
BRIEF_KEY = "work/brief.md"
REPORT_KEY = "outputs/report.md"


def resolve_config() -> dict:
    """Environment first, then ~/.xns/credentials — the same order as
    langchain-xns, so a machine set up for one recipe works for all."""
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
        profile = raw["profiles"][
            os.environ.get("XNS_PROFILE") or raw.get("active_profile", "default")
        ]
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


class Workspace:
    """The bucket, wrapped in the three operations the agents need."""

    def __init__(self, bucket: str = BUCKET):
        cfg = resolve_config()
        self.bucket = bucket
        # Path-style addressing and an explicit SigV4 signer. The signer
        # matters for presigned URLs: botocore still presigns with SigV2 by
        # default, and this gateway is SigV4-only, so a default client
        # produces links that 403.
        self.s3 = boto3.client(
            "s3",
            endpoint_url=cfg["endpoint"],
            aws_access_key_id=cfg["access_key_id"],
            aws_secret_access_key=cfg["secret_access_key"],
            region_name=cfg["region"],
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def ensure_bucket(self) -> None:
        try:
            self.s3.create_bucket(Bucket=self.bucket)
        except ClientError:
            pass  # already exists

    def write(self, key: str, content: str) -> str:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=content.encode("utf-8"))
        return f"wrote {len(content)} bytes to {key}"

    def read(self, key: str) -> str:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")
        except ClientError as exc:
            # Agents recover better from a sentence than from a traceback.
            return f"ERROR: could not read {key}: {exc.response['Error'].get('Code', 'Unknown')}"

    def list(self, prefix: str = "") -> list[str]:
        keys, paginator = [], self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def share_link(self, key: str, seconds: int = 3600) -> str:
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=seconds
        )


class WriteInput(BaseModel):
    key: str = Field(description="Object key, e.g. 'work/brief.md'")
    content: str = Field(description="Full text to store at that key")


class ReadInput(BaseModel):
    key: str = Field(description="Object key to fetch, e.g. 'work/brief.md'")


class ListInput(BaseModel):
    prefix: str = Field(default="", description="Key prefix, e.g. 'work/'")


class WriteArtifact(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "write_artifact"
    description: str = (
        "Store text in the shared workspace under an object key. Use this to "
        "hand work to another agent instead of pasting it into your answer."
    )
    args_schema: Type[BaseModel] = WriteInput
    workspace: Workspace

    def _run(self, key: str, content: str) -> str:
        return self.workspace.write(key, content)


class ReadArtifact(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "read_artifact"
    description: str = "Fetch the text stored at an object key in the shared workspace."
    args_schema: Type[BaseModel] = ReadInput
    workspace: Workspace

    def _run(self, key: str) -> str:
        return self.workspace.read(key)


class ListArtifacts(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "list_artifacts"
    description: str = "List object keys in the shared workspace under a prefix."
    args_schema: Type[BaseModel] = ListInput
    workspace: Workspace

    def _run(self, prefix: str = "") -> str:
        keys = self.workspace.list(prefix)
        return "\n".join(keys) if keys else f"(nothing under '{prefix}')"


def build_crew(topic: str, workspace: Workspace) -> Crew:
    write, read = WriteArtifact(workspace=workspace), ReadArtifact(workspace=workspace)

    researcher = Agent(
        role="Researcher",
        goal=f"Produce a short, factual brief on: {topic}",
        backstory=(
            "You write tight briefs. You store your output in the shared "
            "workspace so colleagues can retrieve it without you repeating it."
        ),
        tools=[write],
        llm=MODEL,
        max_iter=3,  # a demo should not be able to loop
        verbose=True,
    )
    reporter = Agent(
        role="Reporter",
        goal="Turn a colleague's brief into a finished report",
        backstory=(
            "You never ask for content to be pasted to you. You are given a "
            "key, you fetch the artifact, and you write the final piece."
        ),
        tools=[read, write],
        llm=MODEL,
        max_iter=3,
        verbose=True,
    )

    research = Task(
        description=(
            f"Research this topic and write a brief of roughly 200 words: {topic}\n"
            f"Store the brief with write_artifact under the key '{BRIEF_KEY}'.\n"
            f"Reply with only that key."
        ),
        expected_output=f"The single key '{BRIEF_KEY}'.",
        agent=researcher,
    )
    report = Task(
        description=(
            f"A brief is stored at '{BRIEF_KEY}'. Fetch it with read_artifact.\n"
            f"Write a polished report of roughly 400 words based on it, and "
            f"store that with write_artifact under '{REPORT_KEY}'.\n"
            f"Reply with only that key."
        ),
        expected_output=f"The single key '{REPORT_KEY}'.",
        agent=reporter,
        context=[research],
    )
    return Crew(agents=[researcher, reporter], tasks=[research, report],
                process=Process.sequential, verbose=True)


def selftest(workspace: Workspace) -> None:
    """Exercise the workspace against the real gateway. No LLM involved."""
    workspace.ensure_bucket()
    probe = "work/_selftest.md"
    body = "# selftest\nwritten by crew.py --selftest\n"

    print(workspace.write(probe, body))
    got = workspace.read(probe)
    assert got == body, f"read back {got!r}, expected {body!r}"
    print(f"read back {len(got)} bytes, identical")

    keys = workspace.list("work/")
    assert probe in keys, f"{probe} missing from listing {keys}"
    print(f"listed work/ — {len(keys)} key(s), probe present")

    missing = workspace.read("work/does-not-exist.md")
    assert missing.startswith("ERROR:"), missing
    print(f"missing key handled: {missing.split(':')[0]}:{missing.split(':')[1]}")

    link = workspace.share_link(probe, seconds=60)
    import urllib.request

    fetched = urllib.request.urlopen(link, timeout=15).read().decode("utf-8")
    assert fetched == body, "presigned URL returned different content"
    print("presigned GET works and returns the same bytes")

    workspace.s3.delete_object(Bucket=workspace.bucket, Key=probe)
    print("\nselftest passed — workspace is reachable and behaves as documented")


def main() -> None:
    args = sys.argv[1:]
    workspace = Workspace()

    if "--selftest" in args:
        selftest(workspace)
        return

    topic = args[0] if args else "The current state of S3-compatible object storage"
    workspace.ensure_bucket()

    build_crew(topic, workspace).kickoff()

    print(f"\nworkspace contents: {', '.join(workspace.list()) or '(empty)'}")
    report = workspace.read(REPORT_KEY)
    if not report.startswith("ERROR:"):
        print(f"\n--- {REPORT_KEY} ({len(report)} bytes) ---\n{report[:600]}")
    print(
        f"\nfetch it:  aws s3 cp s3://{workspace.bucket}/{REPORT_KEY} - "
        f"--endpoint-url {resolve_config()['endpoint']}"
    )


if __name__ == "__main__":
    main()
