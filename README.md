# XNS AI Cookbooks

Working recipes for AI pipelines on [XNS](https://xns.tech) — S3-compatible
storage with no per-read charge.

Every time a RAG pipeline re-reads a corpus, a training job pulls a checkpoint,
or an agent fetches a shared artifact, the storage side of that read is free —
compute and model API costs are yours as usual. These recipes target the
workflows where repeated reads dominate the storage bill.

Each recipe states its limitations explicitly. These are starter recipes —
single-process, happy-path — and each one says exactly where that stops
being enough.

## Recipes

| Recipe | Pipeline | Status |
|--------|----------|--------|
| [Multimodal RAG](multimodal-rag/) | Video speech + frames → transcripts + vision captions cached in XNS → query | Ready |
| Agentic Document Parsing | PDFs/spreadsheets → MCP-driven extraction loops | Coming soon |
| Fine-Tune Checkpointing | Model weights ↔ GPU clusters via S3 multipart | Coming soon |
| Agent Workspace | CrewAI/AutoGen shared-disk pattern | Coming soon |

Each recipe includes three things:

1. **Architecture blueprint** — a text diagram showing where XNS sits in the
   pipeline and why reads being free matters at that point.
2. **Runnable script** — Python, under 60 seconds on a laptop once prerequisites
   are in place.
3. **Config block** — JSON to wire XNS into Claude Desktop, Cursor, or any MCP
   client.

## Prerequisites

You need a running XNS Relayer (the S3 gateway). Two paths:

**Docker Compose** (if you have a Linux host with Docker):

```bash
git clone https://github.com/xns-cloud/relayer-quickstart
cd relayer-quickstart
docker compose up -d
```

**AI-assisted setup** (the MCP server walks you through it):

```bash
claude mcp add relayer -- npx @xns-cloud/relayer-mcp@latest
```

Then ask the agent to *"set up XNS storage."* It handles prerequisites, account
registration, install, and credential provisioning. Either path writes
`~/.xns/credentials`, which every recipe reads automatically.

## Links

- [XNS](https://xns.tech) — self-hosted S3-compatible storage, zero egress
- [relayer-quickstart](https://github.com/xns-cloud/relayer-quickstart) — Docker Compose setup
- [relayer-mcp](https://github.com/xns-cloud/relayer-mcp) — MCP server for AI code assistants
- [langchain-xns](https://github.com/xns-cloud/langchain-xns) — LangChain loaders and stores
- [S3 compatibility](https://xns.tech/s3-compatibility) — conformance test results

## License

[Apache-2.0](LICENSE)
