# ContextStack Enterprise LangChain Reference Stack

A 100% Python, 100% open-source reference monorepo demonstrating an **enterprise-grade agent platform** built "the LangChain stack way" — using **LangChain v1**, **LangGraph**, native **MCP (`langchain[mcp]` + FastMCP)**, **PostgreSQL (`pgvector` + checkpoints + store)**, **Redis**, **OpenTelemetry**, and self-hosted **Langfuse**.

Everything in v1 runs locally at **\$0 in licences**. All CI tests, offline evaluations, and failure drills run deterministically against mock models with zero API spend.

---

## Architecture (3-Plane Separation)

- **Control Plane (`packages/core`):** Signed JWT auth, frozen `RunContext` (`context_schema`), deterministic `PolicyEngine` with separation of duties, workflow `release_version` + `checkpoint_compat`, and resource bounds.
- **Execution Plane (`services/api`, `services/worker`, `packages/workflows`, `packages/mcp_servers`):**
  - FastAPI **never** executes a graph — it authenticates, writes to PostgreSQL `runs`, and streams reconnectable SSE (`Last-Event-ID`) from Redis Streams.
  - A separate worker process claims runs via `FOR UPDATE SKIP LOCKED`, holds fenced leases `(lease_owner, attempt)` with heartbeats, runs LangGraph with `durability="sync"`, and automatically reclaims crashed workers' runs.
  - Flagship workflow: **IT / Ops Change-Request & Incident Desk** (`StateGraph` outer orchestrator + LangChain v1 `create_agent` with `PIIMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `interrupt()`/`Command(resume=...)` HITL approval, idempotent writes, and deterministic end-state verification).
- **Evidence Plane (`packages/telemetry`, `evals/`):**
  - OpenTelemetry spans with mandatory PII masking → self-hosted Langfuse (with a zero-code env-var toggle to commercial LangSmith).
  - Separate, DB-enforced append-only PostgreSQL `audit_log` (role grants + trigger).
  - Offline evaluation suite (`evals/datasets/core.jsonl`) with deterministic ground-truth evaluators and a hand-labeled calibration set (`evals/datasets/calibration.jsonl`).

See [docs/oss-vs-commercial.md](./docs/oss-vs-commercial.md) for the complete build-vs-buy breakdown against **LangSmith Platform** and **LangSmith Deployments / Agent Server**.

---

## Quickstart (Core Profile — \$0, No API Keys Required)

```bash
cp .env.example .env
docker compose up -d --wait      # Starts Postgres (pgvector) on 5439 + Redis on 6389
uv sync                          # Installs the Python workspace
uv run ops-cli db migrate        # Creates schemas, append-only audit_log grants, LangGraph tables
uv run ops-cli seed              # Seeds tenant-alpha & tenant-beta runbooks, services, tickets
uv run pytest -v                 # Runs all unit, integration, and real-process failure drills
uv run ops-cli eval              # Runs the offline evaluation regression gate
```

### Run Individual Failure & Security Drills

```bash
uv run ops-cli drill crash-mid-inference
uv run ops-cli drill crash-after-write
uv run ops-cli drill restart-hitl
uv run ops-cli drill tenant-isolation
uv run ops-cli drill prompt-injection-in-tenant
uv run ops-cli drill version-upgrade
uv run ops-cli drill self-approval
uv run ops-cli drill double-approve
uv run ops-cli drill edit-recheck
uv run ops-cli drill cancel-after-write
```

---

## Interactive Demo (API + Worker + Authenticated MCP + Single-File Web UI)

To run the full authenticated Streamable HTTP MCP path, start the two FastMCP servers, then the worker and API:

```bash
uv run ops-mcp-infra                      # Streamable HTTP MCP server on :8101
uv run ops-mcp-ticketing                  # Streamable HTTP MCP server on :8102
MCP_TRANSPORT=http uv run ops-worker      # Claims runs & calls MCP servers with signed JWTs
uv run ops-api                            # FastAPI + single-file Demo UI on :8000
```

*(For quick zero-daemon local runs without starting the MCP servers, omit `MCP_TRANSPORT=http` to use direct Postgres store mode.)*

Open **`http://127.0.0.1:8000`** for the single-file Demo UI (persona/tenant switcher for `alice`, `bob`, `carol`, `dana`, authenticated `fetch()` SSE stream, and HITL Approval Inbox).

### Local LangGraph Studio

```bash
uv run langgraph dev
```

### Full Stack (LiteLLM Proxy + Self-Hosted Langfuse)

```bash
docker compose --profile full up -d
```

- **Langfuse UI:** `http://localhost:3000`
- **LiteLLM Proxy:** `http://localhost:4000`

---

## Note on Data Handling

Raw request text is stored unredacted in PostgreSQL (`runs` and checkpoints) as the tenant-scoped system of record. PII redaction is enforced at two independent boundaries: `PIIMiddleware` before any LLM call, and a span-masking processor/hook before any trace export.
