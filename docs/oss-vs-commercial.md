# Open Source vs. Commercial Boundaries: LangChain, LangGraph & LangSmith

This guide maps every enterprise capability built in **ContextStack LangChain** (100% MIT open source, \$0 in licences) to its commercial counterpart from LangChain, Inc. (**LangSmith Platform** and **LangSmith Deployments / Agent Server**), and highlights what **neither** side solves out of the box.

---

## 1. Component-by-Component Build vs. Buy Matrix

| Capability Built in This Repo | Open-Source Implementation (Here) | Commercial Counterpart (LangChain, Inc.) | Practical Trade-Off for Enterprise Buyers |
|---|---|---|---|
| **Agent & Graph Orchestration** | `langchain` v1 (`create_agent` + middleware) + `langgraph` (`StateGraph`, `interrupt`, `Command`) | None — core SDKs are MIT | Zero lock-in. Portable across self-hosted or managed runtimes. |
| **Durable Checkpointing & Long-Term Memory** | `AsyncPostgresSaver` & `AsyncPostgresStore` (`langgraph-checkpoint-postgres`) on self-managed Postgres | Managed Postgres persistence inside LangSmith Deployment | OSS is free, but your team owns connection pooling, backups, schema migrations, and checkpoint TTL pruning. |
| **Task Queue, Worker Leases & Crash Recovery** | Postgres `runs` table (`FOR UPDATE SKIP LOCKED`), fenced leases `(lease_owner, attempt)`, heartbeat, reclaim loop (`services/worker`) | **LangSmith Deployment / Agent Server** task queue & queue workers | **This is the core of what Agent Server sells.** Building it teaches every failure mode; buying Agent Server saves engineering & operating worker fleets, heartbeats, and horizontal autoscaling. |
| **Threads, Runs & Reconnectable SSE API** | Custom FastAPI app (`services/api`) + Redis Streams (`Last-Event-ID`) + cancel signals | Agent Server REST/Streaming API (threads, runs, assistants, background runs) | Ours is a focused, auditable subset. Agent Server ships a broader standardized API and SDK (`langgraph-sdk`). |
| **Concurrent Run ("Double-Texting") Policy** | Partial unique index `uq_runs_one_active_per_thread` ("reject" strategy returning `409`) | Agent Server supports **4 strategies**: `reject`, `enqueue`, `interrupt`, `rollback` | Not included in OSS `langgraph`. Buy or build if your UX allows users to interrupt/queue messages mid-run. |
| **Local Visual Debugging** | `langgraph dev` (free local Studio UI via `langgraph.json`) | Studio connected to remote production deployments | Free local Studio is great for development; production Studio debugging requires commercial LangSmith Deployment. |
| **Observability, Tracing & Datasets** | `langsmith[otel]` (`LANGSMITH_OTEL_ONLY=true`) → OTLP → self-hosted **Langfuse** (MIT OSS) | **LangSmith Platform** (Cloud SaaS, BYOC, or Self-Hosted Enterprise with ClickHouse/SmithDB) | Self-hosting Langfuse gives full OSS ownership, but requires running Postgres + ClickHouse + Redis + MinIO. Switching to LangSmith in this repo is an **env-var toggle** (zero code changes). |
| **Cron Jobs & Webhooks** | Not built (non-goal in v1) | Ships built-in with Agent Server | Easy win on the buy side if workflows need recurring schedules or outbound webhooks. |

---

## 2. What Neither OSS Nor Commercial LangChain Solves For You

Enterprise buyers often assume buying an enterprise AI platform solves security and reliability automatically. This repo proves firsthand that **these five layers must still be engineered by your platform team either way**:

1. **Downstream Tool Authorization & Tenant Isolation:** LangSmith auth handlers govern *threads and assistants*, not your internal billing, infra, or ticketing APIs. You still need signed internal tokens (no spoofable headers), SQL-level ACL filtering on RAG (`WHERE tenant_id = $1 AND allowed_roles && $2`), and application/database isolation.
2. **External Side-Effect Idempotency & Reconciliation:** Checkpointing only gives **at-least-once** execution. You still must persist an `operation_id` synchronously (`durability="sync"`) *before* any external write, send it as an idempotency key, reject payload mismatches, and reconcile `outcome_unknown` / cancel-after-write states.
3. **Human Governance & Separation of Duties:** Enforcing that a requester cannot approve their own change (`CHECK` constraint + API check), gating approvals on durable pauses, re-validating edited plans, and verifying `approved_plan_hash` before execution.
4. **Authoritative Audit Logging vs. Debug Traces:** Traces are high-volume debugging data (and must be PII-masked before export). Compliance requires a compact, DB-enforced append-only `audit_log` recording who requested, who approved, and what mutated.
5. **Production Hardening Beyond v2.1:** While this repo now enforces PostgreSQL **Row-Level Security (RLS)** on all 8 tenant-scoped tables (fail-closed by default via transaction-local `app.current_tenant`), production deployments still need HA ClickHouse/Postgres clustering, secrets rotation (Vault/KMS), and rainbow deployments for zero-downtime `checkpoint_compat` migrations.

---

## 3. Switching This Repo to Commercial LangSmith (Zero Code Changes)

In `.env`, flip three variables:

```env
TELEMETRY_BACKEND=langsmith
LANGSMITH_TRACING=true
LANGSMITH_OTEL_ENABLED=true
LANGSMITH_OTEL_ONLY=false
LANGSMITH_API_KEY=<your-langsmith-api-key>
```

No graph, worker, or API code changes are required.
