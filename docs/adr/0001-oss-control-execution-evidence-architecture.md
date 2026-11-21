# ADR 0001: 100% OSS 3-Plane Enterprise LangChain Reference Architecture

- **Status:** Accepted
- **Date:** 2026-11-21
- **Scope:** `contextstack-langchain` reference monorepo (`incident_desk` + `refund_ops`)

## 1. Context

Teams adopting LangChain v1 and LangGraph for internal operational workflows face a sharp divide between:

1. **Notebook / prototype demos**, where an HTTP handler directly invokes a graph, tools mutate state without idempotency or human governance, and observability is either absent or leaks PII.
2. **Commercial managed platforms** (LangSmith Platform + LangSmith Deployments / Agent Server), which provide managed queues, worker fleets, concurrency policies, and tracing, but obscure what infrastructure is actually required—and still leave application-level authorization, separation of duties, and external-write idempotency to the engineering team.

The goal of this repository is to build a **100% Python, 100% open-source, \$0-license** reference architecture using native LangChain v1 / LangGraph / MCP primitives, prove its reliability and security guarantees through real-process failure drills, and ground the build-vs-buy decision in concrete operational evidence.

## 2. Decision

We adopt a **3-plane separation** implemented as a `uv` workspace monorepo:

### A. Control Plane (`packages/core`)
- **Explicit Identity & Frozen Runtime Context:** Every request authenticates via signed JWT (`CallerIdentity`) and constructs a frozen `RunContext` passed via LangGraph `context_schema` (`Runtime[RunContext]` / `ToolRuntime[RunContext]`). Tenant identity is never trusted from LLM output or graph state.
- **Deterministic Policy & Separation of Duties:** All proposed mutations (`ChangePlan`) pass through `PolicyEngine` before reaching human approval, and edited plans are re-validated with a fresh `operation_id`. Requesters cannot approve their own changes (enforced at both the API layer and PostgreSQL `CHECK (decided_by IS NULL OR decided_by <> requested_by)`).
- **Explicit Demo Opt-In & Availability Trade-offs:** Local persona token issuance (`/v1/dev/token`) is fail-closed by default (`enable_dev_tokens: bool = False`) and must be explicitly enabled via `ENABLE_DEV_TOKENS=true` for local demos/tests. Redis is treated as strictly best-effort (live SSE + wake-ups + per-tenant rate-limit counter); during a Redis outage, the API rate limiter deliberately fails open and the worker degrades to PostgreSQL polling so committed approvals are never dropped.
- **Workflow Compatibility Gating:** Each workflow publishes a deterministic `release_version` hash and integer `checkpoint_compat`. Workers refuse incompatible checkpoints cleanly (`blocked_version` + `version_blocked` audit event) rather than corrupting in-flight state.

### B. Execution Plane (`services/api`, `services/worker`, `packages/workflows`, `packages/mcp_servers`)
- **Decoupled API and Worker Processes:** FastAPI (`services/api`) **never** imports `packages/workflows` or executes a graph. It authenticates callers, enforces per-tenant rate limits, writes to PostgreSQL `runs`, and streams reconnectable SSE (`Last-Event-ID`) from Redis Streams.
- **Postgres-Backed Lease-Fenced Worker Queue:** Separate `ops-worker` processes claim runs via `FOR UPDATE OF r, t SKIP LOCKED`, hold fenced leases `(lease_owner, attempt)` renewed by background heartbeats, run LangGraph with `durability="sync"` on `AsyncPostgresSaver` + `AsyncPostgresStore`, and automatically reclaim runs whose workers crash (`SIGKILL` / `os._exit(137)`).
- **Per-Thread Serialization & `multitask_strategy`:** A partial unique index (`uq_runs_one_running_per_thread`) plus `app_threads` row locking serializes concurrent workers per thread and supports `reject`, `enqueue`, and `interrupt` without stranding pending HITL approvals.
- **Authenticated Streamable HTTP MCP & Single-TX Idempotency:** LLM investigation agents (`create_agent` with `PIIMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`) receive **read-only tools only**. Write tools execute only in deterministic post-approval nodes using short-lived signed MCP JWTs (`approved_by`, `operation_id`), verified against `approved_plan_hash`, and deduplicated transactionally in `mock_operations` (`FOR UPDATE` + `request_hash` conflict detection).
- **Fail-Closed PostgreSQL Row-Level Security (RLS):** Enabled and forced on all 9 tenant-scoped tables for `app_rw`, requiring transaction-local `app.current_tenant` (or explicit `app.bypass_rls = 'on'` in system queue/seed transactions).
- **Reusable Runtime Proven by Two Workflows:** `WORKFLOW_REGISTRY` dispatches dynamically by `app_threads.workflow_name` across `incident_desk` (IT/Ops rollbacks & tickets) and `refund_ops` (Customer Billing refunds) with zero changes to `services/api`.

### C. Evidence Plane (`packages/telemetry`, `evals/`)
- **Dual-Track Evidence:** High-volume debug traces export via OpenTelemetry (`langsmith[otel]` / `OTel` -> self-hosted Langfuse v4, with a zero-code env-var toggle to commercial LangSmith) behind mandatory PII/secret masking. Compliance evidence is written separately to a PostgreSQL `audit_log` table protected by both least-privilege role grants (`SELECT, INSERT` only for `app_rw`) and a `BEFORE UPDATE OR DELETE` trigger.
- **Offline Regression Gate:** Deterministic evaluators and a calibrated judge run locally at \$0 via `uv run ops-cli eval`.

## 3. Alternatives Considered

1. **In-Process Graph Execution inside FastAPI:** Rejected. An API restart or timeout drops in-flight runs, breaks `Last-Event-ID` reconnection, and couples web concurrency to long-running LLM/tool latency.
2. **Commercial LangSmith Deployments / Agent Server from Day 1:** Deferred as a documented buy-side seam ([docs/oss-vs-commercial.md](../oss-vs-commercial.md)). Building the worker queue, lease fencing, and concurrency policies directly exposes the real failure modes teams must understand before choosing build vs. buy.
3. **Application-Only Tenant Filtering (No RLS):** Used in v1, upgraded to fail-closed PostgreSQL RLS in v2.1 so a missed `WHERE tenant_id = ...` clause returns zero rows rather than leaking cross-tenant data.
4. **Adding a Third Workflow or Full Production Infra (K8s, HA ClickHouse, Cron/Webhooks):** Rejected for this reference repo. Two distinct domains (`incident_desk` + `refund_ops`) already prove runtime reuse; further infrastructure should be driven by a concrete deployment requirement.

## 4. Consequences

### Positive
- Every reliability and security claim is backed by automated boundary tests (real `os._exit(137)` crashes, live TCP FastAPI + FastMCP + 3 parallel workers, MCP outage timeouts, raw-SQL RLS verification, and live OTLP/HTTP PII capture).
- Local development, CI, and client demos run deterministically at \$0 with no paid API keys required.
- Clean seams exist to swap in commercial LangSmith (env vars), real LLM providers (LiteLLM Proxy virtual keys), or commercial Agent Server without rewriting the Control or Evidence planes.

### Trade-offs & Known Limits
- **Operating the OSS Execution Plane has real cost:** Owning worker leases, claim-race serialization, heartbeat tuning, approval/enqueue state transitions, and checkpoint schema upgrades is non-trivial—which is the primary reason many teams buy Agent Server in production.
- **Workflow boilerplate:** `incident_desk` and `refund_ops` share the runtime, queue, auth, policy, and audit substrate, while still duplicating parts of the approval/execution graph wiring. Extracting a shared graph factory is straightforward if a third workflow is ever needed.
- **What is proven vs. not proven:** The mock test suite and offline eval prove *runtime durability, concurrency safety, tenant isolation, PII redaction, and approval governance*. They do **not** measure real-provider LLM reasoning accuracy; teams adopting this pattern for production should run a labeled evaluation against their target production models across both workflows and track both passes and misses.
