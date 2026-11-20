"""PostgreSQL schema DDL, append-only audit trigger, and role grants (spec §3)."""

from __future__ import annotations

import psycopg

SCHEMA_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_rw') THEN
        CREATE ROLE app_rw LOGIN PASSWORD 'app_rw_dev_password';
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS app_threads (
    thread_id         TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL,
    created_by        TEXT NOT NULL,
    creator_roles     TEXT[] NOT NULL,
    workflow_name     TEXT NOT NULL,
    release_version   TEXT NOT NULL,
    checkpoint_compat INT  NOT NULL,
    status            TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_app_threads_tenant
    ON app_threads (tenant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    thread_id         TEXT NOT NULL REFERENCES app_threads(thread_id),
    tenant_id         TEXT NOT NULL,
    kind              TEXT NOT NULL,
    input             JSONB NOT NULL,
    status            TEXT NOT NULL,
    attempt           INT  NOT NULL DEFAULT 0,
    max_attempts      INT  NOT NULL DEFAULT 3,
    lease_owner       TEXT,
    lease_expires_at  TIMESTAMPTZ,
    cancel_requested  BOOLEAN NOT NULL DEFAULT false,
    error_category    TEXT,
    error_detail      TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_runs_one_active_per_thread
    ON runs (thread_id) WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS idx_runs_claimable
    ON runs (status, lease_expires_at);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id        TEXT PRIMARY KEY,
    thread_id          TEXT NOT NULL REFERENCES app_threads(thread_id),
    tenant_id          TEXT NOT NULL,
    requested_by       TEXT NOT NULL,
    operation_id       TEXT NOT NULL UNIQUE,
    plan               JSONB NOT NULL,
    plan_hash          TEXT NOT NULL,
    approved_plan      JSONB,
    approved_plan_hash TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    decided_by         TEXT,
    decided_at         TIMESTAMPTZ,
    comment            TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (decided_by IS NULL OR decided_by <> requested_by)
);
CREATE INDEX IF NOT EXISTS idx_approvals_inbox
    ON approvals (tenant_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id         TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    allowed_roles  TEXT[] NOT NULL,
    title          TEXT NOT NULL,
    content        TEXT NOT NULL,
    embedding      VECTOR(1536) NOT NULL,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_kb_tenant ON kb_documents (tenant_id);

CREATE TABLE IF NOT EXISTS mock_services (
    tenant_id       TEXT NOT NULL,
    service_name    TEXT NOT NULL,
    current_version TEXT NOT NULL,
    status          TEXT NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, service_name)
);

CREATE TABLE IF NOT EXISTS mock_tickets (
    tenant_id       TEXT NOT NULL,
    ticket_id       TEXT NOT NULL,
    service_name    TEXT,
    title           TEXT NOT NULL,
    status          TEXT NOT NULL,
    notes           JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, ticket_id)
);

CREATE TABLE IF NOT EXISTS mock_operations (
    operation_id    TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    resource_type   TEXT NOT NULL,
    resource_id     TEXT NOT NULL,
    action          TEXT NOT NULL,
    request_hash    TEXT NOT NULL,
    status          TEXT NOT NULL,
    result_payload  JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    event_id         UUID PRIMARY KEY,
    tenant_id        TEXT NOT NULL,
    actor_id         TEXT NOT NULL,
    requested_by     TEXT,
    approved_by      TEXT,
    thread_id        TEXT NOT NULL,
    run_id           TEXT NOT NULL,
    workflow_version TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    tool_id          TEXT,
    operation_id     TEXT,
    decision         TEXT NOT NULL,
    details          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_thread
    ON audit_log (tenant_id, thread_id, created_at DESC);

-- Second barrier: DB trigger rejecting any UPDATE or DELETE on audit_log
CREATE OR REPLACE FUNCTION prevent_audit_log_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only; % is not permitted', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_log_append_only ON audit_log;
CREATE TRIGGER trg_audit_log_append_only
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION prevent_audit_log_mutation();

-- Role grants: app_rw has full access to operational tables, INSERT+SELECT ONLY on audit_log
GRANT USAGE ON SCHEMA public TO app_rw;
GRANT SELECT, INSERT, UPDATE, DELETE ON
    app_threads,
    runs,
    approvals,
    kb_documents,
    mock_services,
    mock_tickets,
    mock_operations
TO app_rw;

REVOKE ALL ON audit_log FROM app_rw;
GRANT SELECT, INSERT ON audit_log TO app_rw;
"""


async def apply_app_migrations(admin_dsn: str) -> None:
    """Apply application DDL, append-only trigger, and role grants using admin credentials."""
    async with await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True) as conn:
        await conn.execute(SCHEMA_DDL)
