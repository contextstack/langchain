-- Enable pgvector extension for ACL-filtered runbook retrieval
CREATE EXTENSION IF NOT EXISTS vector;

-- Application role used by FastAPI, Worker, and MCP servers.
-- Note: audit_log grants (INSERT/SELECT only, no UPDATE/DELETE) are applied in migrations.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_rw') THEN
        CREATE ROLE app_rw LOGIN PASSWORD 'app_rw_dev_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE contextstack TO app_rw;
GRANT USAGE ON SCHEMA public TO app_rw;
