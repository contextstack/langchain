"""Boundary-first PostgreSQL Row-Level Security (RLS) proof tests (v2.1).

Connects as the real `app_rw` database role and executes raw SQL WITHOUT any
`WHERE tenant_id = ...` filter to prove PostgreSQL itself enforces tenant isolation.
"""

from __future__ import annotations

import psycopg
import pytest

from core.config import get_settings
from core.db import apply_app_migrations, set_tenant_context
from gateway.mocks import DeterministicMockEmbeddings
from knowledge.retriever import seed_knowledge_and_domain


async def test_postgres_rls_enforces_tenant_isolation_without_where_clauses() -> None:
    settings = get_settings()
    await apply_app_migrations(settings.database_admin_url)

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await seed_knowledge_and_domain(conn, embeddings=DeterministicMockEmbeddings())

        # 1. Fail-closed: with NO app.current_tenant set, app_rw sees 0 rows
        async with conn.transaction(), conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM kb_documents")
            assert (await cur.fetchone())[0] == 0  # type: ignore[index]
            await cur.execute("SELECT COUNT(*) FROM mock_services")
            assert (await cur.fetchone())[0] == 0  # type: ignore[index]
            await cur.execute("SELECT COUNT(*) FROM mock_tickets")
            assert (await cur.fetchone())[0] == 0  # type: ignore[index]

        # 2. Scoped to tenant-alpha: raw SELECT * without WHERE returns ONLY tenant-alpha
        async with conn.transaction(), conn.cursor() as cur:
            await set_tenant_context(conn, "tenant-alpha")
            await cur.execute("SELECT doc_id, tenant_id FROM kb_documents")
            kb_rows = await cur.fetchall()
            assert len(kb_rows) > 0
            assert all(r[1] == "tenant-alpha" for r in kb_rows)
            assert "rb-beta-billing-secret" not in {r[0] for r in kb_rows}

            await cur.execute("SELECT service_name, tenant_id FROM mock_services")
            svc_rows = await cur.fetchall()
            assert {r[0] for r in svc_rows} == {"checkout-api", "payments-worker"}
            assert all(r[1] == "tenant-alpha" for r in svc_rows)

        # 3. Cross-tenant INSERT blocked by Postgres RLS WITH CHECK policy
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with conn.transaction():
                await set_tenant_context(conn, "tenant-alpha")
                await conn.execute(
                    """
                    INSERT INTO mock_services (tenant_id, service_name, current_version, status)
                    VALUES ('tenant-beta', 'forged-svc', 'v1.0.0', 'healthy')
                    """
                )

        # 4. Transaction-local context does not leak after transaction ends
        async with conn.transaction(), conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM mock_services")
            assert (await cur.fetchone())[0] == 0  # type: ignore[index]
