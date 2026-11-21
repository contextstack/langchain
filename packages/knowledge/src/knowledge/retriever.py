"""ACL-filtered pgvector runbook retriever and idempotent domain seeder (spec §3.3, §4.2)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.embeddings import Embeddings
from psycopg import AsyncConnection
from pydantic import BaseModel

from core.db import set_system_rls_bypass, set_tenant_context
from knowledge.fixtures import SEED_RUNBOOKS, SEED_SERVICES, SEED_TICKETS


class RetrievedRunbook(BaseModel, frozen=True):
    doc_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    distance: float


def format_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


async def retrieve_runbooks(
    conn: AsyncConnection[Any],
    *,
    embeddings: Embeddings,
    tenant_id: str,
    roles: frozenset[str],
    query: str,
    limit: int = 4,
) -> list[RetrievedRunbook]:
    """Query kb_documents with BOTH Postgres RLS and SQL-level tenant_id + role filtering."""
    if not roles:
        return []
    query_vec = format_pgvector(embeddings.embed_query(query))
    async with conn.transaction(), conn.cursor() as cur:
        await set_tenant_context(conn, tenant_id)
        await cur.execute(
            """
            SELECT doc_id, title, content, metadata, (embedding <=> %s::vector) AS distance
              FROM kb_documents
             WHERE tenant_id = %s
               AND allowed_roles && %s::text[]
             ORDER BY embedding <=> %s::vector ASC
             LIMIT %s
            """,
            (query_vec, tenant_id, sorted(roles), query_vec, limit),
        )
        rows = await cur.fetchall()
    return [
        RetrievedRunbook(
            doc_id=row[0],
            title=row[1],
            content=row[2],
            metadata=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
            distance=float(row[4]),
        )
        for row in rows
    ]


async def seed_knowledge_and_domain(
    conn: AsyncConnection[Any],
    *,
    embeddings: Embeddings,
) -> dict[str, int]:
    """Idempotently seed runbooks, mock services, and mock tickets."""
    async with conn.transaction():
        await set_system_rls_bypass(conn)
        for rb in SEED_RUNBOOKS:
            vec = format_pgvector(embeddings.embed_query(f"{rb.title}\n{rb.content}"))
            await conn.execute(
                """
                INSERT INTO kb_documents
                    (doc_id, tenant_id, allowed_roles, title, content, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (doc_id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    allowed_roles = EXCLUDED.allowed_roles,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (
                    rb.doc_id,
                    rb.tenant_id,
                    rb.allowed_roles,
                    rb.title,
                    rb.content,
                    vec,
                    json.dumps(rb.metadata),
                ),
            )

        for svc in SEED_SERVICES:
            await conn.execute(
                """
                INSERT INTO mock_services (tenant_id, service_name, current_version, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, service_name) DO UPDATE SET
                    current_version = EXCLUDED.current_version,
                    status = EXCLUDED.status,
                    updated_at = now()
                """,
                (svc.tenant_id, svc.service_name, svc.current_version, svc.status),
            )

        for ticket in SEED_TICKETS:
            await conn.execute(
                """
                INSERT INTO mock_tickets (tenant_id, ticket_id, service_name, title, status, notes)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (tenant_id, ticket_id) DO UPDATE SET
                    service_name = EXCLUDED.service_name,
                    title = EXCLUDED.title,
                    status = EXCLUDED.status,
                    notes = EXCLUDED.notes,
                    updated_at = now()
                """,
                (
                    ticket.tenant_id,
                    ticket.ticket_id,
                    ticket.service_name,
                    ticket.title,
                    ticket.status,
                    json.dumps(ticket.notes),
                ),
            )

    return {
        "runbooks": len(SEED_RUNBOOKS),
        "services": len(SEED_SERVICES),
        "tickets": len(SEED_TICKETS),
    }
