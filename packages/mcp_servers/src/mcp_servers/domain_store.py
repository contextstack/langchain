"""Domain store for Mock Infra & Mock Ticketing with transactional idempotency (spec §3.3)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import psycopg

from knowledge.fixtures import SEED_SERVICES, SEED_TICKETS


class IdempotencyConflictError(ValueError):
    """Raised when an operation_id is reused with a different request payload."""


class ResourceNotFoundError(LookupError):
    """Raised when a tenant-scoped resource does not exist."""


def compute_request_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InMemoryDomainStore:
    """In-memory equivalent of PostgresDomainStore for fast, zero-Docker unit tests."""

    def __init__(self) -> None:
        self.services: dict[tuple[str, str], dict[str, Any]] = {
            (s.tenant_id, s.service_name): {
                "tenant_id": s.tenant_id,
                "service_name": s.service_name,
                "current_version": s.current_version,
                "status": s.status,
            }
            for s in SEED_SERVICES
        }
        self.tickets: dict[tuple[str, str], dict[str, Any]] = {
            (t.tenant_id, t.ticket_id): {
                "tenant_id": t.tenant_id,
                "ticket_id": t.ticket_id,
                "service_name": t.service_name,
                "title": t.title,
                "status": t.status,
                "notes": list(t.notes),
            }
            for t in SEED_TICKETS
        }
        self.operations: dict[str, dict[str, Any]] = {}

    async def get_service_status(self, *, tenant_id: str, service_name: str) -> dict[str, Any]:
        svc = self.services.get((tenant_id, service_name))
        if svc is None:
            raise ResourceNotFoundError(f"Service '{service_name}' not found")
        return dict(svc)

    async def list_recent_deploys(
        self, *, tenant_id: str, service_name: str
    ) -> list[dict[str, Any]]:
        svc = await self.get_service_status(tenant_id=tenant_id, service_name=service_name)
        return [
            {
                "service_name": service_name,
                "version": svc["current_version"],
                "status": svc["status"],
                "previous_version": "v2.4.0" if service_name == "checkout-api" else "v1.8.0",
            }
        ]

    async def get_operation_status(self, *, tenant_id: str, operation_id: str) -> dict[str, Any]:
        op = self.operations.get(operation_id)
        if op is None or op["tenant_id"] != tenant_id:
            return {"operation_id": operation_id, "status": "not_found"}
        return {
            "operation_id": operation_id,
            "status": op["status"],
            "resource_type": op["resource_type"],
            "resource_id": op["resource_id"],
            "action": op["action"],
            "result_payload": op["result_payload"],
        }

    async def execute_service_change(
        self,
        *,
        tenant_id: str,
        service_name: str,
        action: str,
        operation_id: str,
        target_version: str | None = None,
    ) -> dict[str, Any]:
        req_hash = compute_request_hash(
            {
                "tenant_id": tenant_id,
                "resource_type": "service",
                "resource_id": service_name,
                "action": action,
                "target_version": target_version,
            }
        )
        existing = self.operations.get(operation_id)
        if existing is not None:
            if existing["tenant_id"] != tenant_id or existing["request_hash"] != req_hash:
                raise IdempotencyConflictError(
                    f"operation_id '{operation_id}' already used with a different payload"
                )
            return {**existing["result_payload"], "deduplicated": True}

        svc = self.services.get((tenant_id, service_name))
        if svc is None:
            raise ResourceNotFoundError(f"Service '{service_name}' not found")

        if action == "rollback" and target_version:
            svc["current_version"] = target_version
            svc["status"] = "healthy"
        elif action in {"restart", "scale_up"}:
            svc["status"] = "healthy"

        result_payload = {
            "operation_id": operation_id,
            "service_name": service_name,
            "action": action,
            "current_version": svc["current_version"],
            "status": svc["status"],
            "deduplicated": False,
        }
        self.operations[operation_id] = {
            "operation_id": operation_id,
            "tenant_id": tenant_id,
            "resource_type": "service",
            "resource_id": service_name,
            "action": action,
            "request_hash": req_hash,
            "status": "succeeded",
            "result_payload": result_payload,
        }
        return result_payload

    async def get_ticket(self, *, tenant_id: str, ticket_id: str) -> dict[str, Any]:
        ticket = self.tickets.get((tenant_id, ticket_id))
        if ticket is None:
            raise ResourceNotFoundError(f"Ticket '{ticket_id}' not found")
        return dict(ticket)

    async def update_ticket(
        self,
        *,
        tenant_id: str,
        ticket_id: str,
        status: str,
        note: str,
        operation_id: str,
    ) -> dict[str, Any]:
        req_hash = compute_request_hash(
            {
                "tenant_id": tenant_id,
                "resource_type": "ticket",
                "resource_id": ticket_id,
                "action": "update_ticket",
                "status": status,
                "note": note,
            }
        )
        existing = self.operations.get(operation_id)
        if existing is not None:
            if existing["tenant_id"] != tenant_id or existing["request_hash"] != req_hash:
                raise IdempotencyConflictError(
                    f"operation_id '{operation_id}' already used with a different payload"
                )
            return {**existing["result_payload"], "deduplicated": True}

        ticket = self.tickets.get((tenant_id, ticket_id))
        if ticket is None:
            raise ResourceNotFoundError(f"Ticket '{ticket_id}' not found")

        ticket["status"] = status
        ticket["notes"].append({"text": note, "operation_id": operation_id})
        result_payload = {
            "operation_id": operation_id,
            "ticket_id": ticket_id,
            "status": ticket["status"],
            "notes_count": len(ticket["notes"]),
            "deduplicated": False,
        }
        self.operations[operation_id] = {
            "operation_id": operation_id,
            "tenant_id": tenant_id,
            "resource_type": "ticket",
            "resource_id": ticket_id,
            "action": "update_ticket",
            "request_hash": req_hash,
            "status": "succeeded",
            "result_payload": result_payload,
        }
        return result_payload


class PostgresDomainStore:
    """Transactional Postgres store enforcing single-TX mutation + mock_operations insert."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def get_service_status(self, *, tenant_id: str, service_name: str) -> dict[str, Any]:
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT tenant_id, service_name, current_version, status
                  FROM mock_services
                 WHERE tenant_id = %s AND service_name = %s
                """,
                (tenant_id, service_name),
            )
            row = await cur.fetchone()
        if row is None:
            raise ResourceNotFoundError(f"Service '{service_name}' not found")
        return {
            "tenant_id": row[0],
            "service_name": row[1],
            "current_version": row[2],
            "status": row[3],
        }

    async def list_recent_deploys(
        self, *, tenant_id: str, service_name: str
    ) -> list[dict[str, Any]]:
        svc = await self.get_service_status(tenant_id=tenant_id, service_name=service_name)
        return [
            {
                "service_name": service_name,
                "version": svc["current_version"],
                "status": svc["status"],
                "previous_version": "v2.4.0" if service_name == "checkout-api" else "v1.8.0",
            }
        ]

    async def get_operation_status(self, *, tenant_id: str, operation_id: str) -> dict[str, Any]:
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT status, resource_type, resource_id, action, result_payload
                  FROM mock_operations
                 WHERE operation_id = %s AND tenant_id = %s
                """,
                (operation_id, tenant_id),
            )
            row = await cur.fetchone()
        if row is None:
            return {"operation_id": operation_id, "status": "not_found"}
        payload = row[4] if isinstance(row[4], dict) else json.loads(row[4])
        return {
            "operation_id": operation_id,
            "status": row[0],
            "resource_type": row[1],
            "resource_id": row[2],
            "action": row[3],
            "result_payload": payload,
        }

    async def execute_service_change(
        self,
        *,
        tenant_id: str,
        service_name: str,
        action: str,
        operation_id: str,
        target_version: str | None = None,
    ) -> dict[str, Any]:
        req_hash = compute_request_hash(
            {
                "tenant_id": tenant_id,
                "resource_type": "service",
                "resource_id": service_name,
                "action": action,
                "target_version": target_version,
            }
        )
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT tenant_id, request_hash, result_payload
                  FROM mock_operations
                 WHERE operation_id = %s
                   FOR UPDATE
                """,
                (operation_id,),
            )
            existing = await cur.fetchone()
            if existing is not None:
                if existing[0] != tenant_id or existing[1] != req_hash:
                    raise IdempotencyConflictError(
                        f"operation_id '{operation_id}' already used with different payload"
                    )
                payload = (
                    existing[2]
                    if isinstance(existing[2], dict)
                    else json.loads(existing[2])
                )
                return {**payload, "deduplicated": True}

            await cur.execute(
                """
                SELECT current_version, status
                  FROM mock_services
                 WHERE tenant_id = %s AND service_name = %s
                   FOR UPDATE
                """,
                (tenant_id, service_name),
            )
            svc = await cur.fetchone()
            if svc is None:
                raise ResourceNotFoundError(f"Service '{service_name}' not found")

            new_version = (
                target_version if (action == "rollback" and target_version) else svc[0]
            )
            new_status = (
                "healthy" if action in {"rollback", "restart", "scale_up"} else svc[1]
            )

            await cur.execute(
                """
                UPDATE mock_services
                   SET current_version = %s, status = %s, updated_at = now()
                 WHERE tenant_id = %s AND service_name = %s
                """,
                (new_version, new_status, tenant_id, service_name),
            )
            result_payload = {
                "operation_id": operation_id,
                "service_name": service_name,
                "action": action,
                "current_version": new_version,
                "status": new_status,
                "deduplicated": False,
            }
            await cur.execute(
                """
                INSERT INTO mock_operations
                    (operation_id, tenant_id, resource_type, resource_id, action,
                     request_hash, status, result_payload)
                VALUES (%s, %s, 'service', %s, %s, %s, 'succeeded', %s::jsonb)
                """,
                (
                    operation_id,
                    tenant_id,
                    service_name,
                    action,
                    req_hash,
                    json.dumps(result_payload),
                ),
            )
            return result_payload

    async def get_ticket(self, *, tenant_id: str, ticket_id: str) -> dict[str, Any]:
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT tenant_id, ticket_id, service_name, title, status, notes
                  FROM mock_tickets
                 WHERE tenant_id = %s AND ticket_id = %s
                """,
                (tenant_id, ticket_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise ResourceNotFoundError(f"Ticket '{ticket_id}' not found")
        notes = row[5] if isinstance(row[5], list) else json.loads(row[5])
        return {
            "tenant_id": row[0],
            "ticket_id": row[1],
            "service_name": row[2],
            "title": row[3],
            "status": row[4],
            "notes": notes,
        }

    async def update_ticket(
        self,
        *,
        tenant_id: str,
        ticket_id: str,
        status: str,
        note: str,
        operation_id: str,
    ) -> dict[str, Any]:
        req_hash = compute_request_hash(
            {
                "tenant_id": tenant_id,
                "resource_type": "ticket",
                "resource_id": ticket_id,
                "action": "update_ticket",
                "status": status,
                "note": note,
            }
        )
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT tenant_id, request_hash, result_payload
                  FROM mock_operations
                 WHERE operation_id = %s
                   FOR UPDATE
                """,
                (operation_id,),
            )
            existing = await cur.fetchone()
            if existing is not None:
                if existing[0] != tenant_id or existing[1] != req_hash:
                    raise IdempotencyConflictError(
                        f"operation_id '{operation_id}' already used with different payload"
                    )
                payload = (
                    existing[2]
                    if isinstance(existing[2], dict)
                    else json.loads(existing[2])
                )
                return {**payload, "deduplicated": True}

            await cur.execute(
                """
                SELECT notes
                  FROM mock_tickets
                 WHERE tenant_id = %s AND ticket_id = %s
                   FOR UPDATE
                """,
                (tenant_id, ticket_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise ResourceNotFoundError(f"Ticket '{ticket_id}' not found")

            notes = row[0] if isinstance(row[0], list) else json.loads(row[0])
            notes.append({"text": note, "operation_id": operation_id})
            await cur.execute(
                """
                UPDATE mock_tickets
                   SET status = %s, notes = %s::jsonb, updated_at = now()
                 WHERE tenant_id = %s AND ticket_id = %s
                """,
                (status, json.dumps(notes), tenant_id, ticket_id),
            )
            result_payload = {
                "operation_id": operation_id,
                "ticket_id": ticket_id,
                "status": status,
                "notes_count": len(notes),
                "deduplicated": False,
            }
            await cur.execute(
                """
                INSERT INTO mock_operations
                    (operation_id, tenant_id, resource_type, resource_id, action,
                     request_hash, status, result_payload)
                VALUES (%s, %s, 'ticket', %s, 'update_ticket', %s, 'succeeded', %s::jsonb)
                """,
                (
                    operation_id,
                    tenant_id,
                    ticket_id,
                    req_hash,
                    json.dumps(result_payload),
                ),
            )
            return result_payload
