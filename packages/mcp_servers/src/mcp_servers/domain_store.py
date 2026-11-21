"""Domain store for Mock Infra, Ticketing & Billing with transactional idempotency (spec §3.3)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import psycopg

from core.db import set_tenant_context
from knowledge.fixtures import SEED_ACCOUNTS, SEED_SERVICES, SEED_TICKETS


class IdempotencyConflictError(ValueError):
    """Raised when an operation_id is reused with a different request payload."""


class ResourceNotFoundError(LookupError):
    """Raised when a tenant-scoped resource does not exist."""


def compute_request_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DomainStore(Protocol):
    """Unified contract honored by both InMemoryDomainStore and PostgresDomainStore."""

    async def get_service_status(self, *, tenant_id: str, service_name: str) -> dict[str, Any]: ...

    async def list_recent_deploys(
        self, *, tenant_id: str, service_name: str
    ) -> list[dict[str, Any]]: ...

    async def get_operation_status(
        self, *, tenant_id: str, operation_id: str
    ) -> dict[str, Any]: ...

    async def execute_service_change(
        self,
        *,
        tenant_id: str,
        service_name: str,
        action: str,
        operation_id: str,
        target_version: str | None = None,
    ) -> dict[str, Any]: ...

    async def get_ticket(self, *, tenant_id: str, ticket_id: str) -> dict[str, Any]: ...

    async def update_ticket(
        self,
        *,
        tenant_id: str,
        ticket_id: str,
        status: str,
        note: str,
        operation_id: str,
    ) -> dict[str, Any]: ...

    async def get_account(self, *, tenant_id: str, account_id: str) -> dict[str, Any]: ...

    async def issue_refund(
        self,
        *,
        tenant_id: str,
        account_id: str,
        amount_cents: int,
        reason: str,
        operation_id: str,
    ) -> dict[str, Any]: ...


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
        self.accounts: dict[tuple[str, str], dict[str, Any]] = {
            (a.tenant_id, a.account_id): {
                "tenant_id": a.tenant_id,
                "account_id": a.account_id,
                "customer_name": a.customer_name,
                "balance_cents": a.balance_cents,
                "tier": a.tier,
                "status": a.status,
            }
            for a in SEED_ACCOUNTS
        }
        self.operations: dict[str, dict[str, Any]] = {}

    def _check_existing_op(
        self, *, operation_id: str, tenant_id: str, req_hash: str
    ) -> dict[str, Any] | None:
        existing = self.operations.get(operation_id)
        if existing is None:
            return None
        if existing["tenant_id"] != tenant_id or existing["request_hash"] != req_hash:
            raise IdempotencyConflictError(
                f"operation_id '{operation_id}' already used with a different payload"
            )
        return {**existing["result_payload"], "deduplicated": True}

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
        dedup = self._check_existing_op(
            operation_id=operation_id, tenant_id=tenant_id, req_hash=req_hash
        )
        if dedup is not None:
            return dedup

        svc = self.services.get((tenant_id, service_name))
        if svc is None:
            raise ResourceNotFoundError(f"Service '{service_name}' not found")

        if action == "rollback" and target_version:
            svc["current_version"] = target_version
        if action in {"rollback", "restart", "scale_up"}:
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
        dedup = self._check_existing_op(
            operation_id=operation_id, tenant_id=tenant_id, req_hash=req_hash
        )
        if dedup is not None:
            return dedup

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

    async def get_account(self, *, tenant_id: str, account_id: str) -> dict[str, Any]:
        acct = self.accounts.get((tenant_id, account_id))
        if acct is None:
            raise ResourceNotFoundError(f"Account '{account_id}' not found")
        return dict(acct)

    async def issue_refund(
        self,
        *,
        tenant_id: str,
        account_id: str,
        amount_cents: int,
        reason: str,
        operation_id: str,
    ) -> dict[str, Any]:
        if amount_cents <= 0:
            raise ValueError("Refund amount must be positive")
        req_hash = compute_request_hash(
            {
                "tenant_id": tenant_id,
                "resource_type": "account",
                "resource_id": account_id,
                "action": "issue_refund",
                "amount_cents": amount_cents,
                "reason": reason,
            }
        )
        dedup = self._check_existing_op(
            operation_id=operation_id, tenant_id=tenant_id, req_hash=req_hash
        )
        if dedup is not None:
            return dedup

        acct = self.accounts.get((tenant_id, account_id))
        if acct is None:
            raise ResourceNotFoundError(f"Account '{account_id}' not found")
        if acct["balance_cents"] < amount_cents:
            raise ValueError(
                f"Insufficient balance on account '{account_id}': "
                f"{acct['balance_cents']} < {amount_cents}"
            )

        acct["balance_cents"] -= amount_cents
        result_payload = {
            "operation_id": operation_id,
            "account_id": account_id,
            "amount_cents": amount_cents,
            "new_balance_cents": acct["balance_cents"],
            "reason": reason,
            "deduplicated": False,
        }
        self.operations[operation_id] = {
            "operation_id": operation_id,
            "tenant_id": tenant_id,
            "resource_type": "account",
            "resource_id": account_id,
            "action": "issue_refund",
            "request_hash": req_hash,
            "status": "succeeded",
            "result_payload": result_payload,
        }
        return result_payload


class PostgresDomainStore:
    """Transactional Postgres store enforcing single-TX mutation + mock_operations insert."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def _lookup_existing_op(
        self,
        cur: psycopg.AsyncCursor[Any],
        *,
        operation_id: str,
        tenant_id: str,
        req_hash: str,
    ) -> dict[str, Any] | None:
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
        if existing is None:
            return None
        if existing[0] != tenant_id or existing[1] != req_hash:
            raise IdempotencyConflictError(
                f"operation_id '{operation_id}' already used with different payload"
            )
        payload = existing[2] if isinstance(existing[2], dict) else json.loads(existing[2])
        return {**payload, "deduplicated": True}

    async def _run_idempotent_tx(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        req_hash: str,
        mutate: Callable[[psycopg.AsyncCursor[Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run a tenant-scoped mutation + mock_operations insert atomically.

        Handles simultaneous retries racing on the same `operation_id` by catching
        `UniqueViolation` after rollback and returning the committed deduplicated result.
        """
        try:
            async with (
                await psycopg.AsyncConnection.connect(self._dsn) as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                await set_tenant_context(conn, tenant_id)
                dedup = await self._lookup_existing_op(
                    cur, operation_id=operation_id, tenant_id=tenant_id, req_hash=req_hash
                )
                if dedup is not None:
                    return dedup

                result_payload = await mutate(cur)
                await cur.execute(
                    """
                    INSERT INTO mock_operations
                        (operation_id, tenant_id, resource_type, resource_id, action,
                         request_hash, status, result_payload)
                    VALUES (%s, %s, %s, %s, %s, %s, 'succeeded', %s::jsonb)
                    """,
                    (
                        operation_id,
                        tenant_id,
                        resource_type,
                        resource_id,
                        action,
                        req_hash,
                        json.dumps(result_payload),
                    ),
                )
                return result_payload
        except psycopg.errors.UniqueViolation as exc:
            async with (
                await psycopg.AsyncConnection.connect(self._dsn) as conn,
                conn.transaction(),
                conn.cursor() as cur,
            ):
                await set_tenant_context(conn, tenant_id)
                dedup = await self._lookup_existing_op(
                    cur, operation_id=operation_id, tenant_id=tenant_id, req_hash=req_hash
                )
                if dedup is not None:
                    return dedup
            raise IdempotencyConflictError(
                f"operation_id '{operation_id}' already used by another tenant"
            ) from exc

    async def get_service_status(self, *, tenant_id: str, service_name: str) -> dict[str, Any]:
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await set_tenant_context(conn, tenant_id)
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
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await set_tenant_context(conn, tenant_id)
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

        async def _mutate(cur: psycopg.AsyncCursor[Any]) -> dict[str, Any]:
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
            return {
                "operation_id": operation_id,
                "service_name": service_name,
                "action": action,
                "current_version": new_version,
                "status": new_status,
                "deduplicated": False,
            }

        return await self._run_idempotent_tx(
            tenant_id=tenant_id,
            operation_id=operation_id,
            resource_type="service",
            resource_id=service_name,
            action=action,
            req_hash=req_hash,
            mutate=_mutate,
        )

    async def get_ticket(self, *, tenant_id: str, ticket_id: str) -> dict[str, Any]:
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await set_tenant_context(conn, tenant_id)
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

        async def _mutate(cur: psycopg.AsyncCursor[Any]) -> dict[str, Any]:
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
            return {
                "operation_id": operation_id,
                "ticket_id": ticket_id,
                "status": status,
                "notes_count": len(notes),
                "deduplicated": False,
            }

        return await self._run_idempotent_tx(
            tenant_id=tenant_id,
            operation_id=operation_id,
            resource_type="ticket",
            resource_id=ticket_id,
            action="update_ticket",
            req_hash=req_hash,
            mutate=_mutate,
        )

    async def get_account(self, *, tenant_id: str, account_id: str) -> dict[str, Any]:
        async with (
            await psycopg.AsyncConnection.connect(self._dsn) as conn,
            conn.transaction(),
            conn.cursor() as cur,
        ):
            await set_tenant_context(conn, tenant_id)
            await cur.execute(
                """
                SELECT tenant_id, account_id, customer_name, balance_cents, tier, status
                  FROM mock_accounts
                 WHERE tenant_id = %s AND account_id = %s
                """,
                (tenant_id, account_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise ResourceNotFoundError(f"Account '{account_id}' not found")
        return {
            "tenant_id": row[0],
            "account_id": row[1],
            "customer_name": row[2],
            "balance_cents": int(row[3]),
            "tier": row[4],
            "status": row[5],
        }

    async def issue_refund(
        self,
        *,
        tenant_id: str,
        account_id: str,
        amount_cents: int,
        reason: str,
        operation_id: str,
    ) -> dict[str, Any]:
        if amount_cents <= 0:
            raise ValueError("Refund amount must be positive")
        req_hash = compute_request_hash(
            {
                "tenant_id": tenant_id,
                "resource_type": "account",
                "resource_id": account_id,
                "action": "issue_refund",
                "amount_cents": amount_cents,
                "reason": reason,
            }
        )

        async def _mutate(cur: psycopg.AsyncCursor[Any]) -> dict[str, Any]:
            await cur.execute(
                """
                SELECT balance_cents, status
                  FROM mock_accounts
                 WHERE tenant_id = %s AND account_id = %s
                   FOR UPDATE
                """,
                (tenant_id, account_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise ResourceNotFoundError(f"Account '{account_id}' not found")
            current_balance = int(row[0])
            if current_balance < amount_cents:
                raise ValueError(
                    f"Insufficient balance on account '{account_id}': "
                    f"{current_balance} < {amount_cents}"
                )
            new_balance = current_balance - amount_cents
            await cur.execute(
                """
                UPDATE mock_accounts
                   SET balance_cents = %s, updated_at = now()
                 WHERE tenant_id = %s AND account_id = %s
                """,
                (new_balance, tenant_id, account_id),
            )
            return {
                "operation_id": operation_id,
                "account_id": account_id,
                "amount_cents": amount_cents,
                "new_balance_cents": new_balance,
                "reason": reason,
                "deduplicated": False,
            }

        return await self._run_idempotent_tx(
            tenant_id=tenant_id,
            operation_id=operation_id,
            resource_type="account",
            resource_id=account_id,
            action="issue_refund",
            req_hash=req_hash,
            mutate=_mutate,
        )
