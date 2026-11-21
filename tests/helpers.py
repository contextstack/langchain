"""Shared test harness utilities for live TCP servers, worker subprocesses, and run polling."""

from __future__ import annotations

import asyncio
import socket
import subprocess
from typing import Any

import httpx


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def poll_run(
    client: httpx.AsyncClient,
    api_base: str,
    headers: dict[str, str],
    run_id: str,
    target_statuses: set[str],
    *,
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(f"{api_base}/v1/runs/{run_id}", headers=headers)
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in target_statuses:
            return last
        await asyncio.sleep(0.2)
    raise AssertionError(f"Run {run_id} did not reach {target_statuses}; last={last}")
