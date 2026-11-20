"""M0 Verification Checklist tests.

Confirms every fast-moving library surface referenced in spec.md §12 against the
pinned versions installed in uv.lock, plus the deterministic CI mocks.
"""

from __future__ import annotations

import inspect
from typing import Any, TypedDict

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from gateway.mocks import EMBEDDING_DIM, DeterministicMockEmbeddings, ScriptedMockChatModel
from workflows.incident_desk.graph import build_studio_graph


# ---------------------------------------------------------------------------
# Checklist #1: langchain.mcp.MCPAdapter importable from langchain[mcp]
# ---------------------------------------------------------------------------
def test_m0_01_langchain_mcp_adapter_import() -> None:
    from langchain.mcp import MCPAdapter  # noqa: PLC0415

    assert MCPAdapter is not None
    sig = inspect.signature(MCPAdapter.__init__)
    assert len(sig.parameters) >= 2


# ---------------------------------------------------------------------------
# Checklist #2: AsyncPostgresStore comes from langgraph-checkpoint-postgres
# ---------------------------------------------------------------------------
def test_m0_02_async_postgres_saver_and_store_imports() -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415
    from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: PLC0415

    assert AsyncPostgresSaver is not None
    assert AsyncPostgresStore is not None


# ---------------------------------------------------------------------------
# Checklist #3: StateGraph(..., context_schema=...) delivers runtime.context
#               inside nodes AND inside create_agent tools/middleware
# ---------------------------------------------------------------------------
class _DemoContext(BaseModel, frozen=True):
    tenant_id: str
    requester_id: str


class _DemoState(TypedDict):
    seen_in_node: str
    seen_in_agent_tool: str


async def test_m0_03_context_schema_in_node_and_create_agent() -> None:
    observed_tool_tenant: list[str] = []

    @tool
    def inspect_service(service_name: str, runtime: ToolRuntime[_DemoContext]) -> str:
        """Inspect a service using trusted runtime context."""
        observed_tool_tenant.append(runtime.context.tenant_id)
        return f"{runtime.context.tenant_id}:{service_name}:healthy"

    mock_model = ScriptedMockChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "inspect_service",
                        "args": {"service_name": "checkout-api"},
                    }
                ],
            ),
            AIMessage(content="Service checkout-api is healthy."),
        ]
    )

    sub_agent = create_agent(
        model=mock_model,
        tools=[inspect_service],
        context_schema=_DemoContext,
    )

    async def node_with_context(
        state: _DemoState, runtime: Runtime[_DemoContext]
    ) -> dict[str, Any]:
        agent_out = await sub_agent.ainvoke(
            {"messages": [{"role": "user", "content": "check checkout-api"}]},
            context=runtime.context,
        )
        return {
            "seen_in_node": runtime.context.tenant_id,
            "seen_in_agent_tool": observed_tool_tenant[0] if observed_tool_tenant else "",
            "_last": agent_out["messages"][-1].content,
        }

    builder = StateGraph(_DemoState, context_schema=_DemoContext)
    builder.add_node("check", node_with_context)
    builder.add_edge(START, "check")
    builder.add_edge("check", END)
    graph = builder.compile()

    result = await graph.ainvoke(
        {"seen_in_node": "", "seen_in_agent_tool": ""},
        context=_DemoContext(tenant_id="tenant-alpha", requester_id="alice"),
    )
    assert result["seen_in_node"] == "tenant-alpha"
    assert result["seen_in_agent_tool"] == "tenant-alpha"


# ---------------------------------------------------------------------------
# Checklist #4: LangChain v1 middleware names & imports
# ---------------------------------------------------------------------------
def test_m0_04_langchain_v1_middleware_imports() -> None:
    import langchain.agents.middleware as mw  # noqa: PLC0415

    assert hasattr(mw, "PIIMiddleware")
    assert hasattr(mw, "SummarizationMiddleware")
    assert hasattr(mw, "ModelCallLimitMiddleware")
    assert hasattr(mw, "ToolCallLimitMiddleware")


# ---------------------------------------------------------------------------
# Checklist #6: Studio graph compiles without checkpointer
# ---------------------------------------------------------------------------
async def test_m0_06_studio_graph_compiles_without_checkpointer() -> None:
    graph = build_studio_graph()
    assert graph.checkpointer is None
    # Default Studio invocation without passing context= uses StudioRunContext defaults
    out = await graph.ainvoke({"request_text": "status only check"})
    assert out["final_status"] == "no_action"


def test_gateway_chat_model_supports_bind_tools_and_structured_output() -> None:
    from core.config import Settings  # noqa: PLC0415
    from gateway.factory import get_chat_model  # noqa: PLC0415

    model = get_chat_model(
        alias="ops-primary",
        tenant_id="tenant-alpha",
        settings=Settings(model_mode="gateway"),
    )
    assert callable(getattr(model, "bind_tools", None))
    assert callable(getattr(model, "with_structured_output", None))


# ---------------------------------------------------------------------------
# Checklist #7 & #9: durability="sync", None crash-resume, Command(resume=...)
# ---------------------------------------------------------------------------
class _RecoveryState(TypedDict):
    steps: list[str]
    decision: str


async def test_m0_07_and_09_durability_sync_and_resume_semantics() -> None:
    fail_once = {"should_fail": True}

    def step_one(state: _RecoveryState) -> dict[str, Any]:
        return {"steps": [*state["steps"], "step_one"]}

    def step_two_flaky(state: _RecoveryState) -> dict[str, Any]:
        if fail_once["should_fail"]:
            fail_once["should_fail"] = False
            raise RuntimeError("simulated crash before step_two completes")
        return {"steps": [*state["steps"], "step_two"]}

    def await_approval(state: _RecoveryState) -> dict[str, Any]:
        decision = interrupt({"prompt": "approve?"})
        return {"decision": str(decision["decision"])}

    def step_three(state: _RecoveryState) -> dict[str, Any]:
        return {"steps": [*state["steps"], f"step_three:{state['decision']}"]}

    builder = StateGraph(_RecoveryState)
    builder.add_node("step_one", step_one)
    builder.add_node("step_two", step_two_flaky)
    builder.add_node("await_approval", await_approval)
    builder.add_node("step_three", step_three)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", "await_approval")
    builder.add_edge("await_approval", "step_three")
    builder.add_edge("step_three", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "m0-recovery-thread"}}

    # 1. Crash in step_two (after step_one is checkpointed with durability="sync")
    with pytest.raises(RuntimeError, match="simulated crash"):
        await graph.ainvoke(
            {"steps": [], "decision": ""},
            config=config,
            durability="sync",
        )

    snap_after_crash = await graph.aget_state(config)
    assert snap_after_crash.next == ("step_two",)
    assert snap_after_crash.values["steps"] == ["step_one"]

    # 2. Resume with None: step_one is NOT re-executed; pauses at interrupt()
    await graph.ainvoke(None, config=config, durability="sync")
    snap_paused = await graph.aget_state(config)
    assert snap_paused.next == ("await_approval",)
    assert len(snap_paused.interrupts) == 1
    assert snap_paused.values["steps"] == ["step_one", "step_two"]

    # 3. Resume from interrupt with Command(resume=...)
    final = await graph.ainvoke(
        Command(resume={"decision": "approved"}),
        config=config,
        durability="sync",
    )
    assert final["steps"] == ["step_one", "step_two", "step_three:approved"]
    assert final["decision"] == "approved"


# ---------------------------------------------------------------------------
# M0 Deliverable: Deterministic mock LLM + 1536-dim mock embeddings
# ---------------------------------------------------------------------------
def test_m0_deterministic_mocks() -> None:
    emb = DeterministicMockEmbeddings()
    v1 = emb.embed_query("checkout-api 502 runbook")
    v2 = emb.embed_query("checkout-api 502 runbook")
    v3 = emb.embed_query("completely different runbook")

    assert len(v1) == EMBEDDING_DIM == 1536
    assert v1 == v2
    assert v1 != v3

    llm = ScriptedMockChatModel(
        responses=[AIMessage(content="turn-1"), AIMessage(content="turn-2")]
    )
    assert llm.invoke("hi").content == "turn-1"
    assert llm.invoke("hi again").content == "turn-2"
