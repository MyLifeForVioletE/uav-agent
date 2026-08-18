"""LangGraph 图构建"""
from langgraph.graph import StateGraph, END

from core.state import AgentState, Deps
from agent.nodes import idle_node, router_node, planning_prep, call_llm, execute_tools
from agent.routes import (
    route_idle, route_router, route_planning_prep,
    route_call_llm, route_execute_tools,
)


def build_graph(deps: Deps):
    """
    构建 LangGraph：5 节点
    idle → router → [planning_prep / call_llm]
    planning_prep → call_llm
    call_llm → execute_tools / idle
    execute_tools → idle / call_llm
    """
    builder = StateGraph(AgentState)

    def _idle(s):
        return idle_node(s, deps)
    async def _router(s):
        return await router_node(s, deps)
    async def _planning_prep(s):
        return await planning_prep(s, deps)
    async def _call_llm(s):
        return await call_llm(s, deps)
    async def _execute_tools(s):
        return await execute_tools(s, deps)

    builder.add_node("idle", _idle)
    builder.add_node("router", _router)
    builder.add_node("planning_prep", _planning_prep)
    builder.add_node("call_llm", _call_llm)
    builder.add_node("execute_tools", _execute_tools)

    builder.set_entry_point("idle")

    # idle → 有输入则 router，否则 END
    builder.add_conditional_edges("idle", route_idle, {
        "router": "router",
        END: END,
    })
    # router → 按意图分派
    builder.add_conditional_edges("router", route_router, {
        "planning_prep": "planning_prep",
        "call_llm": "call_llm",
    })
    # planning_prep → call_llm
    builder.add_conditional_edges("planning_prep", route_planning_prep, {
        "call_llm": "call_llm",
    })
    # call_llm → 有工具调用则 execute_tools；详细规划有下一阶段则回 planning_prep；否则 idle
    builder.add_conditional_edges("call_llm", route_call_llm, {
        "execute_tools": "execute_tools",
        "planning_prep": "planning_prep",
        "idle": "idle",
    })
    # execute_tools → 有待确认问题则等用户，否则 LLM 自动继续
    builder.add_conditional_edges("execute_tools", route_execute_tools, {
        "call_llm": "call_llm",
        "idle": "idle",
    })

    return builder.compile()
