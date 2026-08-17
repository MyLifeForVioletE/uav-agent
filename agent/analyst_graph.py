"""分析 agent 专用图：直接调算法工具，无宏观/详细规划、无用户确认

结构：
idle → call_llm → execute_tools ─┬→ call_llm（还有工具调用，继续）
                                  └→ END（LLM 输出最终分析结果）
"""
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from core.state import AgentState, Deps
from agent.nodes import call_llm, execute_tools


def analyst_idle_node(state: AgentState, deps: Deps = None) -> AgentState:
    """分析图 idle 节点：注入任务输入，标记是否有输入"""
    state["_tool_calls"] = None
    state["_last_response"] = None

    # 缺参重试：指挥回复后注入待重试动作，直接走 execute_tools 重新执行
    retry = state.get("_inject_retry_action")
    if retry:
        state["_inject_retry_action"] = None
        state["_tool_calls"] = [{
            "name": retry.get("tool_name", ""),
            "args": retry.get("tool_inputs") or {},
            "id": "call_retry_0",
        }]
        state["_last_response"] = {"content": "重试执行缺参动作"}
        state["_has_input"] = True
        state["output"] = ""
        state["pending_question"] = ""
        return state

    uid = state.get("user_input")
    uid = uid.strip() if uid else ""
    if not uid:
        state["_has_input"] = False
        return state

    if not state.get("original_scenario"):
        state["original_scenario"] = uid

    state["messages"].append(HumanMessage(content=uid))
    state["output"] = ""
    state["pending_question"] = ""
    state["user_input"] = ""     # 清空，防止回流
    state["_has_input"] = True
    return state


def route_analyst_idle(state: AgentState) -> str:
    """idle → 有输入则 call_llm；注入的重试工具调用则直接 execute_tools，否则结束"""
    if state.get("_tool_calls"):
        return "execute_tools"
    return "call_llm" if state.get("_has_input") else END


def route_analyst_call_llm(state: AgentState) -> str:
    """call_llm → 有工具调用则 execute_tools，否则输出最终结果并结束"""
    return "execute_tools" if state.get("_tool_calls") else END


def route_analyst_execute_tools(state: AgentState) -> str:
    """execute_tools → 缺参挂起则结束（由 SubAgent 上报指挥）；无待确认问题则回 call_llm，否则结束"""
    if state.get("_analyst_param_pending"):
        return END
    return "call_llm" if not state.get("pending_question") else END


def build_analyst_graph(deps: Deps):
    """构建分析 agent 图：3 节点，无规划/确认逻辑"""
    builder = StateGraph(AgentState)

    def _idle(s):
        return analyst_idle_node(s, deps)

    async def _call_llm(s):
        return await call_llm(s, deps)

    async def _execute_tools(s):
        return await execute_tools(s, deps)

    builder.add_node("idle", _idle)
    builder.add_node("call_llm", _call_llm)
    builder.add_node("execute_tools", _execute_tools)

    builder.set_entry_point("idle")

    builder.add_conditional_edges("idle", route_analyst_idle, {
        "call_llm": "call_llm",
        "execute_tools": "execute_tools",
        END: END,
    })
    builder.add_conditional_edges("call_llm", route_analyst_call_llm, {
        "execute_tools": "execute_tools",
        END: END,
    })
    builder.add_conditional_edges("execute_tools", route_analyst_execute_tools, {
        "call_llm": "call_llm",
        END: END,
    })

    return builder.compile()
