"""idle 节点：输入处理与状态重置"""
from core.state import AgentState, Deps
from langchain_core.messages import HumanMessage


def idle_node(state: AgentState, deps: Deps = None) -> AgentState:
    """idle 节点：只做输入处理和状态重置，由 router 分发下游"""
    state["_tool_calls"] = None
    state["_last_response"] = None

    uid = state.get("user_input")
    uid = uid.strip() if uid else ""
    if not uid:
        state["_has_input"] = False
        if not state.get("pending_question"):
            state["pending_question"] = "请输入指令。"
        return state

    if not state.get("original_scenario"):
        state["original_scenario"] = uid

    state["_last_input"] = uid        # 保存原始输入供 router 使用
    state["messages"].append(HumanMessage(content=uid))  # 提前加到消息队列
    state["output"] = ""
    state["pending_question"] = ""
    state["user_input"] = ""     # 清空，防止后续回流
    state["_has_input"] = True   # 通知 route 去 router
    return state
