"""LangGraph 路由函数：节点间的条件分派"""
from langgraph.graph import END

from core.state import AgentState


def route_idle(state: AgentState) -> str:
    """idle → 有输入则 router，否则 END（等主循环继续）"""
    return "router" if state.get("_has_input") else END


def route_router(state: AgentState) -> str:
    """router → 按 _intent 分派到下游节点"""
    intent = state.pop("_intent", "chat")
    if intent == "scene":
        return "scene_node"
    elif intent in ("planning", "confirm"):
        return "planning_prep"
    elif intent == "detail_planning":
        return "call_llm"
    else:
        return "call_llm"


def route_planning_prep(state: AgentState) -> str:
    """planning_prep → 始终去 call_llm"""
    return "call_llm"


def route_scene(state: AgentState) -> str:
    """scene_node → 回到 idle 等用户"""
    return "idle"


def route_call_llm(state: AgentState) -> str:
    """call_llm → 有工具调用则 execute_tools；详细规划进行中且有下一阶段则回 planning_prep；否则 idle"""
    if state.get("_tool_calls"):
        return "execute_tools"
    # 详细规划逐阶段进行中：还有未拆解的阶段 → 回 planning_prep 注入下一阶段
    if (state.get("macro_plan_confirmed")
            and not state.get("detail_plan_done")
            and state.get("current_phase_idx", -1) >= 0):
        phases = state.get("macro_phases", [])
        idx = state.get("current_phase_idx", 0)
        if idx < len(phases) - 1:
            return "planning_prep"
    return "idle"


def route_execute_tools(state: AgentState) -> str:
    """execute_tools → 有待确认问题则 idle（等用户回应），否则 call_llm（自动继续）"""
    return "idle" if state.get("pending_question") else "call_llm"
