"""scene_node 节点：电磁场景构建"""
from core.state import AgentState, Deps
from tools.scene_builder import build_scene


def scene_node(state: AgentState, deps: Deps = None) -> AgentState:
    """scene_node 节点：独立处理场景构建"""
    uid = state.get("user_input", "")
    state["user_input"] = None
    if len(uid) > 200:
        result = build_scene(uid)
        state["output"] = result
        state["pending_question"] = result
    else:
        state["pending_question"] = "请提供详细的电磁场景描述，包括载体、装备、天线、项目等具体信息。"
    return state
