"""Coordinator 路由函数"""
import sys
from langgraph.graph import END
from core.state import AgentState


def route_coordinator_idle(state: AgentState) -> str:
    """idle → 根据意图分派"""
    intent = state.get("_coordinator_intent", "")
    
    # 用户确认后继续下一步
    if intent == "go_allocator":
        return "task_allocator"
    if intent == "go_dispatcher":
        return "fleet_dispatcher"
    
    # 有用户输入则进入router
    if state.get("user_input"):
        return "router"
    
    return END


def route_after_collect(state: AgentState) -> str:
    """parameter_collector → 始终去 task_decomposer"""
    return "task_decomposer"


def route_after_decompose(state: AgentState) -> str:
    """task_decomposer → 有sub_tasks则task_allocator，否则END"""
    if not state.get("sub_tasks"):
        return END
    return "task_allocator"


def route_after_allocate(state: AgentState) -> str:
    """task_allocator → 始终去 fleet_dispatcher"""
    return "fleet_dispatcher"


def route_fleet_dispatch(state: AgentState) -> str:
    """fleet_dispatcher → 根据状态决定下一步"""
    # 子agent在等待用户确认
    if state.get("_sub_awaiting"):
        return "idle"
    
    # 有冲突待解决
    if state.get("_active_conflicts"):
        return "conflict_resolver"
    
    # 所有子 agent 完成
    if state.get("_fleet_all_done"):
        return "result_agg"
    
    # 仍有活跃子 agent → 继续轮询
    return "fleet_dispatcher"


def route_after_conflict(state: AgentState) -> str:
    """conflict_resolver → fleet_dispatcher"""
    return "fleet_dispatcher"


def route_after_aggregate(state: AgentState) -> str:
    """result_agg → idle"""
    return "idle"
