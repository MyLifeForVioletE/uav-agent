"""Coordinator 路由函数"""
import sys
from langgraph.graph import END
from core.state import AgentState


def route_coordinator_idle(state: AgentState) -> str:
    """idle → 根据意图分派"""
    # 有 pending_question 则等待用户确认，图结束
    if state.get("pending_question"):
        return END
    
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


def route_coordinator_router(state: AgentState) -> str:
    """router → 按意图分派"""
    intent = state.get("_coordinator_intent", "")
    if intent == "fleet_planning":
        return "parameter_collector"
    return "single_agent"


def route_after_collect(state: AgentState) -> str:
    """parameter_collector → 有pending_question则idle等确认，否则task_decomposer"""
    if state.get("pending_question"):
        return "idle"
    return "task_decomposer"


def route_after_decompose(state: AgentState) -> str:
    """task_decomposer → 有pending_question则idle等确认，否则task_allocator"""
    pq = state.get("pending_question")
    print(f"[DEBUG] route_after_decompose: pending_question={pq!r} sub_tasks={len(state.get('sub_tasks', []))}", file=sys.stderr, flush=True)
    if pq:
        return "idle"
    if not state.get("sub_tasks"):
        return END
    return "task_allocator"


def route_after_allocate(state: AgentState) -> str:
    """task_allocator → 有pending_question则idle等确认，否则fleet_dispatcher"""
    if state.get("pending_question"):
        return "idle"
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
