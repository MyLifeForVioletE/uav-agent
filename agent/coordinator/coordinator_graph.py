"""Coordinator LangGraph 图构建"""
from langgraph.graph import StateGraph, END

from core.state import AgentState, Deps
from agent.coordinator.coordinator_nodes import (
    coordinator_idle_node,
    coordinator_router_node,
    parameter_collector_node,
    task_decomposer_node,
    task_allocator_node,
    fleet_dispatcher_node,
    conflict_resolver_node,
    result_aggregator_node,
)
from agent.coordinator.coordinator_routes import (
    route_coordinator_idle,
    route_after_collect,
    route_after_decompose,
    route_after_allocate,
    route_fleet_dispatch,
    route_after_conflict,
    route_after_aggregate,
)


def build_coordinator_graph(deps: Deps):
    """
    构建 Coordinator LangGraph：8 节点
    
    idle → router → parameter_collector → task_decomposer
    parameter_collector → [idle(等待用户补充) / task_decomposer]
    task_decomposer → [idle(等待确认) / task_allocator]
    task_allocator → [idle(等待确认) / fleet_dispatcher]
    fleet_dispatcher → [result_agg / conflict_resolver / fleet_dispatcher(轮询)]
    conflict_resolver → fleet_dispatcher
    result_agg → idle
    """
    builder = StateGraph(AgentState)
    
    # 定义节点（通过闭包捕获 deps）
    def _idle(s):
        return coordinator_idle_node(s, deps)
    
    def _router(s):
        return coordinator_router_node(s, deps)
    
    def _collector(s):
        return parameter_collector_node(s, deps)
    
    def _decomposer(s):
        return task_decomposer_node(s, deps)
    
    def _allocator(s):
        return task_allocator_node(s, deps)
    
    async def _dispatcher(s):
        return await fleet_dispatcher_node(s, deps)
    
    def _resolver(s):
        return conflict_resolver_node(s, deps)
    
    def _aggregator(s):
        return result_aggregator_node(s, deps)
    
    # 添加节点
    builder.add_node("idle", _idle)
    builder.add_node("router", _router)
    builder.add_node("parameter_collector", _collector)
    builder.add_node("task_decomposer", _decomposer)
    builder.add_node("task_allocator", _allocator)
    builder.add_node("fleet_dispatcher", _dispatcher)
    builder.add_node("conflict_resolver", _resolver)
    builder.add_node("result_agg", _aggregator)
    
    # 设置入口
    builder.set_entry_point("idle")
    
    # 添加边
    builder.add_conditional_edges("idle", route_coordinator_idle, {
        "router": "router",
        "task_allocator": "task_allocator",
        "fleet_dispatcher": "fleet_dispatcher",
        END: END,
    })
    
    builder.add_edge("router", "parameter_collector")
    
    builder.add_conditional_edges("parameter_collector", route_after_collect, {
        "idle": "idle",
        "task_decomposer": "task_decomposer",
    })
    
    builder.add_conditional_edges("task_decomposer", route_after_decompose, {
        "idle": "idle",
        "task_allocator": "task_allocator",
        END: END,
    })
    
    builder.add_conditional_edges("task_allocator", route_after_allocate, {
        "idle": "idle",
        "fleet_dispatcher": "fleet_dispatcher",
    })
    
    builder.add_conditional_edges("fleet_dispatcher", route_fleet_dispatch, {
        "fleet_dispatcher": "fleet_dispatcher",  # 继续轮询
        "conflict_resolver": "conflict_resolver",
        "result_agg": "result_agg",
        "idle": "idle",  # 子agent等待用户确认
    })
    
    builder.add_conditional_edges("conflict_resolver", route_after_conflict, {
        "fleet_dispatcher": "fleet_dispatcher",
    })
    
    builder.add_conditional_edges("result_agg", route_after_aggregate, {
        "idle": "idle",
    })
    
    return builder.compile()
