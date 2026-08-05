"""Coordinator Agent 模块：多机协同的顶层协调器"""
from .coordinator_graph import build_coordinator_graph
from .coordinator_nodes import (
    coordinator_idle_node,
    coordinator_router_node,
    task_decomposer_node,
    task_allocator_node,
    fleet_dispatcher_node,
    conflict_resolver_node,
    result_aggregator_node,
)
