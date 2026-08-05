"""Fleet 管理模块：多机协同的生命周期管理、状态同步、任务分配"""
from .fleet_manager import FleetManager
from .state_sync import StateSync, StateSyncMessage
from .task_allocator import TaskAllocator
