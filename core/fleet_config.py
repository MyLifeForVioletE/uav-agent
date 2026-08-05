"""多机协同配置数据结构"""
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class CollaborationMode(str, Enum):
    """协同模式"""
    SINGLE = "single"  # 单机模式
    MULTI = "multi"    # 多机模式


class UAVRole(str, Enum):
    """UAV 角色"""
    LEADER = "leader"        # 领航机
    FOLLOWER = "follower"    # 跟随机
    SCOUT = "scout"          # 侦察机
    JAMMER = "jammer"        # 干扰机
    INTERFERER = "interferer" # 干扰机（别名）
    RELAY = "relay"          # 中继机


class ExecutionMode(str, Enum):
    """执行模式"""
    PARALLEL = "parallel"    # 并行执行
    SEQUENTIAL = "sequential"  # 顺序执行
    HYBRID = "hybrid"        # 混合执行（依赖图决定）


class SubAgentStatus(str, Enum):
    """子Agent状态"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    DONE = "done"
    ERROR = "error"


@dataclass
class UAVConfig:
    """单个UAV配置"""
    uav_id: str                           # 唯一标识
    uav_type: str = "generic"             # 类型（侦察/干扰/通用）
    role: UAVRole = UAVRole.SCOUT         # 角色
    capabilities: list[str] = field(default_factory=list)  # 能力列表
    
    # 资源约束
    max_payload_kg: float = 5.0           # 最大载荷
    max_flight_time_min: float = 60.0     # 最大飞行时间
    max_speed_mps: float = 20.0           # 最大速度
    
    # 通信配置
    comms_range_km: float = 10.0          # 通信范围
    data_rate_mbps: float = 10.0          # 数据传输率
    
    # 起始位置
    home_position: tuple[float, float] | None = None  # (lat, lon)
    
    def to_dict(self) -> dict:
        return {
            "uav_id": self.uav_id,
            "uav_type": self.uav_type,
            "role": self.role.value,
            "capabilities": self.capabilities,
            "max_payload_kg": self.max_payload_kg,
            "max_flight_time_min": self.max_flight_time_min,
            "max_speed_mps": self.max_speed_mps,
            "comms_range_km": self.comms_range_km,
            "data_rate_mbps": self.data_rate_mbps,
            "home_position": self.home_position,
        }


@dataclass
class SubTask:
    """子任务定义"""
    task_id: str                          # 唯一标识
    task_name: str                        # 任务名称
    goal: str                             # 任务目标
    
    # 分配信息
    assigned_uav: str | None = None       # 分配的UAV ID（未分配为None）
    assigned_uav_role: str | None = None  # 期望的角色
    
    # 依赖关系
    prerequisite_tasks: list[str] = field(default_factory=list)  # 前置任务ID
    shared_data_keys: list[str] = field(default_factory=list)    # 需要共享的数据键
    
    # 约束
    constraints: list[str] = field(default_factory=list)
    
    # 时间窗口
    earliest_start: float | None = None   # 最早开始时间（秒）
    deadline: float | None = None         # 截止时间（秒）
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "goal": self.goal,
            "assigned_uav": self.assigned_uav,
            "assigned_uav_role": self.assigned_uav_role,
            "prerequisite_tasks": self.prerequisite_tasks,
            "shared_data_keys": self.shared_data_keys,
            "constraints": self.constraints,
            "earliest_start": self.earliest_start,
            "deadline": self.deadline,
        }


@dataclass
class FleetPhase:
    """舰队阶段"""
    phase_id: str                         # 阶段ID
    phase_name: str                       # 阶段名称
    goal: str                             # 阶段目标
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    sub_tasks: list[SubTask] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "phase_id": self.phase_id,
            "phase_name": self.phase_name,
            "goal": self.goal,
            "execution_mode": self.execution_mode.value,
            "sub_tasks": [t.to_dict() for t in self.sub_tasks],
        }


@dataclass
class FleetPlan:
    """舰队级规划"""
    scenario: str                         # 多机协同任务场景描述
    uav_count: int                        # 参与协同的无人机数量
    uav_configs: list[UAVConfig] = field(default_factory=list)
    fleet_phases: list[FleetPhase] = field(default_factory=list)
    coordination_rules: list[dict] = field(default_factory=list)  # 协调规则
    
    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "uav_count": self.uav_count,
            "uav_configs": [c.to_dict() for c in self.uav_configs],
            "fleet_phases": [p.to_dict() for p in self.fleet_phases],
            "coordination_rules": self.coordination_rules,
        }


@dataclass
class InterUAVDependency:
    """跨机依赖关系"""
    from_uav: str                        # 依赖来源UAV
    to_uav: str                          # 依赖目标UAV
    depends_on_task: str                 # 依赖的任务ID
    data_key: str                        # 数据键
    data_type: str = "any"               # 数据类型
    
    def to_dict(self) -> dict:
        return {
            "from_uav": self.from_uav,
            "to_uav": self.to_uav,
            "depends_on_task": self.depends_on_task,
            "data_key": self.data_key,
            "data_type": self.data_type,
        }


def generate_fleet_id() -> str:
    """生成唯一的Fleet ID"""
    import uuid
    return f"fleet_{uuid.uuid4().hex[:8]}"
