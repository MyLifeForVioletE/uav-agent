"""图状态定义与依赖注入"""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    """图状态的类型定义：贯穿所有节点的共享状态
    
    使用 total=False 使所有字段可选，支持向后兼容：
    - 单机模式：只填充原有字段
    - 多机模式：额外填充协同字段
    """
    # ═══════════════════════════════════════════
    #  原有字段（单机/多机通用）
    # ═══════════════════════════════════════════
    session_id: str           # 会话ID（用于Redis存储）
    messages: list            # 消息历史（HumanMessage / AIMessage / ToolMessage）
    user_input: str | None    # 用户最新输入（idle 节点处理后传给 call_llm）
    plan_text: str            # 规划阶段的文本展示（阶段标题列表）
    steps: list               # 结构化步骤列表（每个步骤包含 tool、phase、args 等）
    plan_generated: bool      # 是否已生成过宏观规划（用于判断任务切换等场景）
    macro_plan_confirmed: bool  # 用户是否已确认宏观规划
    original_scenario: str    # 用户最初输入的任务场景原文
    detail_plan_done: bool      # 详细规划是否已生成
    detail_plan_confirmed: bool # 用户是否已确认详细规划
    macro_phases: list          # 从宏观规划JSON解析出的阶段列表
    current_phase_idx: int      # 当前正在拆解的阶段索引（-1=未开始）
    detail_actions: list        # 逐阶段累积的原子动作列表
    _skill_injected: set       # 已注入的 skill 名称集合（如 {"task_planning", "scene"}）
    current_step_idx: int     # 当前执行到的步骤索引
    output: str               # 当前轮的输出文本（LLM 回复或工具结果）
    pending_question: str     # 等待用户输入的提示语
    calc_type: int | None     # 最近一次使用的计算类型枚举值
    file_path: str | None     # 最近一次使用的参数 XML 文件路径
    _tool_calls: list | None  # 最近一次 LLM 响应中的工具调用列表（临时存储）
    _last_response: object | None  # 最近一次 LLM 响应（AIMessage，临时存储）
    _intent: str                   # router 节点暂存的意图分类结果
    
    # ═══════════════════════════════════════════
    #  新增：多机协同字段（Coordinator 使用）
    # ═══════════════════════════════════════════
    # --- 协同模式标识 ---
    collaboration_mode: str          # "single" | "multi" (默认 "single")
    
    # --- Fleet 层级状态（仅 coordinator 使用）---
    fleet_id: str                    # 本次协同任务唯一ID
    uav_configs: list                # UAV配置列表 [{uav_id, uav_type, capabilities, ...}]
    uav_count: int                   # 参与协同的无人机数量
    
    # --- 子任务分配 ---
    sub_tasks: list                  # 协调器生成的子任务列表 [{task_id, task_name, goal, assigned_uav, ...}]
    sub_task_assignments: dict       # {uav_id: [task_ids]} 分配映射
    active_uav_ids: list             # 当前活跃的UAV ID列表
    
    # --- 子agent状态汇总 ---
    uav_states: dict                 # {uav_id: SubAgentStatus} 各子agent实时状态
    
    # --- 跨机协作 ---
    shared_results: dict             # {task_id: result_data} 已完成子任务的共享结果
    inter_uav_dependencies: list     # [{from_uav, to_uav, depends_on_task, data_key}]
    coordination_decisions: list     # 协调器的决策记录
    
    # --- 消息总线（Kafka）---
    coordinator_mailbox: list        # 指挥 agent 收到的消息列表（子agent上报/请示归档）

    # --- Coordinator 内部状态 ---
    _coordinator_intent: str         # coordinator 层级意图分类结果
    _fleet_plan_generated: bool      # 是否已生成舰队级规划
    _fleet_plan_confirmed: bool      # 用户是否已确认舰队级规划
    _fleet_all_done: bool            # 所有子agent是否已完成
    _active_conflicts: list          # 当前活跃的冲突列表
    _sub_agents_initialized: bool    # 子agent是否已初始化
    _fleet_tick_count: int            # dispatcher tick 计数器（防死循环）
    _sub_awaiting: str                # 正在等待用户确认的子任务ID
    _fleet_manager: object            # FleetManager 实例（运行时持久化）
    _dispatch_count: int              # dispatcher 调用计数（防死循环）
    _current_task_idx: int            # 当前正在执行的子任务索引（fleet_dispatcher 使用）
    _last_collected_input: str        # parameter_collector 记录的上次用户输入（防重复处理）

    # --- 单机图内部状态 ---
    _has_input: bool                  # idle 节点标记：本轮是否有新输入
    _last_input: str                  # idle 保存的原始用户输入（供 router 使用）
    _analyst_mode: bool               # 分析 agent 图标记：true 时工具调用走算法参数补齐管线


def default_agent_state() -> AgentState:
    """创建默认的单机模式状态（向后兼容）"""
    return {
        "session_id": "",
        "messages": [],
        "user_input": None,
        "plan_text": "",
        "steps": [],
        "plan_generated": False,
        "macro_plan_confirmed": False,
        "original_scenario": "",
        "detail_plan_done": False,
        "detail_plan_confirmed": False,
        "macro_phases": [],
        "current_phase_idx": -1,
        "detail_actions": [],
        "_skill_injected": set(),
        "current_step_idx": -1,
        "output": "",
        "pending_question": "",
        "calc_type": None,
        "file_path": None,
        "_tool_calls": None,
        "_last_response": None,
        "_intent": "",
        # 多机字段默认值
        "collaboration_mode": "single",
        "fleet_id": "",
        "uav_configs": [],
        "uav_count": 0,
        "sub_tasks": [],
        "sub_task_assignments": {},
        "active_uav_ids": [],
        "uav_states": {},
        "shared_results": {},
        "inter_uav_dependencies": [],
        "coordination_decisions": [],
        "coordinator_mailbox": [],
        "_coordinator_intent": "",
        "_fleet_plan_generated": False,
        "_fleet_plan_confirmed": False,
        "_fleet_all_done": False,
        "_active_conflicts": [],
        "_sub_agents_initialized": False,
        "_fleet_tick_count": 0,
        "_sub_awaiting": "",
        "_fleet_manager": None,
        "_dispatch_count": 0,
        "_current_task_idx": 0,
        "_last_collected_input": "",
        "_has_input": False,
        "_last_input": "",
    }


class Deps:
    """依赖注入：传递工具映射表、无工具 LLM、各规划器"""
    __slots__ = ("tools", "llm_no_tools", "macro_planner", "constraint_planner", "detail_planner", "decomposer_planner")
    def __init__(self, tools, llm_no_tools, macro_planner, constraint_planner, detail_planner, decomposer_planner=None):
        self.tools = tools
        self.llm_no_tools = llm_no_tools
        self.macro_planner = macro_planner
        self.constraint_planner = constraint_planner
        self.detail_planner = detail_planner
        self.decomposer_planner = decomposer_planner
