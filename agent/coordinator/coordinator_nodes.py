"""
Coordinator 节点函数：
1. coordinator_idle — 处理用户输入
2. coordinator_router — 判断是协同任务还是单机任务
3. parameter_collector — 收集用户参数，存入 Redis
4. task_decomposer — LLM 拆分多机子任务
5. task_allocator — 将子任务分配到具体 UAV
6. fleet_dispatcher — 启动/管理子 agent 实例，轮询状态
7. conflict_resolver — 解决跨 UAV 资源冲突
8. result_aggregator — 聚合各子 agent 结果
"""
import json
import re
import sys
import uuid
import requests

from core.state import AgentState, Deps
from core.config import OLLAMA_BASE, MODEL, BASE_DIR
from core.fleet_config import (
    SubTask, UAVConfig, FleetPhase, FleetPlan,
    ExecutionMode, UAVRole, generate_fleet_id,
)
from core.redis_manager import get_redis_manager
from fleet.fleet_manager import FleetManager


def _role_capabilities(role: str) -> list[str]:
    """根据 UAV 角色返回默认能力列表"""
    mapping = {
        "scout": ["radar", "optical"],
        "jammer": ["jammer"],
        "interferer": ["jammer"],
        "leader": ["radar", "optical"],
        "follower": ["radar"],
        "relay": ["comms_relay"],
    }
    return mapping.get(role, ["radar"])


def _load_algorithm_list_text() -> str:
    """从 algorithms.json 加载全部算法（含 roles 归属、输入输出概要），供任务分解时参考"""
    path = BASE_DIR / "algorithms.json"
    if not path.is_file():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    lines = []
    for cap in data.get("capabilities", []):
        name = cap.get("name", "")
        desc = cap.get("description", "")
        roles = cap.get("roles", [])
        params = ", ".join(f"{k}({v.get('type', '')})" for k, v in cap.get("input_schema", {}).items())
        outputs = ", ".join(k for k in cap.get("output_schema", {}))
        lines.append(
            f"- {name}：{desc}\n"
            f"  角色归属: {', '.join(roles) if roles else '未指定'}\n"
            f"  输入: {params or '无'}\n"
            f"  输出: {outputs or '无'}"
        )
    return "\n".join(lines)


def coordinator_idle_node(state: AgentState, deps: Deps) -> AgentState:
    """Coordinator idle 节点：处理用户输入"""
    uid = state.get("user_input", "")
    
    # 如果有 pending_question，正在等用户确认，不做任何操作
    if state.get("pending_question"):
        return state
    
    # 用户已确认（pending_question 已被 main loop 清除），根据状态设定意图
    if uid:
        if state.get("_sub_awaiting"):
            # 子agent在等待用户确认，回到dispatcher
            state["_coordinator_intent"] = "go_dispatcher"
        elif state.get("sub_tasks") and not state.get("sub_task_assignments"):
            state["_coordinator_intent"] = "go_allocator"
        elif state.get("sub_task_assignments") and not state.get("_fleet_all_done"):
            state["_coordinator_intent"] = "go_dispatcher"
    
    return state


def coordinator_router_node(state: AgentState, deps: Deps) -> AgentState:
    """
    Coordinator 路由节点：
    - 分析用户意图是否涉及多机协同
    """
    uid = state.get("user_input", "")
    if not uid:
        state["_coordinator_intent"] = "chat"
        return state
    
    state["collaboration_mode"] = "multi"
    state["_coordinator_intent"] = "fleet_planning"
    if not state.get("original_scenario"):
        state["original_scenario"] = uid
    
    return state


def parameter_collector_node(state: AgentState, deps: Deps) -> AgentState:
    """
    参数收集节点：
    - 询问用户是否还有信息要补充
    - 用户补充信息时调用 LLM 提取参数存入 Redis
    - 用户确认完毕后进入任务拆解
    """
    redis_mgr = get_redis_manager()
    session_id = state.get("session_id", "default")
    user_input = state.get("user_input", "")
    
    sys.stderr.write(f"[ParameterCollector] session_id={session_id}\n")
    sys.stderr.flush()
    
    # 第一次进入或 pending_question 已清除：询问用户是否提供完信息
    if not user_input or user_input == state.get("_last_collected_input", ""):
        state["pending_question"] = '请确认是否已提供所有必要信息？\n如有补充请输入，如已提供完请输入"确认"。'
        state["output"] = state["pending_question"]
        return state
    
    # 记录本次输入，避免重复处理
    state["_last_collected_input"] = user_input
    
    # 用户确认完毕
    if user_input.strip() in ("确认", "确定", "好了", "没有了", "完成", "yes", "ok"):
        state["output"] = "[参数收集完毕] 正在进入任务拆解..."
        state["pending_question"] = ""
        return state
    
    # 用户补充了信息：调用 LLM 结构化提取并合并到 Redis
    from langchain_core.messages import HumanMessage
    
    # 读取上下文模板 + 已有上下文
    template_path = BASE_DIR / "core" / "context_template.json"
    template_text = "（无）"
    if template_path.is_file():
        template_text = template_path.read_text(encoding="utf-8")
    existing_context = redis_mgr.get_context(session_id)
    existing_text = json.dumps(existing_context, ensure_ascii=False, indent=2) if existing_context else "（暂无）"
    
    extract_prompt = f"""分析用户输入，提取结构化信息并合并到上下文中。

上下文模板结构（请优先按此结构填充）：
{template_text}

已有上下文（已填入的值）：
{existing_text}

用户输入：{user_input}

要求：
1. 模板中 uavs 和 targets 是数组类型，用户说了多个就生成多个条目，每条带唯一 id
2. 用户说"三架无人机"就生成3个uav条目（id=UAV_1~UAV_3），没说的字段保持空值
3. 用户说"两个目标"就生成2个 target 条目（id=T1~T2），没说的字段保持空值
4. **"无人机到位置X"**表示该无人机的任务目标是该位置，应填入对应 target 的 Longitude 和 Latitude，**不影响该无人机的 Longitude/Latitude**
5. **"基地坐标"或"无人机在XX"**才决定无人机的 Longitude/Latitude，所有无人机在基地则 Longitude/Latitude=基地坐标
6. **"位置X的坐标是XX"**填入对应 target 的 Longitude 和 Latitude
7. 不要删除已有上下文中已有的条目和字段
8. 输出完整的 JSON，包含所有需要保留的字段
9. **禁止输出模板中不存在的字段（如 position、x、y）**；坐标必须写成 Longitude 和 Latitude 两个字段，输出前检查一遍确保没有 position 字段

只输出 JSON 对象，不要输出其他内容。"""
    
    try:
        response = deps.llm_no_tools.invoke([HumanMessage(content=extract_prompt)])
        raw = response.content.strip()
        
        # 提取 JSON 部分
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            new_context = json.loads(json_match.group())
            
            # 规范化 uavs 和 targets 的 ID，确保命名一致
            uavs = new_context.get("uavs")
            if isinstance(uavs, list):
                for i, uav in enumerate(uavs):
                    uav["id"] = f"UAV_{i+1}"
            targets = new_context.get("targets")
            if isinstance(targets, list):
                for i, tgt in enumerate(targets):
                    tgt["id"] = f"T{i+1}"
            
            # 合并到 Redis
            redis_mgr.update_context(session_id, new_context)
            
            # 调试：检查 Redis 中的 key
            sys.stderr.write(f"[ParameterCollector] 保存 context:{session_id}\n")
            sys.stderr.flush()
            raw_check = redis_mgr.get(f"context:{session_id}")
            sys.stderr.write(f"[ParameterCollector] 验证 get(context:{session_id})={raw_check}\n")
            sys.stderr.flush()
            
            # 显示当前上下文摘要
            full_context = redis_mgr.get_context(session_id)
            sys.stderr.write(f"[ParameterCollector] 结构化上下文: {json.dumps(full_context, ensure_ascii=False)}\n")
            sys.stderr.flush()
            
            # 统计字段数
            field_count = sum(len(v) if isinstance(v, dict) else 1 for v in full_context.values())
            state["output"] = f'[已记录 {field_count} 个字段] 继续补充或输入"确认"完成。'
        else:
            state["output"] = "[未提取到结构化信息] 请用更具体的格式描述，如坐标 (x, y)、文件名 xxx.json 等。"
    
    except Exception as e:
        sys.stderr.write(f"[ParameterCollector] LLM提取失败: {e}\n")
        sys.stderr.flush()
        state["output"] = f"[提取失败] 请用更具体的格式描述，如坐标 (x, y)、文件名 xxx.json 等。"
    
    state["pending_question"] = '请确认是否已提供所有必要信息？\n如有补充请输入，如已提供完请输入"确认"。'
    
    return state


def task_decomposer_node(state: AgentState, deps: Deps) -> AgentState:
    """
    子任务分解节点：
    - 基于用户的多机任务描述 + 已解析的 Redis 上下文
    - LLM 生成子任务列表
    """
    uid = state.get("original_scenario") or state.get("user_input", "")
    session_id = state.get("session_id", "")
    
    # 清除旧的任务数据，防止 LLM 解析失败时读到 stale 数据
    state["sub_tasks"] = []
    state["sub_task_assignments"] = {}
    state["uav_configs"] = []
    
    # 从 Redis context 获取实际的 UAV 数量和 target 数量
    context = {}
    uav_count = 1
    target_count = 0
    context_json_str = "(无)"
    try:
        redis_mgr = get_redis_manager()
        context = redis_mgr.get_context(session_id) or {}
        uavs = context.get("uavs", [])
        if isinstance(uavs, list):
            uav_count = len(uavs)
        targets = context.get("targets", [])
        if isinstance(targets, list):
            target_count = len(targets)
        context_json_str = json.dumps(context, ensure_ascii=False, indent=2)
    except Exception:
        pass
    
    # RAG 检索最相似的任务分解参考
    ref_text = ""
    if deps.decomposer_planner:
        try:
            contexts, _ = deps.decomposer_planner.plan(uid, retrieve_k=5, rerank_n=1)
            if contexts:
                ref_text = f"\n【参考类似场景的任务分解】\n{contexts}\n"
        except Exception:
            pass
    
    # 构造分解 prompt
    schema_path = BASE_DIR / "schema" / "plan" / "fleet_plan.json"
    schema_text = ""
    if schema_path.is_file():
        schema_text = schema_path.read_text(encoding="utf-8")
    
    algo_text = _load_algorithm_list_text()
    
    prompt = f"""你是多机协同任务规划专家。请参考类似场景的分解方式，将以下任务分解为子任务。

任务描述：{uid}

【当前已经解析的上下文】
{context_json_str}
{ref_text}

当前可用的无人机数量为 {uav_count}。

{f"JSON Schema 参考：{schema_text}" if schema_text else ""}

{f"【可用的算法工具（含角色归属）】\n算法列表仅用于说明各算法的归属与用途，**不代表每个算法都要生成一个子任务**。一个子任务可对应多个算法，也可能不需要算法。\n{algo_text}\n" if algo_text else ""}

【字段说明】
- scenario: 对本次任务场景的一句话概括，如"单架无人机对单个目标进行侦察"
- task_name: 子任务的动作名称，直接用用户说的动作词，注意加上对象，如"无人机1侦察目标2""信息处理agent分析目标1带宽、场强""无人机2对目标1实施干扰"
- goal: 子任务的具体目标，写明该子任务要产出什么。采集类子任务写原始数据（如"获取目标的扫频数据"），分析类子任务写最终期望信息（如"获取目标的频率、带宽、信号强度"）。**不同子任务的 goal 不要互相重复**。
- executor: "uav"=无人机agent执行，"info_processor"=信息处理agent执行，"coordinator"=主agent执行
- assigned_uav_role: 无人机角色，scout=侦察，jammer=干扰
- prerequisite_tasks: 前置依赖的任务ID列表，没有依赖则填[]
- constraints: 约束条件，没有则填[]

请以JSON格式返回任务分解方案：
{{
    "scenario": "string",
    "uav_count":"int",
    "sub_tasks": [
        {{
            "task_id": "T1",
            "task_name": "子任务名称",
            "goal": "子任务目标",
            "executor": "uav",
            "assigned_uav_role": "scout",
            "prerequisite_tasks": [],
            "constraints": []
        }}
    ]
}}

【重要】uav_count 必须严格等于 {uav_count}。
assigned_uav_role 必须与用户说的无人机类型一致。

【重要】executor 字段的含义：
- "uav"：由无人机子agent执行。适用于：
  - 抵近目标位置并使用传感器/设备进行侦察、采集原始数据（飞行是采集动作的一部分，**不单独成任务**）
  - 实施干扰
  - 任何需要物理执行、采集原始数据的任务
  - 以上【可用的算法工具】中"角色归属"含 uav 的算法，应由无人机agent执行
- "info_processor"：由信息处理agent执行。适用于：
  - 对无人机采集的数据进行分析处理（如扫频数据、信号强度分析、带宽/频率估计）
  - 数据汇总、统计、格式化输出
  - 任何基于无人机采集数据的分析计算任务
  - 以上【可用的算法工具】中"角色归属"含 info_processor 的算法，应由信息处理agent执行
  - 注意：如果该处理任务依赖无人机的采集结果，必须把对应的采集子任务填进 prerequisite_tasks
- "coordinator"：由主agent执行，仅适用于最终决策判断与全局信息汇总。主agent不执行具体的数据分析。

注意：
1. **用户说的每一个执行动作都必须生成对应的子任务**。用户说了"侦察"，就必须有侦察子任务；说了"分析数据"，就必须有分析子任务；说了"干扰"，就必须有干扰子任务。
2. 如果任务涉及多个目标位置、多个目标对象或多个执行主体，应拆分为多个子任务。
3. **禁止过度拆分（最重要的规则）**：
   - 允许且必要：采集(uav)与分析(info_processor)拆成两个子任务（执行主体不同，且分析任务依赖采集结果填 prerequisite_tasks）。
   - **绝对禁止**：拆出单独的"飞行/前往/返航"子任务。飞行是侦察/采集动作的一部分，必须并入对应的采集子任务。
   - 反例（错误做法）：T1=无人机飞往目标位置（goal：飞行至目标位置70,15.01）；T2=无人机进行扫频侦察（goal：获取目标的扫频数据）。
   - 正例（正确做法）：T1=无人机飞抵目标位置执行扫频侦察（goal：飞行至目标位置并获取目标的扫频数据）；T2=信息处理agent分析扫频数据。
   - 判断标准：拆分后的每个子任务必须执行主体不同、或面向不同目标、或产出不同信息；只是"先飞到再侦察"不构成拆分理由。
4. 每个子任务的 goal 必须包含该子任务期望获取的**所有信息**，不能遗漏。
5. 只有用户明确说了分析/计算/评估类动作（如"分析""计算""评估"），或子任务需要调用分析类算法产出最终结果时，才生成 info_processor 子任务。
6. 返回前自查一遍：如果发现某个子任务只描述"飞行/前往/返航/抵达"而没有信息产出，把它并入相关的采集子任务后再输出。**输出中不允许存在这种子任务**。

只返回JSON，不要其他内容。"""
    
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 4096},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        
        # 解析JSON
        print(f"[DEBUG] decomposer LLM raw output:\n{content[:2000]}", file=sys.stderr, flush=True)
        data = _extract_json(content)
        if data:
            # 生成 fleet_id
            fleet_id = generate_fleet_id()
            state["fleet_id"] = fleet_id
            
            # 解析 UAV 数量（强制使用 Redis context 中的实际数量）
            state["uav_count"] = uav_count
            
            # 解析子任务（自动补齐缺失的 task_id）
            sub_tasks = []
            existing_ids = {t.get("task_id") for t in data.get("sub_tasks", []) if t.get("task_id")}
            next_id = 1
            while f"T{next_id}" in existing_ids:
                next_id += 1
            for task in data.get("sub_tasks", []):
                tid = task.get("task_id")
                if not tid or str(tid).strip() == "":
                    tid = f"T{next_id}"
                    next_id += 1
                    while f"T{next_id}" in existing_ids:
                        next_id += 1
                sub_tasks.append({
                    "task_id": tid,
                    "task_name": task.get("task_name", ""),
                    "goal": task.get("goal", ""),
                    "executor": task.get("executor", "uav"),
                    "assigned_uav_role": task.get("assigned_uav_role"),
                    "prerequisite_tasks": task.get("prerequisite_tasks", []),
                    "constraints": task.get("constraints", []),
                })
            state["sub_tasks"] = sub_tasks
            
            if not sub_tasks:
                state["output"] = "[Coordinator] 任务分解失败：LLM返回的子任务列表为空"
                return state
            
            # 生成展示文本 - 详细的任务分解
            output = "=" * 60 + "\n"
            output += "【任务分解方案】\n"
            output += "=" * 60 + "\n\n"
            
            for i, task in enumerate(sub_tasks, 1):
                output += f"子任务 {i}: {task.get('task_name', '')}\n"
                output += f"  任务ID: {task.get('task_id', '')}\n"
                output += f"  目标: {task.get('goal', '')}\n"
                prereqs = task.get("prerequisite_tasks", [])
                if prereqs:
                    output += f"  依赖任务: {', '.join(prereqs)}\n"
                constraints = task.get("constraints", [])
                if constraints:
                    output += f"  约束条件: {', '.join(constraints)}\n"
                output += "\n"
            
            output += "=" * 60 + "\n"
            output += "请确认任务分解方案。确认后我将进行任务分配。\n"
            
            state["output"] = output
            state["pending_question"] = "确认任务分解方案？（输入确认继续）"
            print("[DEBUG] decomposer: output + pending_question SET", file=sys.stderr, flush=True)
        else:
            state["output"] = "[Coordinator] 任务分解失败，无法解析LLM输出"
            print("[DEBUG] decomposer: else branch (no pending_question)", file=sys.stderr, flush=True)
            
    except Exception as e:
        state["output"] = f"[Coordinator] 任务分解异常: {e}"
        print(f"[DEBUG] decomposer: EXCEPTION {e}", file=sys.stderr, flush=True)
    
    return state


def task_allocator_node(state: AgentState, deps: Deps) -> AgentState:
    """
    子任务分配节点：
    - coordinator 任务由主 agent 执行
    - uav 任务分配给无人机子 agent
    """
    from fleet.task_allocator import TaskAllocator
    
    allocator = TaskAllocator()
    sub_tasks = state.get("sub_tasks", [])
    
    # 从任务中自动生成 UAV 配置
    # 根据 uav_count 生成指定数量的 UAV，按角色分配
    max_uavs = state.get("uav_count", 1)
    
    # 收集所有需要的角色
    roles_needed = []
    for task in sub_tasks:
        role = task.get("assigned_uav_role")
        if role and task.get("executor", "uav") == "uav":
            roles_needed.append(role)
    
    # 去重保持顺序
    seen = set()
    unique_roles = [r for r in roles_needed if not (r in seen or seen.add(r))]
    
    uav_configs = []
    for i in range(max_uavs):
        role = unique_roles[i % len(unique_roles)] if unique_roles else "scout"
        uav_id = f"UAV_{i + 1}"
        try:
            role_enum = UAVRole(role)
        except ValueError:
            role_enum = UAVRole.SCOUT
        caps = _role_capabilities(role)
        uav_configs.append(UAVConfig(
            uav_id=uav_id, role=role_enum, capabilities=caps
        ))
    
    # 如果没有任何 UAV 任务但有 coordinator 任务，至少创建一架默认 UAV
    if not uav_configs:
        uav_configs.append(UAVConfig(
            uav_id="UAV_1", role=UAVRole.SCOUT, capabilities=["radar", "optical"]
        ))
    
    state["uav_configs"] = [c.to_dict() for c in uav_configs]
    state["uav_count"] = len(uav_configs)
    
    # 只分配 executor="uav" 的任务
    uav_task_objects = []
    for task in sub_tasks:
        if task.get("executor", "uav") == "uav":
            uav_task_objects.append(SubTask(
                task_id=task.get("task_id", ""),
                task_name=task.get("task_name", ""),
                goal=task.get("goal", ""),
                assigned_uav_role=task.get("assigned_uav_role"),
                prerequisite_tasks=task.get("prerequisite_tasks", []),
                constraints=task.get("constraints", []),
            ))
    
    # 执行分配
    assignments = allocator.allocate(uav_task_objects, uav_configs)
    
    # 为 coordinator / info_processor 任务添加标记
    for task in sub_tasks:
        if task.get("executor") == "coordinator":
            assignments.setdefault("coordinator", []).append(task.get("task_id", ""))
        elif task.get("executor") == "info_processor":
            assignments.setdefault("info_processor", []).append(task.get("task_id", ""))

    state["sub_task_assignments"] = assignments

    # 初始化活跃 UAV 列表（不含 coordinator / info_processor）
    state["active_uav_ids"] = [k for k in assignments.keys() if k not in ("coordinator", "info_processor")]
    
    # 生成分配展示文本
    output = "=" * 60 + "\n"
    output += "【任务分配方案】\n\n"
    
    for i, task in enumerate(sub_tasks, 1):
        task_id = task.get("task_id", "")
        task_name = task.get("task_name", "")
        goal = task.get("goal", "")
        executor = task.get("executor", "uav")
        
        if executor == "coordinator":
            assignee = "主Agent（协调者）"
            role_info = ""
        elif executor == "info_processor":
            assignee = "信息处理Agent"
            role_info = ""
        else:
            # 找到分配的 UAV
            assignee = "未分配"
            for uav_id, task_ids in assignments.items():
                if task_id in task_ids and uav_id not in ("coordinator", "info_processor"):
                    assignee = uav_id
                    break
            # 显示角色信息
            role_info = f" ({task.get('assigned_uav_role', '未知角色')})"
        
        output += f"子任务 {i}: {task_name}\n"
        output += f"  目标: {goal}\n"
        output += f"  执行者: {assignee}{role_info}\n\n"
    
    output += "=" * 60 + "\n"
    output += "请确认分配方案。确认后我将创建无人机agent并开始执行。\n"
    
    state["output"] = output
    state["pending_question"] = "确认任务分配？（输入确认后开始执行）"
    
    return state


async def fleet_dispatcher_node(state: AgentState, deps: Deps) -> AgentState:
    """
    Fleet 调度节点：按顺序执行子任务
    """
    # 安全限制：防止无限调度循环
    _dc = state.setdefault("_dispatch_count", 0) + 1
    state["_dispatch_count"] = _dc
    if _dc > 200:
        state["_fleet_all_done"] = True
        state["output"] = "[Fleet] 超过最大调度次数，强制结束"
        return state
    
    # 首次进入时创建子 agent 实例
    if not state.get("_sub_agents_initialized"):
        redis_mgr = get_redis_manager()
        fleet_mgr = FleetManager(deps, redis_mgr=redis_mgr)
        fleet_mgr.initialize_sub_agents(state)
        state["_fleet_manager"] = fleet_mgr
        state["_sub_agents_initialized"] = True
        state["_current_task_idx"] = 0
    
    # 获取 fleet manager
    fleet_mgr = state.get("_fleet_manager")
    if not fleet_mgr:
        redis_mgr = get_redis_manager()
        fleet_mgr = FleetManager(deps, redis_mgr=redis_mgr)
        fleet_mgr.initialize_sub_agents(state)
        state["_fleet_manager"] = fleet_mgr
    
    # 如果子agent在等待用户确认，注入用户输入
    if state.get("_sub_awaiting") and state.get("user_input"):
        task_id = state["_sub_awaiting"]
        agent_id = fleet_mgr.task_to_agent.get(task_id)
        if agent_id:
            if agent_id == FleetManager._COORD_ID:
                fleet_mgr._coordinator_agent.inject_user_input(state["user_input"])
            elif agent_id == FleetManager._INFO_PROCESSOR_ID:
                fleet_mgr._info_processor_agent.inject_user_input(state["user_input"])
            else:
                agent = fleet_mgr.sub_agents.get(agent_id)
                if agent:
                    agent.inject_user_input(state["user_input"])
            state["_sub_awaiting"] = ""
    
    # 推进一步
    await fleet_mgr.tick(state)
    
    # 如果子agent有 pending_question，tick 已设好 output，直接返回
    if state.get("_sub_awaiting"):
        return state
    
    # 更新输出
    current_task_idx = state.get("_current_task_idx", 0)
    sub_tasks = state.get("sub_tasks", [])
    assignments = state.get("sub_task_assignments", {})
    shared_results = state.get("shared_results", {})
    
    output = "=" * 60 + "\n"
    output += "【多机任务执行状态】\n"
    output += "=" * 60 + "\n\n"
    
    for i, task in enumerate(sub_tasks):
        task_id = task.get("task_id", "")
        task_name = task.get("task_name", "")
        goal = task.get("goal", "")
        executor = task.get("executor", "uav")
        
        # 找到执行者
        if executor == "coordinator":
            executor_label = "主Agent"
        elif executor == "info_processor":
            executor_label = "信息处理Agent"
        else:
            executor_label = "未分配"
            for uav_id, task_ids in assignments.items():
                if task_id in task_ids and uav_id not in ("coordinator", "info_processor"):
                    executor_label = uav_id
                    break
        
        # 状态标记
        if task_id in shared_results:
            status = "✅ 已完成"
        elif i == current_task_idx:
            status = "🔄 执行中"
        else:
            status = "⬜ 待执行"
        
        output += f"{status} 子任务 {i+1}: {task_name}\n"
        output += f"  目标: {goal}\n"
        output += f"  执行者: {executor_label}\n\n"
    
    output += "=" * 60 + "\n"
    
    if state.get("_fleet_all_done"):
        output += "\n所有任务执行完成！\n"
    elif current_task_idx < len(sub_tasks):
        current_task = sub_tasks[current_task_idx]
        executor = current_task.get("executor", "uav")
        output += f"\n当前执行: 子任务 {current_task_idx+1} - {current_task.get('task_name', '')}\n"
        output += f"目标: {current_task.get('goal', '')}\n"
        output += "（正在进行：宏观规划→详细规划→执行）\n"
    
    state["output"] = output
    
    return state


def conflict_resolver_node(state: AgentState, deps: Deps) -> AgentState:
    """
    冲突解决节点：
    - 解决跨 UAV 资源冲突
    """
    conflicts = state.get("_active_conflicts", [])
    
    resolution_text = "冲突解决：\n"
    for conflict in conflicts:
        # 简单解决策略
        resolution_text += f"- 已处理冲突: {conflict.get('description', '')}\n"
    
    state["output"] = f"[ConflictResolver] {resolution_text}"
    state["_active_conflicts"] = []
    
    return state


def result_aggregator_node(state: AgentState, deps: Deps) -> AgentState:
    """
    结果聚合节点：
    - 收集所有子 agent 的执行结果
    - 合并为统一的任务报告
    """
    fleet_mgr = state.get("_fleet_manager")
    
    if fleet_mgr:
        results = fleet_mgr.get_aggregated_results(state)
        state["output"] = "[ResultAggregator] 舰队任务执行完成"
    else:
        state["output"] = "[ResultAggregator] 无法获取Fleet管理器"
    
    # 重置 fleet 状态（清除所有残留状态，防止死循环）
    state["_fleet_all_done"] = False
    state["_sub_agents_initialized"] = False
    state["_fleet_tick_count"] = 0
    state["_dispatch_count"] = 0
    state["user_input"] = ""
    state["_sub_awaiting"] = ""
    state["_coordinator_intent"] = ""
    state["pending_question"] = ""
    
    return state


def _repair_json(raw: str) -> str:
    """修复 LLM 生成的常见 JSON 格式错误，提高解析成功率"""
    # 1. 修复双冒号 "key":: → "key": null,
    raw = re.sub(r'":\s*:', '": null,', raw)
    # 2. 修复缺失值的键 "key":\n"next_key" → "key": null,\n"next_key"
    raw = re.sub(r'":\s*\n\s*"', '": null,\n"', raw)
    # 3. 修复 "key": } → "key": null }
    raw = re.sub(r'":\s*}', '": null }', raw)
    # 4. 修复 "key": , → "key": null,
    raw = re.sub(r'":\s*,', '": null,', raw)
    # 5. 去掉尾随逗号
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)
    return raw


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON 对象（带错误修复）"""
    # 尝试 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        blob = _repair_json(m.group(1))
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
    
    # 尝试裸 JSON：找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        blob = _repair_json(text[start:end + 1])
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
    
    # 终极尝试：逐行修复后重试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json(text))
        except json.JSONDecodeError:
            pass
    
    return None


def _format_fleet_plan(data: dict) -> str:
    """格式化舰队计划为可读文本"""
    lines = []
    
    lines.append(f"场景: {data.get('scenario', '未知')}")
    lines.append(f"UAV数量: {data.get('uav_count', 0)}")
    lines.append("")
    
    # UAV 配置
    lines.append("UAV 配置:")
    for cfg in data.get("uav_configs", []):
        lines.append(f"  - {cfg.get('uav_id', '')}: 角色={cfg.get('role', '')}, 能力={cfg.get('capabilities', [])}")
    lines.append("")
    
    # 阶段和子任务
    lines.append("任务阶段:")
    for phase in data.get("fleet_phases", []):
        lines.append(f"  [{phase.get('phase_id', '')}] {phase.get('phase_name', '')}")
        lines.append(f"    目标: {phase.get('goal', '')}")
        lines.append(f"    执行模式: {phase.get('execution_mode', 'parallel')}")
        
        for task in phase.get("sub_tasks", []):
            lines.append(f"    - {task.get('task_id', '')}: {task.get('task_name', '')}")
            lines.append(f"      目标: {task.get('goal', '')}")
            if task.get("prerequisite_tasks"):
                lines.append(f"      依赖: {task.get('prerequisite_tasks', [])}")
    
    return "\n".join(lines)
