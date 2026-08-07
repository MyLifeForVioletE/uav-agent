"""
Fleet 生命周期管理器：
管理多个 SubAgent 的创建、执行、状态收集、结果聚合
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict
from core.state import AgentState, Deps
from core.redis_manager import RedisManager
from core.config import BASE_DIR, COORDINATOR_ID
from core.kafka_bus import get_kafka_bus
from agent.sub_agent import SubAgent


def _load_algorithm_list_text() -> str:
    """从 algorithms.json 加载全部算法（含输入/输出 schema），供指挥缺参推导时参考"""
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
        params = ", ".join(f"{k}({v.get('type', '')}: {v.get('description', '')})" for k, v in cap.get("input_schema", {}).items())
        outputs = ", ".join(f"{k}({v.get('description', k)})" if isinstance(v, dict) else str(v) for k, v in cap.get("output_schema", {}).items())
        lines.append(
            f"- {name}：{desc}\n"
            f"  输入: {params or '无'}\n"
            f"  输出: {outputs or '无'}"
        )
    return "\n".join(lines)


def _is_stub_placeholder_output(output: str) -> bool:
    """判断是否为非 exe 算法 stub 的占位输出（格式: 输出;字段名列表）。

    stub 工具不调用 exe，stdout 仅为 "输出;tFreq, tBand" 之类的字段名列表，
    不含真实数据。若把它交给指挥 LLM 提取，LLM 常把字段名当成值写入 Redis，
    污染算法的空占位（如 sweepData 从 "" 变成 "sweepData"）。此类输出应跳过提取。
    """
    if not output:
        return False
    import re
    # stub 输出为逗号/空格分隔的字段名标识符列表，不含数字、坐标等真实数据
    return bool(re.fullmatch(r"输出;[\s,，、]*(?:[A-Za-z_][A-Za-z0-9_]*[\s,，、]*)*", str(output).strip()))


class FleetManager:
    """Fleet 管理器：协调多个子agent的生命周期"""
    
    _COORD_ID = "_coordinator_"
    _INFO_PROCESSOR_ID = "_info_processor_"

    def __init__(self, deps: Deps, redis_mgr: RedisManager = None):
        self.deps = deps
        self.redis_mgr = redis_mgr
        self.sub_agents: Dict[str, SubAgent] = {}   # uav_id → SubAgent
        self.task_to_agent: Dict[str, str] = {}       # task_id → agent_id
        self._running_tasks: set = set()              # 正在执行的 task_id 集合
        self._completed_tasks: set = set()            # 已完成的 task_id 集合（执行完成）
        self._planned_tasks: set = set()              # 已规划但未执行的任务
        self._saved_plans: dict = {}                  # task_id → {detail_actions}
        self._phase: str = "planning"                 # "planning" | "execution"
        self._info_processor_agent: SubAgent | None = None  # 信息处理类子任务的 agent
        self._bus = get_kafka_bus()
    
    def initialize_sub_agents(self, state: AgentState):
        """根据 uav_configs 创建子 agent 实例 + 创建信息处理 agent"""
        configs = state.get("uav_configs", [])
        assignments = state.get("sub_task_assignments", {})
        session_id = state.get("session_id", "default")
        
        for config in configs:
            uav_id = config["uav_id"]
            agent = SubAgent(agent_id=uav_id, deps=self.deps, initial_config=config, mode="uav", redis_mgr=self.redis_mgr, session_id=session_id)
            self.sub_agents[uav_id] = agent
            
            # 分配子任务
            task_ids = assignments.get(uav_id, [])
            agent._assigned_task_ids = task_ids
            
            # 建立 task_id -> agent_id 映射
            for tid in task_ids:
                self.task_to_agent[tid] = uav_id

        # 创建信息处理 agent（基于无人机采集数据的分析处理类子任务）
        self._info_processor_agent = SubAgent(
            agent_id=self._INFO_PROCESSOR_ID, deps=self.deps,
            initial_config={}, mode="info_processor", redis_mgr=self.redis_mgr, session_id=session_id
        )
        ip_task_ids = assignments.get("info_processor", [])
        self._info_processor_agent._assigned_task_ids = ip_task_ids
        for tid in ip_task_ids:
            self.task_to_agent[tid] = self._INFO_PROCESSOR_ID

    # ── 指挥 agent 收件箱：处理子 agent 的上报/请示/结果 ─────────────

    async def _process_coordinator_inbox(self, state: AgentState):
        """拉取指挥收件箱：report/result 程序化归档；request 交给指挥 LLM 决策回复"""
        if not self._bus:
            return
        try:
            msgs = await self._bus.poll(self._COORD_ID, timeout=0.1)
        except Exception as e:
            sys.stderr.write(f"[Fleet] 指挥收件箱拉取失败: {e}\n")
            sys.stderr.flush()
            return
        if not msgs:
            return

        session_id = state.get("session_id", "default")
        mailbox = state.setdefault("coordinator_mailbox", [])
        for m in msgs:
            mtype = m.get("msg_type", "")
            from_id = m.get("from", "")
            content = m.get("content", "")
            # 只处理本会话的消息；非本会话的视为残留，直接消费丢弃
            if m.get("payload", {}).get("session_id") != session_id:
                sys.stderr.write(f"[Fleet] 丢弃非本会话消息({mtype} from {from_id}): {content[:60]}\n")
                sys.stderr.flush()
                continue
            mailbox.append(m)
            if mtype == "result":
                sys.stderr.write(f"[Fleet] 指挥收到 {from_id} 结果: {content[:120]}\n")
                sys.stderr.flush()
                await self._handle_agent_result(state, m)
            elif mtype in ("report",):
                await self._archive_report(state, m)
                await self._record_plan_report(state, m)
                sys.stderr.write(f"[Fleet] 指挥收到 {from_id} 上报({mtype}): {content[:120]}\n")
                sys.stderr.flush()
            elif mtype == "request":
                sys.stderr.write(f"[Fleet] 指挥收到 {from_id} 请示: {content[:120]}\n")
                sys.stderr.flush()
                await self._handle_coordinator_request(state, m)

    async def _archive_report(self, state: AgentState, msg: dict):
        """子 agent 上报/结果归档：结果写入 shared_results，payload 中的上下文更新写入 Redis"""
        from_id = msg.get("from", "")
        payload = msg.get("payload") or {}

        # 上下文更新落地 Redis
        context_updates = payload.get("context_updates")
        if context_updates and self.redis_mgr:
            try:
                self.redis_mgr.update_context(state.get("session_id", "default"), context_updates)
            except Exception as e:
                sys.stderr.write(f"[Fleet] 上报上下文更新失败: {e}\n")
                sys.stderr.flush()

        # 归档到 shared_results（消息引用了任务/产出结果）
        result = payload.get("result")
        task_id = payload.get("task_id", "")
        key = task_id if task_id else from_id
        if result is not None or msg.get("content"):
            state.setdefault("shared_results", {})[key] = {
                "status": "done",
                "output": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                "source": from_id,
            }
        # 分析结果写入 Redis 上下文，供后续任务使用
        if result is not None and self.redis_mgr and state.get("session_id"):
            try:
                self.redis_mgr.update_context(state["session_id"], {"targets": [{"analysis_result": result}]})
            except Exception:
                pass

    async def _record_plan_report(self, state: AgentState, msg: dict):
        """指挥记录 UAV 上报的宏观/详细规划 + 当前执行动作（落 Redis + state）"""
        payload = msg.get("payload") or {}
        report_type = payload.get("report_type", "")
        from_id = msg.get("from", "")
        session_id = state.get("session_id", "default")

        state.setdefault("recorded_plans", {}).setdefault(from_id, {})
        entry = state["recorded_plans"][from_id]

        if report_type == "macro_plan":
            entry["macro_plan"] = payload.get("macro_phases", [])
        elif report_type == "detail_plan":
            entry["detail_plan"] = payload.get("detail_actions", [])
        elif report_type in ("action_start", "action_executed"):
            log = entry.setdefault("execution_log", [])
            log.append({
                "step_idx": payload.get("step_idx", 0),
                "total": payload.get("total", 0),
                "action": payload.get("action", {}),
                "success": payload.get("success"),
            })
            if report_type == "action_executed":
                self._write_action_script_line(state, msg)

        if self.redis_mgr:
            try:
                self.redis_mgr.set(f"plans:{session_id}", state.get("recorded_plans", {}), expire=86400)
            except Exception as e:
                sys.stderr.write(f"[Fleet] 规划记录落 Redis 失败: {e}\n")
                sys.stderr.flush()

    def _script_path(self, session_id: str) -> str:
        """统一任务脚本路径：整个任务写入同一个文件"""
        out_dir = BASE_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"actions_script_{session_id}.txt"

    def _finalize_script_segment(self, state: AgentState, reason: str = ""):
        """在统一脚本中写入暂停标记（整个任务仍写入同一个文件）"""
        session_id = state.get("session_id", "default")
        try:
            script_path = self._script_path(session_id)
            with open(script_path, "a", encoding="utf-8") as f:
                f.write(f"# === 暂停 === {reason}\n" if reason else "# === 暂停 ===\n")
            sys.stderr.write(f"[Fleet] 脚本暂停标记: {script_path} :: {reason}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[Fleet] 脚本暂停标记写入失败: {e}\n")
            sys.stderr.flush()

    def _write_action_script_line(self, state: AgentState, msg: dict):
        """把已执行的动作 + 实际调用参数值写入脚本文件（txt，每行一个动作）"""
        payload = msg.get("payload") or {}
        from_id = msg.get("from", "")
        session_id = state.get("session_id", "default")
        action = payload.get("action", {}) or {}
        step_idx = payload.get("step_idx", 0)
        total = payload.get("total", 0)

        action_name = action.get("action_name", "")
        tool_name = action.get("tool_name", action.get("action_name", ""))
        tool_inputs = action.get("tool_inputs", {}) or {}
        params = ", ".join(f"{k}={v}" for k, v in tool_inputs.items() if str(v).strip())
        output = payload.get("output", "") or ""

        agent_label = "分析Agent" if from_id == self._INFO_PROCESSOR_ID else "无人机Agent"
        # total=0 表示分析 agent（总步骤不定，只显示连续步骤号）；否则显示 N/total
        step_info = f"步骤 {step_idx+1}" if total <= 0 else f"步骤 {step_idx+1}/{total}"
        line = f"[{from_id} {agent_label} {step_info}] {action_name or (tool_name if tool_name else '未知动作')}"
        if tool_name and params:
            line += f" | {tool_name}({params})"
        elif tool_name:
            line += f" | {tool_name}"
        if output:
            out_summary = output.strip().replace("\n", " ")
            if len(out_summary) > 300:
                out_summary = out_summary[:300] + "...(截断)"
            line += f"  => {out_summary}"

        try:
            script_path = self._script_path(session_id)
            with open(script_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            sys.stderr.write(f"[Fleet] 已写入脚本文件: {script_path} :: {line}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[Fleet] 脚本文件写入失败: {e}\n")
            sys.stderr.flush()

    async def _handle_coordinator_request(self, state: AgentState, msg: dict):
        """子 agent 请示：缺参请求走专项处理；其余交指挥 LLM 决策"""
        payload = msg.get("payload") or {}
        if payload.get("request_type") == "missing_param":
            await self._handle_missing_param_request(state, msg)
            return
        await self._handle_generic_request(state, msg)

    async def _handle_missing_param_request(self, state: AgentState, msg: dict):
        """缺参请示：指挥先从算法库判断能否用现有信息推导，不能则询问用户"""
        payload = msg.get("payload") or {}
        from_id = msg.get("from", "")
        session_id = state.get("session_id", "default")
        missing = payload.get("missing_params", [])
        tool_name = payload.get("tool_name", "")
        goal = payload.get("goal", "")
        task_id = payload.get("task_id", "")
        correlation_id = msg.get("correlation_id")

        if not missing:
            await self._send_reply(state, from_id, "缺参信息为空，请重试。", correlation_id)
            return

        # 防循环：同一 agent+动作 的推导尝试上限
        attempts_key = f"{from_id}:{tool_name}:{payload.get('action_name', '')}"
        attempts = state.setdefault("_param_resolve_attempts", {})
        count = attempts.get(attempts_key, 0)
        attempts[attempts_key] = count + 1
        force_ask_user = count >= 3

        llm = self.deps.llm_no_tools if self.deps else None
        ctx = {}
        if self.redis_mgr:
            try:
                ctx = self.redis_mgr.get_context(session_id) or {}
            except Exception:
                ctx = {}

        if llm and not force_ask_user:
            decision = await self._derive_param_via_algorithm(state, msg, missing, ctx)
            algo_name = decision.get("algorithm")
            if decision.get("derivable") and algo_name:
                action = {
                    "executor": "tool",
                    "tool_name": algo_name,
                    "tool_inputs": decision.get("tool_inputs", {}) or {},
                    "action_name": decision.get("action_name") or f"调用{algo_name}计算缺失参数",
                    "goal": decision.get("goal") or f"为 {tool_name} 计算缺失参数",
                }
                # 指挥记录：插入的算法动作
                state.setdefault("recorded_plans", {}).setdefault(from_id, {}).setdefault("inserted_actions", []).append(action)
                if self.redis_mgr:
                    try:
                        self.redis_mgr.set(f"plans:{session_id}", state.get("recorded_plans", {}), expire=86400)
                    except Exception:
                        pass
                sys.stderr.write(f"[Fleet] 缺参推导成功: {tool_name} 缺 {[p.get('name') for p in missing]} → 用 {algo_name}\n")
                sys.stderr.flush()
                await self._send_reply(state, from_id,
                    f"指挥已推导：调用算法 {algo_name} 计算缺失参数，动作已加入执行列表。",
                    correlation_id,
                    payload={"kind": "algorithm", "action": action})
                return

        # 推导失败/无 LLM/超限 → 询问用户：先结束当前脚本段，恢复后另起新段
        param_desc = "、".join(f"{p.get('name', '')}({p.get('description', '')})" for p in missing)
        self._finalize_script_segment(state, f"缺参暂停：{tool_name} 需要 {param_desc}，等待用户提供")
        state["_awaiting_user_param"] = {
            "agent_id": from_id,
            "task_id": task_id,
            "tool_name": tool_name,
            "params": missing,
            "correlation_id": correlation_id,
        }
        state["_sub_awaiting"] = task_id
        question = (f"无人机 {from_id} 执行动作「{payload.get('action_name', '')}」需要参数 {param_desc}，"
                    f"指挥无法从现有信息推导。\n请提供该参数的值（如坐标格式 70,15.01）。")
        state["pending_question"] = question
        state["output"] = question
        sys.stderr.write(f"[Fleet] 缺参推导失败/超限，询问用户: {question[:120]}\n")
        sys.stderr.flush()

    async def _derive_param_via_algorithm(self, state: AgentState, msg: dict, missing: list, ctx: dict) -> dict:
        """指挥 LLM：判断能否用现有信息调用某算法得到缺失参数"""
        llm = self.deps.llm_no_tools if self.deps else None
        if not llm:
            return {"derivable": False}
        payload = msg.get("payload") or {}
        from_id = msg.get("from", "")
        tool_name = payload.get("tool_name", "")
        goal = payload.get("goal", "")
        action = payload.get("action", {})
        known_inputs = action.get("tool_inputs", {}) if isinstance(action, dict) else {}

        algo_text = _load_algorithm_list_text()
        missing_text = json.dumps(missing, ensure_ascii=False)
        known_text = json.dumps(known_inputs, ensure_ascii=False)
        ctx_text = json.dumps(ctx, ensure_ascii=False)

        prompt = (
            "你是多无人机电磁侦察任务的指挥 agent。无人机 agent 调用算法工具时缺少参数，"
            "请你判断能否用【当前已掌握的现有信息】调用【算法库中的某个算法】直接算出缺失参数。\n\n"
            f"【缺参者】{from_id}\n"
            f"【目标工具】{tool_name}\n"
            f"【动作目标】{goal}\n"
            f"【缺失参数】{missing_text}\n"
            f"【该动作已明确的参数】{known_text}\n\n"
            f"【算法库】\n{algo_text}\n\n"
            f"【当前共享上下文（已有信息）】\n{ctx_text}\n\n"
            "【判断规则】\n"
            "1. 只有当选中的算法能产出缺失参数（语义匹配，如路径规划的起点/终点可产出坐标类参数），"
            "且该算法每个输入参数都能从现有上下文或已知参数中得到明确值时，derivable 才为 true。\n"
            "2. 缺失参数可能是选中的算法的输入参数，也可能是其输出。选择能算出该参数的最短链路。\n"
            "3. 禁止编造数值。任何参数拿不到明确值就 derivable=false。\n"
            "4. 若缺失参数是 file 类型（如参数文件），可选用生成该文件的算法；无法生成则 derivable=false。\n\n"
            "只输出 JSON：\n"
            '{"derivable": true/false, "algorithm": "算法名或空", "tool_inputs": {算法输入参数: 值}, '
            '"action_name": "人类可读动作名", "goal": "该动作要算出什么参数", "reason": "简短理由"}'
        )
        try:
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = str(response.content).strip()
        except Exception as e:
            sys.stderr.write(f"[Fleet] 缺参推导调用失败: {e}\n")
            sys.stderr.flush()
            return {"derivable": False}
        import re as _re
        try:
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if not m:
                return {"derivable": False}
            decision = json.loads(m.group())
            if not isinstance(decision, dict):
                return {"derivable": False}
            return decision
        except Exception as e:
            sys.stderr.write(f"[Fleet] 缺参推导解析失败: {raw[:200]}\n")
            sys.stderr.flush()
            return {"derivable": False}

    async def _handle_agent_result(self, state: AgentState, msg: dict):
        """子 agent 算法/分析结果：指挥 LLM 从原始输出提取字段，更新 Redis 上下文 + 归档 shared_results"""
        payload = msg.get("payload") or {}
        output = payload.get("algorithm_output", "") or msg.get("content", "")
        if not output:
            return
        tool_name = payload.get("tool_name", "")
        goal = payload.get("goal", "")
        source = payload.get("source", "uav")
        from_id = msg.get("from", "")

        # 归档 shared_results
        key = payload.get("task_id") or from_id
        state.setdefault("shared_results", {})[key] = {
            "status": "done",
            "output": str(output)[:500],
            "source": from_id,
        }

        # 非 exe 算法 stub 占位输出（"输出;字段名列表"）不含真实数据，
        # 跳过指挥 LLM 提取，避免把字段名写成值污染空占位。
        if _is_stub_placeholder_output(output):
            sys.stderr.write(f"[Fleet] {tool_name or from_id} stub 占位输出，跳过上下文提取: {str(output)[:80]}\n")
            sys.stderr.flush()
            return

        # 指挥 LLM 提取 → 写 Redis
        if self.redis_mgr:
            try:
                await self._update_context_from_agent_result(
                    state.get("session_id", "default"), from_id, tool_name, goal, output,
                    task_id=payload.get("task_id", ""), task_name=payload.get("task_name", ""),
                )
            except Exception as e:
                sys.stderr.write(f"[Fleet] 结果上下文更新失败: {e}\n")
                sys.stderr.flush()

    async def _update_context_from_agent_result(self, session_id: str, agent_id: str,
                                                tool_name: str, goal: str, output: str,
                                                task_id: str = "", task_name: str = ""):
        """指挥侧：LLM 从算法/分析输出中提取关键字段并写入 Redis 上下文（原 UAV 直写逻辑搬移）"""
        llm = self.deps.llm_no_tools if self.deps else None
        if not llm:
            return
        current_context = self.redis_mgr.get_context(session_id) or {}

        # 工具输出定义，帮助 LLM 理解返回字段含义
        tool_params = ""
        algo = {}
        try:
            with open(BASE_DIR / "algorithms.json", encoding="utf-8") as f:
                algos = {a["name"]: a for a in json.load(f).get("capabilities", [])}
            algo = algos.get(tool_name, {})
            schema = algo.get("output_schema", {})
            if schema:
                lines = ["【工具输出定义】"]
                for pname, pinfo in schema.items():
                    lines.append(f"- {pname}: {pinfo.get('description', pname)}")
                tool_params = "\n".join(lines) + "\n\n"
        except Exception:
            pass

        # 若算法把结果写入输出文件（如 pathPlanning 的 "Output file: path_xxx.txt"），读取其内容一并交给 LLM
        import re as _re
        output_enriched = str(output)[:3000]
        try:
            exe_path = algo.get("executable", "")
            work_dir = algo.get("working_dir", "") or (str(Path(exe_path).parent) if exe_path else "")
            m = _re.search(r"Output file:\s*([^\s\r\n]+)", output_enriched)
            if m and work_dir:
                out_file = Path(work_dir) / m.group(1)
                if out_file.is_file():
                    file_content = out_file.read_text(encoding="utf-8", errors="replace")
                    output_enriched += f"\n\n【输出文件 {m.group(1)} 内容】\n{file_content[:3000]}"
        except Exception:
            pass

        prompt_parts = [
            "你是一个任务执行系统的上下文更新模块。\n",
            "算法工具执行完成后，从工具返回的结果中提取关键数据，更新 Redis 上下文，"
            "以便后续子任务（如数据分析）可以直接使用。\n\n",
            f"【当前 agent】{agent_id}\n\n",
            f"【所属任务】{task_id} - {task_name}\n\n",
            f"【动作目标】{goal}\n\n",
            f"【调用的算法】{tool_name}\n\n",
            f"【算法返回结果】\n{output_enriched}\n\n",
            tool_params,
            f"【当前 Redis 上下文】{json.dumps(current_context, ensure_ascii=False)}\n\n",
            "【上下文字段规范（必须遵守）】\n"
            "1. 模板中 uavs/targets/base 的位置字段只有 Longitude 和 Latitude，不存在 position/x/y 字段。\n"
            "2. 更新位置时必须写入 Longitude 和 Latitude 两个字段，例如 uavs.0.Longitude、uavs.0.Latitude。\n"
            "3. 坐标值若是 \"70,15.01\" 这类用逗号分隔的字符串，必须拆成两条分别写入 Longitude（70）和 Latitude（15.01）。\n"
            "4. 其余数据（如路径 path、扫频数据 sweep_data、分析结论 analysis_result）可按语义新增字段，但不得新增位置类字段。\n\n"
            "【结果归属判断（必须遵守）】\n"
            "1. 本结果来自任务 {task_id}（{task_name}），必须把数据写入该任务对应的 target 条目。\n"
            "2. 若 task_id 与当前上下文中某个 target 的 id 一致（如 task_id=T1、target.id=T1），优先写入该 target。\n"
            "3. 若 task_id 匹配不上，根据 goal 与结果内容判断应归属的 target；确实无法确定时写入 targets 的第一个元素。\n\n"
            "分析思路：\n"
            "1. 从算法返回结果中提取有意义的数据（如扫频数据、目标参数、位置、路径等）\n"
            "2. 决定这些数据应该存储到上下文的哪个字段（如 targets.T1.sweep_data、uavs.0.Longitude/Latitude、uavs.0.path）\n"
            "3. 如果结果中没有可提取的数据，返回空对象 {}\n\n"
            "请输出一个 JSON 对象表示要更新到 Redis 的字段映射。\n",
            'key 为点号分隔的路径（如 "targets.T1.sweep_data" 或 "uavs.UAV_1.Latitude"），\n',
            "数组既可用数字索引也可用 id 值来定位元素。\n",
            "value 为具体的数值或字符串。",
        ]
        prompt = "".join(prompt_parts)

        from langchain_core.messages import HumanMessage
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = str(response.content).strip()
        except Exception as e:
            sys.stderr.write(f"[Fleet] 指挥结果提取调用失败: {e}\n")
            sys.stderr.flush()
            return

        try:
            updates = json.loads(content)
        except json.JSONDecodeError:
            m = _re.search(r'\{.*\}', content, _re.DOTALL)
            updates = json.loads(m.group()) if m else None

        if not updates or not isinstance(updates, dict):
            return

        nested = {}
        for key_path, val in updates.items():
            parts = key_path.split(".")
            # 过滤 path：航线已由 tool_executor（_write_path_to_redis）结构化写入 uavs[].path，
            # 不允许 LLM 重复写入（会污染 targets.T1.path，或用扁平数组覆盖结构）。
            if "path" in {p.strip() for p in parts}:
                sys.stderr.write(f"[Fleet] 忽略 LLM 写入的 path 字段: {key_path}（由 tool_executor 负责）\n")
                sys.stderr.flush()
                continue
            d = nested
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val

        self.redis_mgr.update_context(session_id, nested)
        sys.stderr.write(f"[Fleet] 指挥更新上下文: {json.dumps(nested, ensure_ascii=False)}\n")
        sys.stderr.flush()

    async def _handle_generic_request(self, state: AgentState, msg: dict):
        """子 agent 请示：交指挥 LLM 决策，生成 reply（可附带向其它 agent 下发的指令）"""
        llm = self.deps.llm_no_tools if self.deps else None
        if not llm:
            sys.stderr.write("[Fleet] 指挥 LLM 不可用，无法决策\n")
            sys.stderr.flush()
            return

        ctx = {}
        if self.redis_mgr:
            try:
                ctx = self.redis_mgr.get_context(state.get("session_id", "default")) or {}
            except Exception:
                ctx = {}

        prompt = (
            "你是多无人机电磁侦察任务的指挥 agent。子 agent 向你请示，请做出决策并回复。\n\n"
            f"【请示者】{msg.get('from', '')}\n"
            f"【请示内容】{msg.get('content', '')}\n"
            f"【当前各 agent 状态】{json.dumps(state.get('uav_states', {}), ensure_ascii=False)}\n"
            f"【已完成任务结果】{json.dumps(state.get('shared_results', {}), ensure_ascii=False)}\n"
            f"【当前共享上下文】{json.dumps(ctx, ensure_ascii=False)}\n\n"
            "请以 JSON 输出决策：\n"
            '{"reply": "对请示者的直接回复（中文，给出具体决策/数据，禁止编造不存在的坐标或参数）", '
            '"instructions": [{"to": "目标agent id", "content": "指令内容"}]}\n'
            "如果无需向其它 agent 下发指令，instructions 为空数组。"
        )
        try:
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = str(response.content).strip()
        except Exception as e:
            sys.stderr.write(f"[Fleet] 指挥决策调用失败: {e}\n")
            sys.stderr.flush()
            raw = ""

        decision = self._parse_decision(raw)
        reply_text = decision.get("reply") or raw or "已收到你的请示。"

        # 回发回复（带 correlation_id 完成 request/reply 配对）
        if self._bus:
            session_id = state.get("session_id", "default")
            await self._bus.send(
                self._COORD_ID, msg.get("from", ""), "reply",
                reply_text, correlation_id=msg.get("correlation_id"),
                payload={"session_id": session_id},
            )
            # 指挥下发的指令
            for inst in decision.get("instructions") or []:
                to_id = inst.get("to")
                if to_id:
                    await self._bus.send(self._COORD_ID, to_id, "instruction", inst.get("content", ""),
                                         payload={"session_id": session_id})
        sys.stderr.write(f"[Fleet] 指挥回复 {msg.get('from','')}: {reply_text[:120]}\n")
        sys.stderr.flush()

    def _parse_decision(self, raw: str) -> dict:
        """解析指挥决策 JSON；失败则回退为纯文本 reply"""
        import re
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {"reply": raw, "instructions": []}

    async def _send_reply(self, state: AgentState, to_agent_id: str, content: str,
                          correlation_id: str = None, payload: dict = None):
        """指挥向子 agent 发送 reply（带 correlation_id 配对 request/reply）+ 可选结构化 payload"""
        if not self._bus:
            return
        session_id = state.get("session_id", "default")
        reply_payload = dict(payload or {})
        reply_payload.setdefault("session_id", session_id)
        await self._bus.send(self._COORD_ID, to_agent_id, "reply", content,
                             correlation_id=correlation_id, payload=reply_payload)
        sys.stderr.write(f"[Fleet] 指挥回复 {to_agent_id}: {content[:120]}\n")
        sys.stderr.flush()

    async def _handle_user_param_answer(self, state: AgentState, user_input: str):
        """用户回答了指挥的缺参问题：LLM 提取参数值 → 回发 reply 给无人机 → 清理等待状态"""
        info = state.get("_awaiting_user_param") or {}
        agent_id = info.get("agent_id", "")
        correlation_id = info.get("correlation_id")
        params = info.get("params", [])
        llm = self.deps.llm_no_tools if self.deps else None
        values = {}

        if llm and params and user_input.strip():
            missing_text = json.dumps(params, ensure_ascii=False)
            prompt = (
                "你是多无人机电磁侦察任务的指挥 agent。用户回答了你关于缺失参数的问题，"
                "请从用户的回答中提取各参数的值。\n\n"
                f"【缺失参数】{missing_text}\n"
                f"【用户回答】{user_input}\n\n"
                "【关键规则】\n"
                "1. 数值型参数（如频率、带宽、功率）必须**保留原始值及其单位**，禁止剥离单位、禁止丢失单位。"
                "例如用户回答 \"2GHz\" 应提取为 \"2GHz\"（值与单位一起保留），而不是 2。\n"
                "2. 单位为 GHz/MHz/kHz/Hz/dBm/W 等时，作为字符串保留（如 \"2GHz\" 或 \"3MHz\"）。\n"
                "3. 坐标参数仍用 x,y 格式。\n\n"
                "只输出 JSON 对象，key 为参数名，value 为保留单位的值字符串。"
                "若某个参数无法从回答中得到明确值，则不输出该 key。"
                "若全部无法提取，输出 {}。"
            )
            try:
                from langchain_core.messages import HumanMessage
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                raw = str(response.content).strip()
                import re as _re
                m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    if isinstance(parsed, dict):
                        values = parsed
            except Exception as e:
                sys.stderr.write(f"[Fleet] 参数提取失败: {e}\n")
                sys.stderr.flush()

        # 兜底：单个缺参且用户输入可能即值
        if not values and len(params) == 1 and user_input.strip():
            values[params[0].get("name")] = user_input.strip()

        sys.stderr.write(f"[Fleet] 用户提供的参数值: {values}\n")
        sys.stderr.flush()

        # 指挥将用户提供的参数同步到 Redis 上下文（黑板共享，供后续任务使用）
        if values and self.redis_mgr:
            try:
                await self._update_context_from_user_params(state, values, info)
            except Exception as e:
                sys.stderr.write(f"[Fleet] 用户参数写 Redis 失败: {e}\n")
                sys.stderr.flush()

        if self._bus:
            await self._send_reply(
                state, agent_id,
                f"指挥已收到你所需参数：{json.dumps(values, ensure_ascii=False)}" if values else f"参数仍未确定，请重试：{user_input}",
                correlation_id,
                payload={"kind": "param_value", "values": values},
            )

        state["_awaiting_user_param"] = {}
        state["_sub_awaiting"] = ""
        state["user_input"] = ""

    async def _update_context_from_user_params(self, state: AgentState, values: dict, info: dict):
        """指挥把用户提供的缺失参数值映射到 Redis 上下文（黑板共享）"""
        llm = self.deps.llm_no_tools if self.deps else None
        if not llm:
            return
        session_id = state.get("session_id", "default")
        current_context = self.redis_mgr.get_context(session_id) or {}

        # 获取目标工具的输入/输出定义，帮助 LLM 语义映射
        tool_params = ""
        try:
            with open(BASE_DIR / "algorithms.json", encoding="utf-8") as f:
                algos = {a["name"]: a for a in json.load(f).get("capabilities", [])}
            schema = algos.get(info.get("tool_name", ""), {}).get("output_schema", {})
            if schema:
                lines = ["【目标工具输出定义】"]
                for pname, pinfo in schema.items():
                    lines.append(f"- {pname}: {pinfo.get('description', pname)}")
                tool_params = "\n".join(lines) + "\n\n"
        except Exception:
            pass

        prompt_parts = [
            "你是一个多无人机任务系统的上下文更新模块。用户为缺失参数补充了值，"
            "请把这些参数值映射到 Redis 上下文中合适的字段，以便后续子任务（如数据分析）直接使用。\n\n",
            f"【无人机】{info.get('agent_id', '')}\n",
            f"【所属任务】{info.get('task_id', '')}\n",
            f"【目标工具】{info.get('tool_name', '')}\n\n",
            f"【用户提供的参数值】\n{json.dumps(values, ensure_ascii=False)}\n\n",
            tool_params,
            f"【当前 Redis 上下文】\n{json.dumps(current_context, ensure_ascii=False)}\n\n",
            "【上下文字段规范（必须遵守）】\n"
            "1. 模板中 uavs/targets/base 的位置字段只有 Longitude 和 Latitude，坐标为 x,y 格式需拆成两条。\n"
            "2. 坐标值 \"x,y\" 拆成 uavs.N.Longitude 与 uavs.N.Latitude（或 targets 对应条目）。\n"
            "3. 目标参数（如频率、带宽、信号强度）写入对应 target 条目（用 id 定位，如 targets.T1.Frequency）。\n"
            "4. 无人机自身属性写 uavs 条目（如 uavs.UAV_1.Frequency）。\n"
"5. 确实无处安放时，可新增通用字段但没有更好的位置则返回 {}\n\n"
            "【值单位规则】\n"
            "1. **保留用户提供的值和单位**，禁止剥离单位、禁止换算。如 \"2GHz\" 就写 \"2GHz\"。\n"
            "2. 坐标字符串（x,y）拆成 Longitude 与 Latitude 两条（唯一需要拆分的情况）。\n\n"
            "【结果归属判断】\n"
            "1. 优先写入 task_id 对应的 target（如 task_id=T1 与 target.id=T1）。\n"
            "2. 无法确定目标时，根据参数名语义放入 target 第一项。\n\n"
            "请输出 JSON 对象表示要更新到 Redis 的字段映射。\n",
            'key 为点号分隔路径（如 "targets.T1.Frequency"），value 为带单位的参数值。'
            "value 若是不带单位的坐标字符串，拆成经度/纬度两条。"
        ]
        prompt = "".join(prompt_parts)

        try:
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = str(response.content).strip()
        except Exception as e:
            sys.stderr.write(f"[Fleet] 用户参数映射调用失败: {e}\n")
            sys.stderr.flush()
            return

        import re as _re
        try:
            m = _re.search(r"\{.*\}", content, _re.DOTALL)
            updates = json.loads(m.group()) if m else None
        except Exception:
            updates = None
        if not updates or not isinstance(updates, dict):
            return

        nested = {}
        for key_path, val in updates.items():
            parts = key_path.split(".")
            d = nested
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val

        self.redis_mgr.update_context(session_id, nested)
        sys.stderr.write(f"[Fleet] 指挥将用户参数写入上下文: {json.dumps(nested, ensure_ascii=False)}\n")
        sys.stderr.flush()

    # ─────────────────────────────────────────────────────────────
    
    async def tick(self, state: AgentState):
        """推进一步：按顺序执行子任务"""
        # 首次 tick：清空各收件箱残留消息（此时本会话尚未产生任何消息，安全）
        if self._bus and state.get("_fleet_tick_count", 0) == 0:
            await self._bus.flush_inbox(self._COORD_ID)
            for agent_id in list(self.sub_agents) + [self._INFO_PROCESSOR_ID]:
                await self._bus.flush_inbox(agent_id)

        # 先处理指挥 agent 收件箱（子 agent 的上报/请示/结果）
        await self._process_coordinator_inbox(state)

        sub_tasks = state.get("sub_tasks", [])
        shared = state.get("shared_results", {})
        
        current_task_idx = state.get("_current_task_idx", 0)
        
        # 安全限制：防止无限循环
        tick_count = state.setdefault("_fleet_tick_count", 0) + 1
        state["_fleet_tick_count"] = tick_count
        print(f"[DEBUG tick #{tick_count}] idx={current_task_idx} phase={self._phase} all_done={state.get('_fleet_all_done')} sub_awaiting={state.get('_sub_awaiting','')}", file=sys.stderr, flush=True)
        if tick_count > 200:
            state["_fleet_all_done"] = True
            state["output"] = "[Fleet] 超过最大 tick 次数，强制结束"
            return
        
        all_planned = all(
            t.get("task_id") in self._planned_tasks for t in sub_tasks
        )
        if current_task_idx >= len(sub_tasks) or (
            self._phase == "planning" and all_planned
        ):
            if self._phase == "planning":
                self._phase = "execution"
                state["_current_task_idx"] = 0
                has_work = any(
                    t.get("task_id") in self._planned_tasks
                    for t in sub_tasks
                )
                if not has_work:
                    state["_fleet_all_done"] = True
            else:
                # idx 越界时先确认是否还有未完成的任务
                remaining = [t for t in sub_tasks if t.get("task_id") not in self._completed_tasks]
                if remaining:
                    next_id = remaining[0]["task_id"]
                    for i, t in enumerate(sub_tasks):
                        if t.get("task_id") == next_id:
                            state["_current_task_idx"] = i
                            break
                    return
                state["_fleet_all_done"] = True
            return
        
        current_task = sub_tasks[current_task_idx]
        task_id = current_task.get("task_id", "")
        executor = current_task.get("executor", "uav")
        
        # 已完成则跳到下一个
        if task_id in self._completed_tasks:
            state["_current_task_idx"] = current_task_idx + 1
            return
        
        # 扫描全部未完成任务，找到第一个前置依赖就绪的
        found = False
        for scan_idx in range(0, len(sub_tasks)):
            scan_task = sub_tasks[scan_idx]
            scan_id = scan_task.get("task_id", "")
            if scan_id in self._completed_tasks:
                continue
            # 规划阶段跳过已规划的任务，执行阶段跳过已规划或在执行的
            if self._phase == "planning" and scan_id in self._planned_tasks:
                continue
            prereqs = scan_task.get("prerequisite_tasks", [])
            if self._phase == "execution":
                prereqs_ok = all(p in self._completed_tasks for p in prereqs)
            else:
                prereqs_ok = all(p in self._planned_tasks | self._completed_tasks for p in prereqs)
            if prereqs_ok:
                if scan_idx != current_task_idx:
                    state["_current_task_idx"] = scan_idx
                    current_task_idx = scan_idx
                    current_task = scan_task
                    task_id = scan_id
                    executor = scan_task.get("executor", "uav")
                found = True
                break
        
        if not found:
            print(f"[DEBUG tick #{tick_count}] NO executable task found (deadlock at idx={current_task_idx})", file=sys.stderr, flush=True)
            state["_fleet_all_done"] = True
            state["output"] = "[Fleet] 所有任务被前置依赖阻塞，无法继续执行"
            return
        
        # ====== 所有子任务通过 agent 生命周期执行 ======
        agent_id = self.task_to_agent.get(task_id)
        if not agent_id:
            self._completed_tasks.add(task_id)
            state["_current_task_idx"] = current_task_idx + 1
            return
        
        # 获取对应的 agent（UAV 子 agent / info_processor）
        if agent_id == self._INFO_PROCESSOR_ID:
            agent = self._info_processor_agent
        else:
            agent = self.sub_agents.get(agent_id)
        
        if not agent:
            self._completed_tasks.add(task_id)
            state["_current_task_idx"] = current_task_idx + 1
            return

        is_info_task = agent_id == self._INFO_PROCESSOR_ID

        # 信息处理任务：规划阶段不运行分析 agent（无宏观/详细规划），直接视为已规划，
        # 执行阶段（前置采集任务完成后）再跑分析图直接调工具完成
        if is_info_task and self._phase == "planning":
            self._planned_tasks.add(task_id)
            self._running_tasks.discard(task_id)
            state["_current_task_idx"] = current_task_idx + 1
            self._update_fleet_state(state)
            return

        # 如果子agent空闲，分配任务
        if agent.status == "idle":
            prereq_results = {p: shared.get(p, {}) for p in prereqs}
            print(f"[FLEET] 分配任务 {task_id}({current_task.get('task_name','')}) → agent={agent_id}", file=sys.stderr, flush=True)
            for pid, pval in prereq_results.items():
                summary = str(pval.get("output", ""))[:120]
                has_actions = len(pval.get("detail_actions", []))
                print(f"[FLEET]   前置 {pid}: output={summary!r} actions_count={has_actions} status={pval.get('status','')}", file=sys.stderr, flush=True)
            enriched_task = {**current_task}
            enriched_task["prerequisite_results"] = prereq_results
            
            if self._phase == "execution" and task_id in self._saved_plans:
                # 执行阶段：跳过规划，直接注入已保存的详细规划
                plan = self._saved_plans[task_id]
                agent.assign_task(enriched_task)
                # 覆盖子 agent 状态，直接进入执行模式
                agent.state["detail_actions"] = plan["detail_actions"]
                agent.state["detail_plan_done"] = True
                agent.state["detail_plan_confirmed"] = True
                agent.state["current_step_idx"] = 0
                agent.state["plan_generated"] = True
                agent.state["macro_plan_confirmed"] = True
            else:
                # 规划阶段：正常分配任务
                agent.assign_task(enriched_task)
            self._running_tasks.add(task_id)
        
        # 每 tick 执行一步
        if agent.status == "running":
            # 规划阶段：用户确认详规 → 不执行 run_step，直接保存规划
            if self._phase == "planning":
                s = agent.state
                if (s.get("detail_plan_done") and not s.get("detail_plan_confirmed")
                        and not s.get("pending_question") and s.get("user_input")):
                    s["detail_plan_confirmed"] = True
                    actions = list(s.get("detail_actions", []))
                    if actions:
                        self._planned_tasks.add(task_id)
                        self._saved_plans[task_id] = {"detail_actions": actions}
                        shared[task_id] = {
                            "status": "planned",
                            "detail_actions": actions,
                            "output": s.get("output", ""),
                            "progress": 0.0,
                        }
                        self._running_tasks.discard(task_id)
                        agent.reset_for_next_task()
                        state["_current_task_idx"] = current_task_idx + 1
                        self._update_fleet_state(state)
                        return
                    # 无 actions → 直接视为完成
                    self._planned_tasks.add(task_id)
                    self._running_tasks.discard(task_id)
                    agent.reset_for_next_task()
                    state["_current_task_idx"] = current_task_idx + 1
                    self._update_fleet_state(state)
                    return
            
            await agent.run_step()

            # 保存子agent状态到 Redis
            if self.redis_mgr:
                self.redis_mgr.save_agent_state(agent_id, agent.state)
            
            # 诊断：读取子 agent 实际状态
            _actual_pq = agent.state.get("pending_question", "")
            _actual_dpc = agent.state.get("detail_plan_confirmed", False)
            _actual_dpd = agent.state.get("detail_plan_done", False)
            _actual_ui = agent.state.get("user_input", "")
            
            # 子agent有等待用户确认的问题，传播到 coordinator（信息处理任务除外，输出即完成）
            if (agent.has_pending_question or _actual_pq) and not is_info_task:
                if not agent.has_pending_question:
                    print(f"[DEBUG tick] has_pending_question=False but state.pending_question='{_actual_pq[:50]}' (fallback)", file=sys.stderr, flush=True)
                state["_sub_awaiting"] = task_id
                state["pending_question"] = _actual_pq
                state["output"] = agent.state.get("output", "")
                print(f"[DEBUG tick] PENDING_QUESTION -> _sub_awaiting={task_id}", file=sys.stderr, flush=True)
                return
            
            if _actual_dpc:
                print(f"[DEBUG tick] agent status after run_step: status={agent.status} pq={bool(_actual_pq)} dpc={_actual_dpd} sidx={agent.state.get('current_step_idx', -1)} is_done={agent.is_done}", file=sys.stderr, flush=True)
            
            # 子agent完成（执行阶段）
            if agent.is_done:
                if self._phase == "planning":
                    shared[task_id] = {
                        "status": "planned",
                        "output": agent.last_output,
                        "progress": 1.0,
                    }
                else:
                    # 执行阶段：真正完成
                    self._completed_tasks.add(task_id)
                    shared[task_id] = {
                        "status": "done",
                        "output": agent.last_output,
                        "progress": 1.0,
                        "detail_actions": agent.state.get("detail_actions", []),
                    }
                
                self._running_tasks.discard(task_id)
                agent.reset_for_next_task()
                state["_current_task_idx"] = current_task_idx + 1
        
        # 更新 fleet 状态
        self._update_fleet_state(state)
    
    def _update_fleet_state(self, state: AgentState):
        """将各子agent状态汇总到 uav_states"""
        uav_states = {}
        for uav_id, agent in self.sub_agents.items():
            uav_states[uav_id] = agent.get_status_dict()
        state["uav_states"] = uav_states

    def get_aggregated_results(self, state: AgentState) -> dict:
        """聚合所有子agent的执行结果"""
        results = {
            "fleet_id": state.get("fleet_id", ""),
            "total_tasks": len(state.get("sub_tasks", [])),
            "completed_tasks": len(self._completed_tasks),
            "uav_results": {},
        }
        
        for uav_id, agent in self.sub_agents.items():
            results["uav_results"][uav_id] = {
                "status": agent.status,
                "progress": agent.progress,
                "output": agent.last_output,
                "detail_actions": agent.state.get("detail_actions", []),
            }

        if self._info_processor_agent:
            ip = self._info_processor_agent
            results["info_processor_results"] = {
                "status": ip.status,
                "progress": ip.progress,
                "output": ip.last_output,
                "detail_actions": ip.state.get("detail_actions", []),
            }

        return results
