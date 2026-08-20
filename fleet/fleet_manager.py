"""
Fleet 生命周期管理器：
管理多个 SubAgent 的创建、执行、状态收集、结果聚合
"""
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict
from core.state import AgentState, Deps
from core.redis_manager import RedisManager
from core.config import BASE_DIR, COORDINATOR_ID
from core.kafka_bus import get_kafka_bus
from core.timing import timing
from agent.sub_agent import SubAgent
from tools.script_writer import write_action_step, write_pause_marker, refresh_dependencies_with_llm


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


# LLM 单次调用总超时（秒）：与 tool_executor 保持一致，防止流式响应慢导致永久挂起
FLEET_LLM_TIMEOUT = 60.0


async def _ainvoke_with_timeout(llm, prompt: str, timeout: float = FLEET_LLM_TIMEOUT):
    """带总超时的 LLM 调用：超时抛 TimeoutError，由调用方降级处理。"""
    from langchain_core.messages import HumanMessage
    response = await asyncio.wait_for(
        llm.ainvoke([HumanMessage(content=prompt)]),
        timeout=timeout,
    )
    return response.content.strip()


class FleetManager:
    """Fleet 管理器：协调多个子agent的生命周期"""
    
    _COORD_ID = "_coordinator_"
    _PROCESSOR_ID = "_processor_"

    def __init__(self, deps: Deps, redis_mgr: RedisManager = None):
        self.deps = deps
        self.redis_mgr = redis_mgr
        self.sub_agents: Dict[str, SubAgent] = {}   # uav_id → SubAgent
        self.task_to_agent: Dict[str, str] = {}       # task_id → agent_id
        self._completed_tasks: set = set()            # 已完成的 task_id 集合（执行完成）
        self._dispatched_tasks: set = set()           # 已通过 Kafka 派发的 task_id 集合（每个任务只派发一次）
        self._processor_agent: SubAgent | None = None  # 信息处理类子任务的 agent
        self._bus = get_kafka_bus()
        # 缺参请示连续卡死 tick 阈值：超过则强制跳过动作（见 _resolve_stuck_param_waits）
        self._STUCK_AWAIT_TICKS = 2
        # 连续无推进 tick 阈值：超过则强制结束，避免无限空转刷屏
        self._STALL_TICK_LIMIT = 30
        # 上次保存时的 agent 状态签名（避免空转期重复 save_agent_state 刷日志）
        self._last_saved_sig: Dict[str, str] = {}
        # 本轮 tick 是否有实质推进（消费了消息 / 派发了任务）——用于空转保护
        self._last_tick_activity = False
    
    def initialize_sub_agents(self, state: AgentState):
        """根据 uav_configs 创建子 agent 实例 + 创建信息处理 agent"""
        configs = state.get("uav_configs", [])
        assignments = state.get("sub_task_assignments", {})
        session_id = state.get("session_id", "default")
        self._last_saved_sig.clear()
        
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
        self._processor_agent = SubAgent(
            agent_id=self._PROCESSOR_ID, deps=self.deps,
            initial_config={}, mode="processor", redis_mgr=self.redis_mgr, session_id=session_id
        )
        ip_task_ids = assignments.get("processor", [])
        self._processor_agent._assigned_task_ids = ip_task_ids
        for tid in ip_task_ids:
            self.task_to_agent[tid] = self._PROCESSOR_ID

    # ── 指挥 agent 收件箱：处理子 agent 的上报/请示/结果 ─────────────

    async def _process_coordinator_inbox(self, state: AgentState):
        """拉取指挥收件箱：report/result 程序化归档；request 交给指挥 LLM 决策回复。

        持续拉取直到本轮无新消息（上限 5 轮），确保同一时刻投递的多条消息全部消费，
        避免某次 poll 只取到部分消息、余下消息等下一 tick 时被完成路径/会话切换抢先。
        """
        if not self._bus:
            return
        for _ in range(5):
            try:
                msgs = await self._bus.poll(self._COORD_ID, timeout=0.1)
            except Exception as e:
                sys.stderr.write(f"[Fleet] 指挥收件箱拉取失败: {e}\n")
                sys.stderr.flush()
                return
            if not msgs:
                break

            session_id = state.get("session_id", "default")
            mailbox = state.setdefault("coordinator_mailbox", [])
            self._last_tick_activity = True
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

    def _finalize_script_segment(self, state: AgentState, reason: str = ""):
        """在统一脚本中写入暂停标记（整个任务仍写入同一个文件）"""
        write_pause_marker(state.get("session_id", "default"), reason)

    def _write_action_script_line(self, state: AgentState, msg: dict):
        """把已执行的动作 + 实际调用参数值写入脚本文件（JSON，steps 中追加一条）"""
        payload = msg.get("payload") or {}
        from_id = msg.get("from", "")
        session_id = state.get("session_id", "default")
        action = payload.get("action", {}) or {}
        output = payload.get("output", "") or ""
        output_fields = payload.get("output_fields") or []
        write_action_step(session_id, from_id, action, action.get("tool_inputs", {}), output, output_fields)

    async def _handle_coordinator_request(self, state: AgentState, msg: dict):
        """子 agent 请示：规划确认/缺参请求走专项处理；其余交指挥 LLM 决策"""
        payload = msg.get("payload") or {}
        if payload.get("request_type") == "plan_confirm":
            await self._handle_plan_confirm_request(state, msg)
            return
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

        # 跨 agent 数据依赖：其他 agent 的工具能产出缺失字段 → 派发动态子任务并挂起请求者
        dispatched = await self._try_dispatch_dependency_producer(
            state, msg, missing, from_id, correlation_id, session_id
        )
        if dispatched:
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

        # 推导失败/无 LLM/超限 → 跳过该动作（不再询问用户）
        param_desc = "、".join(f"{p.get('name', '')}({p.get('description', '')})" for p in missing)
        sys.stderr.write(f"[Fleet] 缺参推导失败/超限，跳过动作: {tool_name} 缺 {param_desc}\n")
        sys.stderr.flush()
        await self._send_reply(state, from_id,
            f"参数 {param_desc} 无法从现有信息推导，已跳过该动作。",
            correlation_id,
            payload={"kind": "param_failed", "params": [p.get("name", "") for p in missing]})

    async def _try_dispatch_dependency_producer(self, state: AgentState, msg: dict,
                                                missing: list, from_id: str,
                                                correlation_id: str, session_id: str) -> bool:
        """跨 agent 数据依赖解析：若其他 agent 的工具能产出缺失字段，派发动态子任务并挂起请求者。

        返回 True 表示已派发动态子任务（请求者等待任务完成后由 _tick_agent 回复重试）。
        """
        if not self.deps:
            return False
        from core.ollama_utils import producer_algos_for_field

        # 找出能产出缺失字段的算法，且执行该算法的 agent 不是请求者自身
        target = None
        for p in missing:
            field = p.get("name", "") if isinstance(p, dict) else p
            if not field:
                continue
            for aname in producer_algos_for_field(field):
                agents = list(self.sub_agents.items()) + [(self._PROCESSOR_ID, self._processor_agent)]
                for agent_id, agent in agents:
                    if agent_id == from_id or agent is None:
                        continue
                    if aname in agent.deps.tools:
                        target = {"agent": agent, "field": field, "algo": aname}
                        break
                if target:
                    break
            if target:
                break
        if not target:
            return False

        agent = target["agent"]
        algo, field = target["algo"], target["field"]

        # 动态任务 id：取 T 序列中未占用的编号
        occupied = {t.get("task_id") for t in state.get("sub_tasks", [])}
        occupied |= self._completed_tasks | self._dispatched_tasks
        next_id = 1
        while f"T{next_id}" in occupied:
            next_id += 1
        tid = f"T{next_id}"

        task = {
            "task_id": tid,
            "task_name": f"执行{algo}产出{field}",
            "goal": f"调用算法 {algo} 对目标执行侦察，产出字段 {field} 数据，供信息处理 agent 使用。",
            "executor": "uav",
            "assigned_uav_role": None,
            "prerequisite_tasks": [],
            "constraints": [f"必须调用算法 {algo} 采集并产出 {field} 数据"],
            "dynamic_dependency": True,
        }
        state.setdefault("sub_tasks", []).append(task)
        if tid not in agent._assigned_task_ids:
            agent._assigned_task_ids.append(tid)
        self.task_to_agent[tid] = agent.agent_id
        state.setdefault("_pending_dependency", {})[tid] = {
            "requester": from_id,
            "correlation_id": correlation_id,
            "tool_name": msg.get("payload", {}).get("tool_name", ""),
            "action_name": msg.get("payload", {}).get("action_name", ""),
            "field": field,
            "missing": missing,
        }
        # 不直接派发：加入分配队列，由 tick 的 _tick_agent 在该 agent 空闲时正常派发
        # （避免在 agent 忙碌时用 Kafka 消息覆盖其当前任务）
        print(f"[FLEET] 缺参数据依赖：{from_id} 缺 {field} → 排队动态子任务 {tid}({algo}) → agent={agent.agent_id}（空闲时派发）",
              file=sys.stderr, flush=True)
        return True

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
            raw = await _ainvoke_with_timeout(llm, prompt)
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
        """子 agent 算法/分析结果：仅归档 shared_results。

        上下文更新不再经指挥 LLM 提取：stub 算法的空占位由
        tool_executor._write_output_placeholders 建立，真实 exe（path_planning）的航迹与
        位置由 tool_executor._write_path_to_redis 确定性写入，各执行 agent 自行落库。
        避免指挥 LLM 把算法输出中的字段名/占位文本当成值写进上下文造成污染。
        """
        payload = msg.get("payload") or {}
        output = payload.get("algorithm_output", "") or msg.get("content", "")
        if not output:
            return
        from_id = msg.get("from", "")

        # 归档 shared_results
        key = payload.get("task_id") or from_id
        state.setdefault("shared_results", {})[key] = {
            "status": "done",
            "output": str(output)[:500],
            "source": from_id,
        }

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
            raw = await _ainvoke_with_timeout(llm, prompt)
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

    async def _handle_plan_confirm_request(self, state: AgentState, msg: dict):
        """规划确认请示：子 agent 的宏观/详细规划已产出，自动确认（不再询问用户）"""
        payload = msg.get("payload") or {}
        from_id = msg.get("from", "")
        task_id = payload.get("task_id", "")
        task_name = payload.get("task_name", "")
        plan_type = payload.get("plan_type", "macro_plan")
        correlation_id = msg.get("correlation_id")

        label = "详细规划" if plan_type == "detail_plan" else "宏观规划"
        sys.stderr.write(f"[Fleet] {from_id} 的{label}已自动确认: {task_id}\n")
        sys.stderr.flush()

        await self._send_reply(state, from_id,
            f"{label}方案已确认，继续执行。",
            correlation_id,
            payload={"kind": "user_confirm", "plan_type": plan_type})

    def _build_confirm_question(self, state: AgentState) -> str:
        """返回队首（最先到达）的待确认规划方案问题；其余待确认的排队等待，不合并展示"""
        awaiting = state.get("_awaiting_user_confirm") or {}
        if not awaiting:
            return ""
        info = next(iter(awaiting.values()))
        return (
            f"{info['agent_id']} 的任务「{info['task_name']}」{info['label']}方案待确认：\n"
            f"{info['plan_text'] or '（无内容）'}\n"
            f"请确认（输入确认继续）。"
        )

    async def _handle_user_plan_confirm(self, state: AgentState, user_input: str):
        """用户回答规划确认问题：确认→回发 user_confirm reply；修改意见→回发 user_reject（子 agent 重新规划）。

        多槽位 + 逐条确认：默认处理最先到达（队首）的待确认方案，其余排队等待；
        也可输入“UAV_xx 确认/修改意见”指定处理某一架（其余保留，不覆盖、不丢失）。
        """
        awaiting = state.get("_awaiting_user_confirm") or {}
        text = (user_input or "").strip()
        confirm_words = ("确认", "确定", "同意", "可以", "好的", "好", "ok", "yes", "是", "继续", "没问题")

        if not awaiting:
            sys.stderr.write("[Fleet] 收到规划确认回复但缺少等待信息\n")
            sys.stderr.flush()
            return

        # 单独处理：输入以 agent_id 开头（如 "UAV_2 确认" / "UAV_2 返程重新规划"）时只作用于该无人机
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:：]?\s*(\S.*)$", text)
        if m and m.group(1) in awaiting:
            pending = [awaiting[m.group(1)]]
            text = m.group(2).strip()
        else:
            pending = [next(iter(awaiting.values()))]

        if text in confirm_words or text in ("确认继续",):
            for info in pending:
                await self._send_reply(
                    state, info["agent_id"],
                    "指挥已确认你的规划方案，请继续。",
                    info["correlation_id"],
                    payload={"kind": "user_confirm"},
                )
        else:
            for info in pending:
                await self._send_reply(
                    state, info["agent_id"],
                    f"用户要求修改规划方案：{text}",
                    info["correlation_id"],
                    payload={"kind": "user_reject", "feedback": text},
                )

        # 清理已处理的槽位；其余无人机的确认请求保留（不覆盖、不丢失）
        for info in pending:
            awaiting.pop(info["agent_id"], None)
        state["user_input"] = ""
        if awaiting:
            remaining = next(iter(awaiting.values()))
            state["_sub_awaiting"] = remaining["task_id"]
            question = self._build_confirm_question(state)
            state["pending_question"] = question
            state["output"] = question
        else:
            state["_awaiting_user_confirm"] = {}
            state["_sub_awaiting"] = ""
            state["pending_question"] = ""
            state["output"] = ""

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
                "3. 坐标参数仍用 x,y 格式。\n"
                "4. 若缺失参数中同时含最低频率和最高频率（如 minFreq/maxFreq、targetMinFreq/targetMaxFreq），"
                "而用户用范围回答（如 \"2GHz到3GHz\"、\"1~2GHz\"、\"2GHz 至 3GHz\"），则拆成两个值分别填入"
                "（minFreq 填下限、maxFreq 填上限，保留单位）。\n\n"
                "只输出 JSON 对象，key 为参数名，value 为保留单位的值字符串。"
                "若某个参数无法从回答中得到明确值，则不输出该 key。"
                "若全部无法提取，输出 {}。"
            )
            try:
                raw = await _ainvoke_with_timeout(llm, prompt)
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

        # 防死循环：用户多次未能提供有效参数值 → 标记当前动作失败，不再无限重问
        if not values:
            gkey = f"{agent_id}:{info.get('tool_name', '')}"
            gc = state.setdefault("_user_param_giveup", {}).get(gkey, 0) + 1
            state["_user_param_giveup"][gkey] = gc
            if gc >= 3:
                sys.stderr.write(f"[Fleet] 用户多次无法提供参数({gkey})，标记当前动作失败\n")
                sys.stderr.flush()
                await self._send_reply(
                    state, agent_id,
                    "用户多次无法提供该参数，已标记当前动作失败。",
                    correlation_id,
                    payload={"kind": "param_failed", "params": [p.get("name", "") for p in params]},
                )
                state["_awaiting_user_param"] = {}
                state["_sub_awaiting"] = ""
                state["user_input"] = ""
                return

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
            "3. 目标参数（如频率、带宽、信号强度）写入对应 target 条目（用 id 定位，如 targets.T1.minFreq、targets.T1.maxFreq）。\n"
            "4. 无人机自身属性写 uavs 条目（如 uavs.UAV_1.minFreq）。\n"
            "5. 确实无处安放时，可新增通用字段但没有更好的位置则返回 {}\n\n"
            "【值单位规则】\n"
            "1. **保留用户提供的值和单位**，禁止剥离单位、禁止换算。如 \"2GHz\" 就写 \"2GHz\"。\n"
            "2. 坐标字符串（x,y）拆成 Longitude 与 Latitude 两条。\n"
            "3. 频率范围字符串（如 \"2GHz~3GHz\"、\"2GHz到3GHz\"）拆成 minFreq（下限）与 maxFreq（上限）两条，值保留单位。\n\n"
            "【结果归属判断】\n"
            "1. 优先写入 task_id 对应的 target（如 task_id=T1 与 target.id=T1）。\n"
            "2. 无法确定目标时，根据参数名语义放入 target 第一项。\n\n"
            "请输出 JSON 对象表示要更新到 Redis 的字段映射。\n",
            'key 为点号分隔路径（如 "targets.T1.minFreq"），value 为带单位的参数值。'
            "value 若是不带单位的坐标字符串，拆成经度/纬度两条；若是频率范围字符串，拆成 minFreq/maxFreq 两条。"
        ]
        prompt = "".join(prompt_parts)

        try:
            content = await _ainvoke_with_timeout(llm, prompt)
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
    
    async def _dispatch_task_via_kafka(self, state: AgentState, agent, task: dict):
        """通过 Kafka 向子 agent 派发任务（替代直接方法调用 assign_task）

        携带任务对象，子 agent 收到后自行 宏观规划→确认→详细规划→确认→执行。
        """
        if not self._bus:
            sys.stderr.write("[Fleet] Kafka 总线不可用，无法派发任务\n")
            sys.stderr.flush()
            return
        session_id = state.get("session_id", "default")
        payload = {
            "session_id": session_id,
            "task": task,
        }
        await self._bus.send(
            self._COORD_ID, agent.agent_id, "task",
            f"指挥分配子任务: {task.get('task_name', '')}",
            payload=payload,
        )

    async def tick(self, state: AgentState):
        """推进一步：逐 agent 推进生命周期（派发就绪任务 / 推进运行 / 完成收尾）"""
        self._last_tick_activity = False
        # 首次 tick：清空各收件箱残留消息（此时本会话尚未产生任何消息，安全）
        if self._bus and state.get("_fleet_tick_count", 0) == 0:
            await self._bus.flush_inbox(self._COORD_ID)
            for agent_id in list(self.sub_agents) + [self._PROCESSOR_ID]:
                await self._bus.flush_inbox(agent_id)

        # 先处理指挥 agent 收件箱（子 agent 的上报/请示/结果）
        with timing.track("指挥收件箱处理"):
            await self._process_coordinator_inbox(state)

        sub_tasks = state.get("sub_tasks", [])

        # 安全限制：防止无限循环
        tick_count = state.setdefault("_fleet_tick_count", 0) + 1
        state["_fleet_tick_count"] = tick_count
        if tick_count > 200:
            state["_fleet_all_done"] = True
            state["output"] = "[Fleet] 超过最大 tick 次数，强制结束"
            return

        # 任务是否已全部完成（_current_task_idx 仅供展示）
        remaining = [t for t in sub_tasks if t.get("task_id") not in self._completed_tasks]
        if remaining:
            first_id = remaining[0]["task_id"]
            for i, t in enumerate(sub_tasks):
                if t.get("task_id") == first_id:
                    state["_current_task_idx"] = i
                    break
        else:
            # 全部任务完成：会话切换前最后一次排空收件箱。
            # 最后一步动作的 action_executed 上报可能仍滞留在生产者缓冲中
            # （aiokafka 默认 linger_ms=5ms），先等待其刷新再消费，否则该上报
            # 会在会话切换后被「非本会话」过滤器丢弃，导致脚本漏掉最后一行。
            await self._process_coordinator_inbox(state)
            await asyncio.sleep(0.1)
            await self._process_coordinator_inbox(state)
            state["_fleet_all_done"] = True
            return

        # 阶段1（串行）：完成/出错归档 + 重置 + 派发就绪任务（操作共享状态，必须串行）
        with timing.track("阶段1串行归档/派发"):
            for agent_id, agent in list(self.sub_agents.items()) + [(self._PROCESSOR_ID, self._processor_agent)]:
                if agent is None:
                    continue
                await self._tick_agent(state, agent_id, agent, sub_tasks)

        # 阶段2（并行）：所有 agent 的 run_step 并发推进（RAG/LLM 调用时间重叠，真并行）
        all_agents = [a for _, a in list(self.sub_agents.items()) + [(self._PROCESSOR_ID, self._processor_agent)] if a is not None]
        if all_agents:
            with timing.track("阶段2并行run_step"):
                await asyncio.gather(*[a.run_step() for a in all_agents])
                if self.redis_mgr:
                    for a in all_agents:
                        sig = self._agent_state_sig(a)
                        if sig != self._last_saved_sig.get(a.agent_id):
                            self.redis_mgr.save_agent_state(a.agent_id, a.state)
                            self._last_saved_sig[a.agent_id] = sig

        # 阶段2.5：gather 期间各 agent 新上报的 report/result 立即消费。
        # 子 agent 在 run_step 末尾异步投递 action_executed/result 并置 status=done，
        # 若不在此排空，完成路径/会话切换会跑在这些上报被消费之前，导致最后一步动作漏写脚本。
        with timing.track("阶段2.5排空上报"):
            await self._process_coordinator_inbox(state)

        # 缺参挂起兜底：连续多个 tick 未收到缺参回复且指挥无待处理项 → 强制跳过动作。
        # 每 tick 都调用（内部有连续计数），确保在 tick 上限前一定触发。
        resolved_stuck = await self._resolve_stuck_param_waits(state)

        # 死锁检测：仍有未完成任务，但没有任何 agent 在运行（且没有可派发的就绪任务）
        # 注意：仅因缺参挂起（_awaiting_reply）而显示 running 的 agent 视为「卡死」，
        # 不计入 any_running，否则空转永远不会被死锁检测兜住（tick 上限前无法结束）。
        await_user_param = state.get("_awaiting_user_param") or {}
        await_user_confirm = state.get("_awaiting_user_confirm") or {}
        pending_dep = state.get("_pending_dependency") or {}
        stuck_awaiting = {
            a.agent_id for a in list(self.sub_agents.values()) + ([self._processor_agent] if self._processor_agent else [])
            if a is not None and a._awaiting_reply and a._pending_resolution
            and not (await_user_param.get("agent_id") == a.agent_id
                     or a.agent_id in await_user_confirm
                     or any(d.get("requester") == a.agent_id for d in pending_dep.values()))
        }
        any_running = any(
            a.status == "running" and a.agent_id not in stuck_awaiting
            for a in self.sub_agents.values()
        ) or (
            self._processor_agent is not None and self._processor_agent.status == "running"
            and self._processor_agent.agent_id not in stuck_awaiting
        )
        if remaining and not any_running:
            all_agents = list(self.sub_agents.values()) + ([self._processor_agent] if self._processor_agent else [])
            idle_with_work = any(
                a.agent_id not in stuck_awaiting and self._find_ready_task(a, sub_tasks) is not None
                for a in all_agents if a is not None
            )
            if not idle_with_work:
                # 已尝试强制跳过缺参卡死 agent；仍无法推进则判定真死锁
                still_blocked = [
                    a for a in all_agents if a is not None and (a._awaiting_reply or a.status == "running")
                ]
                if not still_blocked or not resolved_stuck:
                    print(f"[DEBUG tick #{tick_count}] NO executable task found (deadlock)", file=sys.stderr, flush=True)
                    state["_fleet_all_done"] = True
                    state["output"] = "[Fleet] 所有任务被前置依赖阻塞，无法继续执行"
                    return

        # 空转保护：连续多 tick 无任何推进（无回复消费、无任务派发、无真正在跑的 agent）
        # → 强制结束，停止刷日志。注意：缺参挂起的「running」不计入活动（防误判为有进展）。
        genuine_running = any(
            a.status == "running" and a.agent_id not in stuck_awaiting
            for a in list(self.sub_agents.values()) + ([self._processor_agent] if self._processor_agent else [])
            if a is not None
        )
        stall_count = state.get("_fleet_stall_count", 0)
        if self._last_tick_activity or genuine_running:
            stall_count = 0
        else:
            stall_count += 1
        state["_fleet_stall_count"] = stall_count
        if stall_count >= self._STALL_TICK_LIMIT:
            print(f"[DEBUG tick #{tick_count}] STALL: 连续 {stall_count} tick 无推进，强制结束", file=sys.stderr, flush=True)
            state["_fleet_all_done"] = True
            state["output"] = "[Fleet] 连续无推进，强制结束"
            return

        # 更新 fleet 状态
        self._update_fleet_state(state)

    async def _resolve_stuck_param_waits(self, state: AgentState) -> bool:
        """缺参挂起兜底：子 agent 发出缺参请示后连续多个 tick 未收到回复，且指挥侧
        当前没有该 agent 的待处理项（请示/回复丢失、动态依赖断链等），则强制回发
        param_failed 并本地清挂起，让该 agent 跳过当前动作继续执行，避免整队无限空转。

        判定采用「连续卡死 tick 计数」（阈值 _STUCK_AWAIT_TICKS）而非墙钟时长，
        保证在 tick 上限之前一定能触发，且不受 tick 速度影响。

        返回 True 表示至少强制跳过了 1 个 agent（后续应重新评估是否仍卡死）。
        """
        if state.get("_sub_awaiting"):
            return False
        await_user_param = state.get("_awaiting_user_param") or {}
        await_user_confirm = state.get("_awaiting_user_confirm") or {}
        pending_dep = state.get("_pending_dependency") or {}
        agents = list(self.sub_agents.items()) + [(self._PROCESSOR_ID, self._processor_agent)]
        stuck = {}
        for agent_id, agent in agents:
            if agent is None:
                continue
            if not agent._awaiting_reply or not agent._pending_resolution:
                continue
            # 指挥侧正在等这个 agent 的某项回复 → 不判定为卡死
            if await_user_param.get("agent_id") == agent_id:
                continue
            if agent_id in await_user_confirm:
                continue
            if any(d.get("requester") == agent_id for d in pending_dep.values()):
                continue
            stuck[agent_id] = agent

        counters = state.setdefault("_stuck_await_ticks", {})
        resolved = False
        for agent_id, agent in stuck.items():
            counters[agent_id] = counters.get(agent_id, 0) + 1
            if counters[agent_id] < self._STUCK_AWAIT_TICKS:
                continue
            sys.stderr.write(
                f"[Fleet] {agent_id} 缺参挂起连续 {counters[agent_id]} tick 且指挥无对应待处理项，"
                f"强制跳过当前动作继续执行\n"
            )
            sys.stderr.flush()
            try:
                await self._send_reply(
                    state, agent_id,
                    "指挥长时间未回复缺参请示，已标记当前动作失败，请继续后续动作。",
                    agent._awaiting_reply,
                    payload={"kind": "param_failed", "params": []},
                )
            except Exception as e:
                sys.stderr.write(f"[Fleet] {agent_id} 超时强制跳过回复失败: {e}\n")
                sys.stderr.flush()
            # 即使回复因故未送达，也直接本地清挂起，避免 run_step 一直早退
            agent._awaiting_reply = None
            agent._pending_resolution = None
            agent._retry_action = None
            agent.state["_skip_current_action"] = True
            resolved = True
        # 已恢复（不再挂起）的 agent 清零计数
        for agent_id in list(counters):
            if agent_id not in stuck:
                counters.pop(agent_id, None)
        return resolved

    def _agent_state_sig(self, agent) -> str:
        """轻量状态签名：仅覆盖决定「是否值得写 Redis」的关键字段，避免每 tick 全量序列化"""
        s = getattr(agent, "state", {}) or {}
        try:
            actions_sig = json.dumps(s.get("detail_actions", []), ensure_ascii=False, default=str)
        except Exception:
            actions_sig = f"{len(s.get('detail_actions', []))}"
        return json.dumps({
            "status": getattr(agent, "status", ""),
            "progress": getattr(agent, "progress", 0.0),
            "last_output": getattr(agent, "last_output", ""),
            "step_idx": s.get("current_step_idx", 0),
            "pending_q": s.get("pending_question", ""),
            "skip": s.get("_skip_current_action", False),
            "confirmed": s.get("detail_plan_confirmed", False),
            "actions": actions_sig,
            "n_msgs": len(s.get("messages", [])),
        }, ensure_ascii=False)

    async def advance_confirmed_execution(self, state: AgentState):
        """确认等待期的轻量推进：只推进已进入执行阶段（detail_plan_confirmed=True）
        的子 agent 执行已保存动作（快速工具调用），不推进规划/待确认 agent。

        目的：用户逐条确认多架无人机规划时，已确认的无人机立即开始执行，
        无需等确认队列全部清空。挂起中的 agent（_awaiting_reply）在 run_step
        开头直接返回，天然不会在此被推进；长耗时的规划 run_step 也不会在本方法触发。
        """
        exec_agents = [
            a for _, a in list(self.sub_agents.items()) + [(self._PROCESSOR_ID, self._processor_agent)]
            if a is not None and a.state.get("detail_plan_confirmed")
        ]
        if not exec_agents:
            return
        with timing.track("确认期推进执行"):
            await asyncio.gather(*[a.run_step() for a in exec_agents])
            if self.redis_mgr:
                for a in exec_agents:
                    sig = self._agent_state_sig(a)
                    if sig != self._last_saved_sig.get(a.agent_id):
                        self.redis_mgr.save_agent_state(a.agent_id, a.state)
                        self._last_saved_sig[a.agent_id] = sig
            # 推进执行产生的 action_executed 上报需立即消费（写脚本），
            # 否则会在确认队列清空前滞留，导致确认期间执行的步骤漏写脚本。
            await self._process_coordinator_inbox(state)

    def _find_ready_task(self, agent, sub_tasks: list) -> dict | None:
        """按分配顺序找到该 agent 第一个「未完成且前置依赖就绪」的任务"""
        for tid in agent._assigned_task_ids:
            if tid in self._completed_tasks:
                continue
            task = next((t for t in sub_tasks if t.get("task_id") == tid), None)
            if not task:
                continue
            prereqs = task.get("prerequisite_tasks", [])
            if all(p in self._completed_tasks for p in prereqs):
                return task
        return None

    async def _tick_agent(self, state: AgentState, agent_id: str, agent, sub_tasks: list):
        """单个 agent 的串行阶段：完成收尾 + 派发就绪任务（run_step 由 tick 的并行阶段统一 gather 调用）"""
        shared = state.setdefault("shared_results", {})
        task = agent._current_task or {}

        # 完成/出错：归档结果 + 重置 agent，准备下一个任务
        if agent.is_done:
            # 收尾前先排空指挥收件箱：该 agent 最后一步动作的 action_executed/result
            # 由 run_step 异步投递，可能仍在收件箱/生产者缓冲中。先消费完再标记完成，
            # 否则刷新依赖、派发下一任务、会话切换都会抢先，导致最后一步漏写脚本。
            await self._process_coordinator_inbox(state)
            tid = task.get("task_id", "")
            if tid and tid not in self._completed_tasks:
                self._completed_tasks.add(tid)
                shared[tid] = {
                    "status": "done",
                    "output": agent.last_output,
                    "progress": 1.0,
                    "detail_actions": agent.state.get("detail_actions", []),
                }
                if agent.status == "error":
                    shared[tid]["error"] = agent.error
                    sys.stderr.write(f"[Fleet] {agent_id} 任务 {tid} 执行出错: {agent.error}\n")
                    sys.stderr.flush()
                # 任务完成：LLM 兜底补全字段血缘依赖（无未匹配字段时零开销直接返回）
                try:
                    await refresh_dependencies_with_llm(
                        state.get("session_id", "default"),
                        self.deps.llm_no_tools if self.deps else None,
                    )
                except Exception as e:
                    sys.stderr.write(f"[Fleet] LLM 依赖兜底失败: {e}\n")
                    sys.stderr.flush()
                # 动态依赖任务完成：回复挂起的请求者（缺参数据已产出，重试当前动作）
                dep = state.get("_pending_dependency", {}).get(tid)
                if dep:
                    try:
                        await self._send_reply(
                            state, dep.get("requester", ""),
                            f"依赖数据（{dep.get('field', '')}）已由 {agent_id} 产出，请重试当前动作。",
                            dep.get("correlation_id"),
                            payload={"kind": "param_value", "values": {}},
                        )
                    except Exception as e:
                        sys.stderr.write(f"[Fleet] 依赖任务完成回复失败: {e}\n")
                        sys.stderr.flush()
                    state.setdefault("_pending_dependency", {}).pop(tid, None)
            agent.reset_for_next_task()

        # idle：找就绪任务并通过 Kafka 派发（每个任务只派发一次）
        if agent.status == "idle":
            ready_task = self._find_ready_task(agent, sub_tasks)
            if ready_task:
                tid = ready_task.get("task_id", "")
                if tid not in self._dispatched_tasks:
                    prereq_results = {p: shared.get(p, {}) for p in ready_task.get("prerequisite_tasks", [])}
                    enriched_task = {**ready_task, "prerequisite_results": prereq_results}
                    print(f"[FLEET] Kafka 派发任务 {tid}({ready_task.get('task_name','')}) → agent={agent_id}", file=sys.stderr, flush=True)
                    for pid, pval in prereq_results.items():
                        summary = str(pval.get("output", ""))[:120]
                        has_actions = len(pval.get("detail_actions", []))
                        print(f"[FLEET]   前置 {pid}: output={summary!r} actions_count={has_actions} status={pval.get('status','')}", file=sys.stderr, flush=True)
                    await self._dispatch_task_via_kafka(state, agent, enriched_task)
                    self._dispatched_tasks.add(tid)
                    self._last_tick_activity = True
    
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

        if self._processor_agent:
            ip = self._processor_agent
            results["processor_results"] = {
                "status": ip.status,
                "progress": ip.progress,
                "output": ip.last_output,
                "detail_actions": ip.state.get("detail_actions", []),
            }

        return results
