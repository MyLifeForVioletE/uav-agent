"""
子 Agent 实例：按 mode 选择专用图，拥有独立状态
"""
import sys
from pathlib import Path
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.state import AgentState, Deps
from core.prompts import SYSTEM_PROMPT, ANALYST_SYSTEM_PROMPT
from core.redis_manager import RedisManager
from core.ollama_utils import filter_tools_for_role
from core.config import COORDINATOR_ID
from core.kafka_bus import get_kafka_bus, new_msg_id
from core.timing import timing
from agent.graph import build_graph
from agent.analyst_graph import build_analyst_graph


class SendMessageInput(BaseModel):
    """send_message 工具参数"""
    to: str = Field(description="接收方 agent ID：指挥 agent 为 _coordinator_，无人机为 UAV_1 等，信息处理 agent 为 _processor_")
    msg_type: str = Field(description="消息类型：report=向指挥上报状态/结果，request=向指挥请示问题并等待回复，instruction=向其它 agent 下发指令")
    content: str = Field(description="消息内容（用自然中文）")
    await_reply: bool = Field(default=False, description="msg_type=request 时设为 true，表示要等待指挥回复后再继续")


class SubAgent:
    """单个 Agent 实例，封装独立的 graph 和 state
    
    支持两种模式：
    - "uav"：无人机子 agent，生成 UAV 专属的任务描述
    - "processor"：信息处理 agent，生成对无人机采集数据的分析处理类任务描述
    """
    
    def __init__(self, agent_id: str, deps, initial_config: dict = None, mode: str = "uav", redis_mgr: RedisManager = None, session_id: str = "default"):
        self.agent_id = agent_id
        self.mode = mode
        self.deps = self._filter_deps(deps)
        self.redis_mgr = redis_mgr
        self._session_id = session_id
        # 分析 agent 用 3 节点极简图；其余复用 6 节点执行图
        self.graph = build_analyst_graph(self.deps) if mode == "processor" else build_graph(self.deps)
        self.state = self._init_state()      # 独立状态实例
        self.status = "idle"                  # idle | running | done | error
        self.progress = 0.0
        self.last_output = ""
        self.error = None
        self._config = initial_config or {}
        self._assigned_task_ids: list[str] = []
        self._current_task: dict | None = None
        self._planning_retries = 0
        self._max_planning_retries = 5
        self._last_pg_state = False
        # Kafka 消息总线（三种 agent 间通信）
        self._bus = get_kafka_bus()
        self._awaiting_reply: str | None = None   # 同步请示：等待的 correlation_id
        self._plan_confirm_requested: str | None = None   # 已请求用户确认的规划类型（macro_plan/detail_plan）

        # 规划上报：宏观/详细规划已上报标记（避免重复上报）
        self._reported_macro_plan = False
        self._reported_detail_plan = False
        # 缺参请求挂起信息：{action, step_idx, missing_params}
        self._pending_resolution: dict | None = None
        # 指挥回复提供的参数值：{param_name: value}
        self._pending_param_values: dict | None = None
        # 缺参待重试的原子动作（processor 模式：指挥回复后注入图重试执行）
        self._retry_action: dict | None = None
        # 异常应对（contingency）文件是否已写（只写一次）
        self._contingency_plan_written = False

    def _filter_deps(self, deps) -> Deps:
        """按 mode 过滤可用工具表，其余依赖（规划器/LLM）保持不变"""
        tools = filter_tools_for_role(deps.tools, self.mode)
        # 子 agent（UAV / 信息处理）额外提供 send_message 通信工具
        if self.mode in ("uav", "processor"):
            tools = {**tools, "send_message": self._make_send_message_tool()}
        return Deps(
            tools=tools,
            llm_no_tools=deps.llm_no_tools,
            macro_planner=deps.macro_planner,
            constraint_planner=deps.constraint_planner,
            detail_planner=deps.detail_planner,
            decomposer_planner=deps.decomposer_planner,
        )

    def _init_state(self) -> AgentState:
        """初始化子agent的独立状态"""
        is_analyst = self.mode == "processor"
        return {
            "messages": [SystemMessage(content=ANALYST_SYSTEM_PROMPT if is_analyst else SYSTEM_PROMPT)],
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
            "_analyst_mode": is_analyst,
            "_agent_id": self.agent_id,
            "session_id": self._session_id,
        }

    # ── Kafka 消息总线：通信工具与收件箱 ───────────────────────────

    def _make_send_message_tool(self) -> StructuredTool:
        """创建 send_message 工具（闭包捕获本 agent），供 LLM 驱动上报/请示/下发"""
        async def _send(to: str, msg_type: str, content: str, await_reply: bool = False) -> str:
            correlation_id = new_msg_id() if await_reply else None
            msg_id = await self._send_message(to, msg_type, content, correlation_id=correlation_id)
            if await_reply:
                self._awaiting_reply = correlation_id
                return (
                    f"请求已发送给指挥 agent（{to}），正在等待指挥回复。"
                    f"correlation_id={correlation_id}。挂起等待回复，收到回复后继续执行。"
                )
            return f"消息已发送给 {to}（{msg_type}），msg_id={msg_id}"

        return StructuredTool.from_function(
            name="send_message",
            coroutine=_send,
            args_schema=SendMessageInput,
            description=(
                "向指挥 agent（_coordinator_）上报状态/结果（report）、请示问题并等待回复（request，需 await_reply=true），"
                "或向其它子 agent 下发指令（instruction）。发送后等待指挥回复时当前任务会挂起。"
            ),
        )

    async def _send_message(self, to: str, msg_type: str, content: str,
                            payload: dict = None, correlation_id: str = None) -> str:
        """通过 Kafka 向目标 agent 发送消息，返回 msg_id"""
        if self._bus is None:
            raise RuntimeError("Kafka 总线未初始化")
        payload = dict(payload or {})
        payload.setdefault("session_id", self._session_id)
        return await self._bus.send(self.agent_id, to, msg_type, content,
                                    payload=payload, correlation_id=correlation_id)

    async def _drain_inbox(self):
        """拉取收件箱：指令/reply 注入 LLM 上下文；匹配到 await 的回复则解除挂起"""
        if self._bus is None:
            return
        try:
            msgs = await self._bus.poll(self.agent_id, timeout=0.1)
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] 收件箱拉取失败: {e}\n")
            sys.stderr.flush()
            return
        if not msgs:
            return
        for m in msgs:
            mtype = m.get("msg_type", "")
            from_id = m.get("from", "")
            content = m.get("content", "")
            # 只处理本会话的消息，丢弃旧会话/无会话的残留消息
            if m.get("payload", {}).get("session_id") != self._session_id:
                sys.stderr.write(f"[SubAgent {self.agent_id}] 丢弃非本会话消息({mtype}): {content[:60]}\n")
                sys.stderr.flush()
                continue
            if mtype == "task":
                # 指挥通过 Kafka 派发的子任务
                await self._handle_task_assignment(m)
                continue
            if mtype in ("instruction", "reply"):
                if mtype == "reply" and self._awaiting_reply:
                    if m.get("correlation_id") == self._awaiting_reply:
                        self._awaiting_reply = None
                        await self._handle_command_reply(m)
                        continue
                self.state["messages"].append(
                    SystemMessage(content=f"[来自 {from_id} 的消息({mtype})] {content}")
                )
                sys.stderr.write(f"[SubAgent {self.agent_id}] 收到 {from_id} 的 {mtype}: {content[:80]}\n")
                sys.stderr.flush()

    # ───────────────────────────────────────────────────────────────
    
    def assign_task(self, task: dict):
        """将分配的子任务注入子agent状态"""
        self.status = "running"
        self._current_task = task
        self._planning_retries = 0
        
        # 注入 Redis 上下文（UAV/目标坐标等）
        context_block = ""
        if self.redis_mgr:
            try:
                ctx = self.redis_mgr.get_context(self._session_id) or {}
                for uav in ctx.get("uavs", []):
                    if uav.get("id") == self.agent_id:
                        lon = uav.get("Longitude", "")
                        lat = uav.get("Latitude", "")
                        if lon and lat:
                            context_block += f"- 当前位置：{lon},{lat}\n"
                        break
                for t in ctx.get("targets", []):
                    lon = t.get("Longitude", "")
                    lat = t.get("Latitude", "")
                    if lon and lat:
                        context_block += f"- 目标 {t.get('id', '')} 位置：{lon},{lat}\n"
                base = ctx.get("base", {})
                if base.get("Longitude") and base.get("Latitude"):
                    context_block += f"- 基地位置：{base['Longitude']},{base['Latitude']}\n"
            except Exception:
                pass
        
        # 根据模式构造不同描述
        if self.mode == "processor":
            task_desc = (
                f"你是信息处理 agent。请根据以下分析子任务，从可用算法工具中选择最匹配的算法，"
                f"从任务描述、上下文或前置任务结果中提取所需参数，然后直接调用该算法工具。\n"
                f"- 任务名称：{task.get('task_name', '')}\n"
                f"- 任务目标：{task.get('goal', '')}\n"
                f"{context_block}"
            )
        else:
            task_desc = (
                f"你是无人机 {self.agent_id} 的规划执行系统。\n"
                f"你被分配了以下子任务：\n"
                f"- 任务名称：{task.get('task_name', '')}\n"
                f"- 任务目标：{task.get('goal', '')}\n"
                f"{context_block}"
            )
        
        # 添加约束信息
        constraints = task.get("constraints", [])
        if constraints:
            task_desc += f"- 约束条件：{', '.join(constraints)}\n"
        
        # 如果有前置任务结果，注入上下文
        prereq_results = task.get("prerequisite_results", {})
        if prereq_results:
            task_desc += "\n前置任务结果：\n"
            for task_id, result in prereq_results.items():
                if isinstance(result, dict):
                    task_desc += f"- {task_id}: {result.get('output', '无结果')}\n"
                else:
                    task_desc += f"- {task_id}: {result}\n"
        
        self.state["user_input"] = task_desc
        self.state["original_scenario"] = task.get("goal", "")
        self.state["_is_sub_task"] = True
    
    async def _handle_task_assignment(self, msg: dict):
        """接收指挥通过 Kafka 派发的子任务并注入状态（替代指挥直接方法调用）"""
        payload = msg.get("payload") or {}
        task = payload.get("task") or {}
        if not task:
            sys.stderr.write(f"[SubAgent {self.agent_id}] 收到空任务派发，忽略\n")
            sys.stderr.flush()
            return
        self.assign_task(task)
        sys.stderr.write(f"[SubAgent {self.agent_id}] 收到指挥派发任务: {task.get('task_name', '')}\n")
        sys.stderr.flush()
    
    async def run_step(self):
        """执行一步（调用 graph.ainvoke 或直接执行已保存的原子动作）"""
        # 先拉取收件箱：注入指挥指令；收到任务派发则进入 running；收到匹配的回复则解除同步请示挂起
        await self._drain_inbox()

        # 尚未收到任务派发（仍 idle 且无当前任务）：本 tick 不推进
        if self.status == "idle" and not self._current_task:
            return

        # 同步请示挂起：请求已发出但指挥尚未回复 → 本 tick 不推进
        if self._awaiting_reply:
            return

        # 分析 agent：直接跑一次分析图即完成（无规划/无用户确认）
        if self.mode == "processor":
            try:
                with timing.track("分析图(processor)"):
                    self.state = await self.graph.ainvoke(self.state, {"recursion_limit": 50})
                    self.last_output = self.state.get("output", "")
                # 缺参挂起：execute_tools 置 _analyst_param_pending，这里上报指挥并挂起等待
                pending = self.state.get("_analyst_param_pending")
                if pending and not self._awaiting_reply:
                    action = pending.get("action", {})
                    self._retry_action = action
                    self.state["_analyst_param_pending"] = None
                    await self._request_param_resolution(action, pending.get("missing", []))
                    return
                if self._awaiting_reply:
                    # 等待指挥回复期间不视为完成
                    self.status = "running"
                    self.progress = 0.5
                    self.state["pending_question"] = ""
                    self.last_output = "已发送请求，等待指挥回复..."
                    return
                # 分析结果文本不再作为待确认问题，直接视为完成
                self.state["pending_question"] = ""
                self.status = "done"
                self.progress = 1.0
                # 分析结论发给指挥，由指挥提取并写 Redis
                await self._report_analysis_to_coordinator()
            except Exception as e:
                self.status = "error"
                self.error = str(e)
                self.last_output = f"分析执行失败: {e}"
            return

        # 执行阶段：直接按顺序执行已保存的 detail_actions，跳过 LLM 规划层
        if (self.state.get("detail_plan_confirmed") and self.state.get("detail_actions")
                and self.state.get("plan_generated") and self.state.get("macro_plan_confirmed")):
            self.state["pending_question"] = ""
            with timing.track("执行动作"):
                await self._execute_saved_action_step()
            self._sync_status()
            return

        was_pg = self.state.get("plan_generated", False)

        try:
            with timing.track("规划图"):
                self.state = await self.graph.ainvoke(self.state, {"recursion_limit": 50})
                self.last_output = self.state.get("output", "")
            self._sync_status()
            # 向指挥上报已生成的宏观/详细规划
            await self._report_plans_if_ready()
        except Exception as e:
            self.status = "error"
            self.error = str(e)
        else:
            if self._awaiting_reply:
                # 请求已发出，等待指挥回复：清除 pending，避免被当作待用户确认的问题
                self.state["pending_question"] = ""
                self.status = "running"
                self.last_output = "已发送请求，等待指挥回复..."
                return
            if not was_pg and not self.state.get("plan_generated"):
                self._planning_retries += 1
                if self._planning_retries >= self._max_planning_retries:
                    self.status = "error"
                    self.error = f"规划阶段重试 {self._max_planning_retries} 次仍无法生成规划"
                    self.last_output = self.error

    async def _execute_saved_action_step(self):
        """直接从 detail_actions 中执行下一步，不经过 LLM"""
        actions = self.state.get("detail_actions", [])
        idx = self.state.get("current_step_idx", 0)
        # 执行未开始（-1 哨兵）时从第一步开始，避免 actions[-1] 误取最后一步
        if idx < 0:
            idx = 0
            self.state["current_step_idx"] = 0

        # 上一步动作因缺参无法解决被标记跳过
        if self.state.get("_skip_current_action"):
            self.state["_skip_current_action"] = False
            if 0 <= idx < len(actions):
                actions[idx]["_failed"] = True
                self.state["detail_actions"] = actions
                sys.stderr.write(f"[SubAgent {self.agent_id}] 跳过动作: {actions[idx].get('action_name', '')}\n")
                sys.stderr.flush()
            self.state["current_step_idx"] = idx + 1
            self.last_output = "参数无法解决，已跳过当前动作"
            self.state["output"] = self.last_output
            return

        if idx >= len(actions):
            self.status = "done"
            self.progress = 1.0
            self._write_contingency_plan_if_done()
            return

        action = dict(actions[idx])
        from agent.executors import dispatch_executor

        # 执行阶段：清空 tool_inputs 中的猜测值，让 tool_executor 走参数解析流程
        if action.get("executor") == "tool" and "tool_inputs" in action:
            action["tool_inputs"] = {}

        # 注入指挥提供的参数值（用户回答 / 指挥推导补充）
        if self._pending_param_values:
            action.setdefault("tool_inputs", {})
            for k, v in self._pending_param_values.items():
                action["tool_inputs"][k] = v
            sys.stderr.write(f"[SubAgent {self.agent_id}] 注入指挥提供的参数: {self._pending_param_values}\n")
            sys.stderr.flush()
            self._pending_param_values = None

        context = {
            "tools": self.deps.tools,
            "messages": self.state["messages"],
            "llm": self.deps.llm_no_tools,
            "state": self.state,
            "agent_id": self.agent_id,
        }

        # 向指挥上报当前正在执行的动作（处理前先报一次，让指挥感知执行开始）
        await self._report_action_start(action, idx, len(actions))

        sys.stderr.write(f"[SubAgent {self.agent_id}] 执行步骤 {idx+1}/{len(actions)}: {action.get('action_name', '')}\n")
        sys.stderr.flush()

        try:
            result = await dispatch_executor(action, context)
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self.last_output = f"执行失败: {e}"
            return

        # 自身工具能产出缺失字段 → 先把产出动作插入当前步骤之前执行，再继续本动作
        if result.get("self_produce"):
            producer = result["self_produce"].get("action") or result["self_produce"]
            actions = self.state.get("detail_actions", [])
            idx = self.state.get("current_step_idx", 0)
            if 0 <= idx <= len(actions):
                actions.insert(idx, producer)
                self.state["detail_actions"] = actions
            self.last_output = f"已插入自身产出动作: {producer.get('action_name', '')}，先执行后再继续"
            self.state["output"] = self.last_output
            sys.stderr.write(
                f"[SubAgent {self.agent_id}] 缺参自产：插入动作 {producer.get('action_name', '')} @ 步骤 {idx}\n"
            )
            sys.stderr.flush()
            return

        # 缺少参数 → 向指挥请求解决，暂停本步骤（不推进索引、不置错误）
        if result.get("missing_params"):
            await self._request_param_resolution(action, result.get("missing_params", []))
            return

        # 将执行器实际解析出的参数值写回动作，供上报脚本记录
        if result.get("tool_inputs"):
            action["tool_inputs"] = result["tool_inputs"]

        # 动作已执行：把动作及实际解析的参数值上报指挥（写脚本文件）
        await self._report_action_executed(action, result, idx, len(actions))

        self.state["current_step_idx"] = idx + 1
        self.last_output = result.get("output", "")
        self.state["output"] = self.last_output
        
        # 执行成功后：上下文更新由执行 agent 用代码确定性写入 Redis（航迹/位置由
        # tool_executor 直接落库，stub 算法输出保持空占位）；算法原始输出上报指挥仅供归档。
        if result.get("success"):
            if action.get("executor") == "tool":
                await self._report_result_to_coordinator(action, result)

        if result.get("error"):
            self.status = "error"
            self.error = result["error"]
        elif result.get("pending"):
            self.state["pending_question"] = self.last_output or "请输入信息"
        elif idx + 1 >= len(actions):
            self.status = "done"
            self.progress = 1.0
            self._write_contingency_plan_if_done()
        else:
            self._sync_status()

    def _write_contingency_plan_if_done(self):
        """agent 全部动作执行完成后，将各动作的异常应对措施（contingency）写入独立文件。

        只写一次：LLM 生成的 contingency 按 action_name 引用同计划动作，此处与规划出的
        动作列表匹配解析为 action_id，输出到 output/contingency_plan_{session_id}.json。
        """
        if self._contingency_plan_written or self.status != "done":
            return
        try:
            from tools.script_writer import write_contingency_plan
            write_contingency_plan(self._session_id, self.state.get("detail_actions", []))
            self._contingency_plan_written = True
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] contingency 文件写入失败: {e}\n")
            sys.stderr.flush()

    async def _report_result_to_coordinator(self, action: dict, result: dict):
        """算法执行成功后将原始输出发给指挥 agent，由指挥 LLM 提取字段并写 Redis"""
        if not self._bus:
            return
        output = result.get("output", "")
        if not output or result.get("error"):
            return
        task = self._current_task or {}
        payload = {
            "source": "uav",
            "task_id": task.get("task_id", ""),
            "task_name": task.get("task_name", ""),
            "tool_name": action.get("tool_name", ""),
            "goal": action.get("goal", ""),
            "algorithm_output": str(output)[:3000],
        }
        try:
            await self._send_message(
                COORDINATOR_ID, "result",
                f"{self.agent_id} 完成 {action.get('goal', '')}，算法结果已上报",
                payload=payload,
            )
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] 结果上报失败: {e}\n")
            sys.stderr.flush()

    async def _report_plans_if_ready(self):
        """向指挥上报已生成的宏观/详细规划（每个规划只上报一次）"""
        # 宏观规划
        if not self._reported_macro_plan:
            from agent.nodes.planning import _find_macro_plan_json
            macro = self.state.get("macro_phases", [])
            if not macro and self.state.get("plan_generated"):
                macro = _find_macro_plan_json(self.state.get("messages", [])) or []
            if macro:
                self._reported_macro_plan = True
                await self._send_report("macro_plan", {"macro_phases": macro})
        # 详细规划
        if not self._reported_detail_plan:
            actions = self.state.get("detail_actions", [])
            if actions and self.state.get("detail_plan_done"):
                self._reported_detail_plan = True
                await self._send_report("detail_plan", {"detail_actions": actions})

    async def _report_action_start(self, action: dict, idx: int, total: int):
        """向指挥上报当前正在执行的原子动作"""
        await self._send_report("action_start", {
            "action": action,
            "step_idx": idx,
            "total": total,
        })

    async def _report_action_executed(self, action: dict, result: dict, idx: int, total: int):
        """动作执行后：把动作及实际解析出的调用参数值上报指挥（用于写脚本文件）"""
        await self._send_report("action_executed", {
            "action": action,
            "step_idx": idx,
            "total": total,
            "success": result.get("success", False),
            "output": str(result.get("output", ""))[:4000],
            "output_fields": result.get("output_fields") or [],
        })

    async def _send_report(self, report_type: str, extra: dict = None):
        """发 report 消息给指挥（非阻塞），payload 携带结构化上报内容"""
        if not self._bus:
            return
        task = self._current_task or {}
        payload = dict(extra or {})
        payload.update({
            "report_type": report_type,
            "task_id": task.get("task_id", ""),
            "task_name": task.get("task_name", ""),
        })
        try:
            await self._send_message(
                COORDINATOR_ID, "report",
                f"{self.agent_id} 上报 {report_type}",
                payload=payload,
            )
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] {report_type} 上报失败: {e}\n")
            sys.stderr.flush()

    def _needs_plan_confirm(self) -> bool:
        """规划确认已被禁用：始终返回 False，规划产出后自动确认"""
        return False

    async def _request_plan_confirmation(self):
        """规划产出待用户确认：向指挥发送 plan_confirm 请示并挂起，等待指挥转达用户后回复"""
        if not self._bus:
            return
        correlation_id = new_msg_id()
        s = self.state
        task = self._current_task or {}
        if s.get("detail_plan_done") and not s.get("detail_plan_confirmed"):
            plan_type = "detail_plan"
            plan_data = s.get("detail_actions", [])
        else:
            plan_type = "macro_plan"
            plan_data = s.get("macro_phases", [])
            if not plan_data:
                # state 中未解析出宏观阶段时，回退从消息历史提取宏观规划 JSON
                from agent.nodes.planning import _find_macro_plan_json
                plan_data = _find_macro_plan_json(s.get("messages", [])) or []
        payload = {
            "request_type": "plan_confirm",
            "plan_type": plan_type,
            "plan_data": plan_data,
            "task_id": task.get("task_id", ""),
            "task_name": task.get("task_name", ""),
        }
        try:
            await self._send_message(
                COORDINATOR_ID, "request",
                f"{self.agent_id} 的{('详细规划' if plan_type == 'detail_plan' else '宏观规划')}需要用户确认",
                payload=payload,
                correlation_id=correlation_id,
            )
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] 规划确认请求失败: {e}\n")
            sys.stderr.flush()
            self.status = "error"
            self.error = f"规划确认请求失败: {e}"
            return

        self._awaiting_reply = correlation_id
        self._plan_confirm_requested = plan_type
        self.state["pending_question"] = ""
        self.status = "running"
        self.last_output = "规划方案已提交，等待用户确认..."
        self.state["output"] = self.last_output
        sys.stderr.write(f"[SubAgent {self.agent_id}] {plan_type} 已请求用户确认，挂起等待\n")
        sys.stderr.flush()

    async def _request_param_resolution(self, action: dict, missing_params: list):
        """缺参：向指挥发送 request，等待指挥回复（推导算法 / 询问用户）"""
        if not self._bus:
            return
        correlation_id = new_msg_id()
        names = "、".join(p.get("name", "") for p in missing_params)
        task = self._current_task or {}
        payload = {
            "request_type": "missing_param",
            "tool_name": action.get("tool_name", action.get("action_name", "")),
            "action_name": action.get("action_name", ""),
            "goal": action.get("goal", ""),
            "missing_params": missing_params,
            "action": action,
            "task_id": task.get("task_id", ""),
        }
        try:
            await self._send_message(
                COORDINATOR_ID, "request",
                f"{self.agent_id} 执行动作 {action.get('action_name', '')} 缺少参数：{names}，请求指挥解决",
                payload=payload,
                correlation_id=correlation_id,
            )
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] 缺参请求失败: {e}\n")
            sys.stderr.flush()
            self.status = "error"
            self.error = f"缺参请求失败: {e}"
            return

        self._awaiting_reply = correlation_id
        self._pending_resolution = {
            "action": action,
            "step_idx": self.state.get("current_step_idx", 0),
            "missing_params": missing_params,
        }
        self._retry_action = action
        self.state["pending_question"] = ""
        self.status = "running"
        self.last_output = f"缺少参数 {names}，已向指挥请求解决..."
        self.state["output"] = self.last_output
        sys.stderr.write(f"[SubAgent {self.agent_id}] 缺参请求已发送，挂起等待指挥回复\n")
        sys.stderr.flush()

    async def _handle_command_reply(self, msg: dict):
        """处理指挥的同步回复：解析结构化 payload（algorithm / param_value）"""
        payload = msg.get("payload") or {}
        kind = payload.get("kind")
        content = msg.get("content", "")
        sys.stderr.write(f"[SubAgent {self.agent_id}] 收到指挥回复 kind={kind}: {content[:100]}\n")
        sys.stderr.flush()
        if kind == "algorithm":
            action = payload.get("action")
            if action:
                idx = self.state.get("current_step_idx", 0)
                actions = self.state.get("detail_actions", [])
                actions.insert(idx, action)
                self.state["detail_actions"] = actions
                self.last_output = f"已按指挥指示插入算法动作: {action.get('action_name', '')}"
                self.state["output"] = self.last_output
                sys.stderr.write(f"[SubAgent {self.agent_id}] 已插入动作: {action.get('action_name', '')}\n")
                sys.stderr.flush()
            self.state["pending_question"] = ""
        elif kind == "param_value":
            values = payload.get("values") or {}
            if self._pending_resolution:
                # 注入到当前挂起动作的 tool_inputs
                action = self._pending_resolution.get("action", {})
                action.setdefault("tool_inputs", {})
                for k, v in values.items():
                    action["tool_inputs"][k] = v
                self._pending_param_values = dict(values)
                # 同步回 detail_actions 以便后续重试仍保留
                idx = self._pending_resolution.get("step_idx", 0)
                actions = self.state.get("detail_actions", [])
                if 0 <= idx < len(actions):
                    actions[idx]["tool_inputs"] = action.get("tool_inputs", {})
                self._pending_resolution = None
            # processor 模式：把待重试动作注入分析图，下一 tick 重新执行
            if self.mode == "processor" and self._retry_action:
                state = self.state
                state["_inject_retry_action"] = self._retry_action
                self._retry_action = None
                sys.stderr.write(f"[SubAgent {self.agent_id}] 缺参已解决，注入重试动作，等待重试执行\n")
                sys.stderr.flush()
            self.last_output = f"已收到指挥提供的参数值: {values}"
            self.state["output"] = self.last_output
        elif kind == "param_failed":
            # 用户/指挥多次无法解决缺参：跳过当前动作，继续后续步骤
            self._pending_resolution = None
            self._retry_action = None
            self.state["pending_question"] = ""
            self.state["_skip_current_action"] = True
            self.last_output = f"参数无法解决（{payload.get('params', [])}），已跳过当前动作。"
            self.state["output"] = self.last_output
            self.state["pending_question"] = ""
        elif kind == "user_confirm":
            # 用户已确认规划方案：宏观确认 → 触发详细规划链路；详细确认 → 直接进入执行
            self._plan_confirm_requested = None
            self.state["pending_question"] = ""
            if self.state.get("detail_plan_done") and not self.state.get("detail_plan_confirmed"):
                self.state["detail_plan_confirmed"] = True
                self.state["_intent"] = "chat"
            elif self.state.get("plan_generated") and not self.state.get("macro_plan_confirmed"):
                self.state["user_input"] = "确认"
                self.state["_intent"] = "confirm"
            self.last_output = "规划方案已确认，继续执行"
            self.state["output"] = self.last_output
        elif kind == "user_reject":
            # 用户要求修改规划：把修改意见注入，重新规划
            self._plan_confirm_requested = None
            self._reported_detail_plan = False
            feedback = payload.get("feedback", "") or content
            self.state["pending_question"] = ""
            if (self.state.get("detail_plan_done")
                    and not self.state.get("detail_plan_confirmed")):
                # 驳回的是详细规划：重置 detail_plan_done 使 router 不再命中"待确认"分支，
                # 保留 detail_actions 供 planning_prep 注入旧动作参考。
                # planning_prep 检测 _replan_feedback 后走单次重规划路径，
                # call_llm 中 _replanning_detail 块会用新动作完全替换 detail_actions。
                self.state["_replan_feedback"] = feedback
                self.state["detail_plan_done"] = False
                self.state["current_phase_idx"] = -1
            elif (self.state.get("plan_generated")
                    and not self.state.get("macro_plan_confirmed")):
                # 驳回的是宏观规划：重置 plan_generated，重新走宏观规划（feedback 随 user_input 注入）
                self.state["plan_generated"] = False
            self.state["user_input"] = f"用户要求修改规划方案，请按以下意见重新规划：{feedback}"
            self.last_output = "用户要求修改规划，重新规划中..."
            self.state["output"] = self.last_output
        else:
            self.state["messages"].append(SystemMessage(content=f"[指挥回复] {content}"))
            if content and not self.state.get("pending_question"):
                self.state["pending_question"] = ""

    async def _report_analysis_to_coordinator(self):
        """分析完成后：把分析结论发给指挥，由指挥提取并写 Redis（补 analysis_result 缺口）"""
        if not self._bus or not self.last_output:
            return
        task = self._current_task or {}
        goal = task.get("goal", "")
        payload = {
            "source": "analysis",
            "task_id": task.get("task_id", ""),
            "task_name": task.get("task_name", ""),
            "tool_name": "",
            "goal": goal,
            "algorithm_output": str(self.last_output)[:3000],
        }
        try:
            await self._send_message(
                COORDINATOR_ID, "result",
                f"{self.agent_id} 完成分析，结论已上报",
                payload=payload,
            )
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] 分析结论上报失败: {e}\n")
            sys.stderr.flush()

    async def _apply_post_action_update(self, action: dict, result: dict):
        """已废弃：上下文更新现由 tool_executor 确定性写入 Redis，不再经 LLM 生成。"""
        return

    def _sync_status(self):
        """从子agent状态同步 status/progress"""
        s = self.state
        
        if s.get("detail_plan_confirmed"):
            actions = s.get("detail_actions", [])
            idx = s.get("current_step_idx", 0)
            if actions:
                self.progress = min(idx / len(actions), 1.0)
            else:
                self.progress = 1.0
        
        if s.get("detail_plan_done") and s.get("detail_plan_confirmed"):
            if s.get("current_step_idx", 0) >= len(s.get("detail_actions", [])):
                self.status = "done"
                self.progress = 1.0
    
    def inject_user_input(self, text: str):
        """注入用户输入（用于确认等）"""
        self.state["user_input"] = text
        self.state["pending_question"] = ""
    
    def get_status_dict(self) -> dict:
        """导出状态字典（供 coordinator 读取）"""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "progress": self.progress,
            "current_action": self._get_current_action_name(),
            "last_output": self.last_output,
            "error": self.error,
            "task_id": self._current_task.get("task_id") if self._current_task else None,
        }
    
    def _get_current_action_name(self) -> str:
        """获取当前正在执行的原子动作名称"""
        actions = self.state.get("detail_actions", [])
        idx = self.state.get("current_step_idx", 0)
        if 0 <= idx < len(actions):
            return actions[idx].get("action_name", "")
        return ""
    
    def reset_for_next_task(self):
        """重置子agent状态，准备执行下一个任务"""
        self.state = self._init_state()
        self.status = "idle"
        self.progress = 0.0
        self.last_output = ""
        self.error = None
        self._current_task = None
        self._awaiting_reply = None
        self._plan_confirm_requested = None
        self._reported_macro_plan = False
        self._reported_detail_plan = False
        self._pending_resolution = None
        self._pending_param_values = None
        
        # 保存重置后的状态到 Redis
        if self.redis_mgr:
            self.redis_mgr.save_agent_state(self.agent_id, self.state)
    
    @property
    def is_done(self) -> bool:
        return self.status in ("done", "error")
    
    @property
    def has_pending_question(self) -> bool:
        return bool(self.state.get("pending_question"))
