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
from agent.graph import build_graph
from agent.analyst_graph import build_analyst_graph


class SendMessageInput(BaseModel):
    """send_message 工具参数"""
    to: str = Field(description="接收方 agent ID：指挥 agent 为 _coordinator_，无人机为 UAV_1 等，信息处理 agent 为 _info_processor_")
    msg_type: str = Field(description="消息类型：report=向指挥上报状态/结果，request=向指挥请示问题并等待回复，instruction=向其它 agent 下发指令")
    content: str = Field(description="消息内容（用自然中文）")
    await_reply: bool = Field(default=False, description="msg_type=request 时设为 true，表示要等待指挥回复后再继续")


class SubAgent:
    """单个 Agent 实例，封装独立的 graph 和 state
    
    支持两种模式：
    - "uav"：无人机子 agent，生成 UAV 专属的任务描述
    - "coordinator"：分析决策协调 agent，生成分析类任务描述
    - "info_processor"：信息处理 agent，生成对无人机采集数据的分析处理类任务描述
    """
    
    def __init__(self, agent_id: str, deps, initial_config: dict = None, mode: str = "uav", redis_mgr: RedisManager = None, session_id: str = "default"):
        self.agent_id = agent_id
        self.mode = mode
        self.deps = self._filter_deps(deps)
        self.redis_mgr = redis_mgr
        self._session_id = session_id
        # 分析 agent 用 3 节点极简图；其余复用 6 节点单机图
        self.graph = build_analyst_graph(self.deps) if mode == "info_processor" else build_graph(self.deps)
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

    def _filter_deps(self, deps) -> Deps:
        """按 mode 过滤可用工具表，其余依赖（规划器/LLM）保持不变"""
        tools = filter_tools_for_role(deps.tools, self.mode)
        # 子 agent（UAV / 信息处理）额外提供 send_message 通信工具
        if self.mode in ("uav", "info_processor"):
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
        is_analyst = self.mode == "info_processor"
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
            if mtype in ("instruction", "reply"):
                if mtype == "reply" and self._awaiting_reply:
                    if m.get("correlation_id") == self._awaiting_reply:
                        self._awaiting_reply = None
                        self.state["messages"].append(SystemMessage(content=f"[指挥回复] {content}"))
                        sys.stderr.write(f"[SubAgent {self.agent_id}] 收到指挥回复，解除挂起\n")
                        sys.stderr.flush()
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
        if self.mode == "coordinator":
            task_desc = (
                f"你是多无人机协同任务的分析决策协调者。\n"
                f"请对以下分析决策类子任务进行规划：\n"
                f"- 任务名称：{task.get('task_name', '')}\n"
                f"- 任务目标：{task.get('goal', '')}\n"
                f"{context_block}"
            )
        elif self.mode == "info_processor":
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
    
    async def run_step(self):
        """执行一步（调用 graph.ainvoke 或直接执行已保存的原子动作）"""
        # 先拉取收件箱：注入指挥指令；收到匹配的回复则解除同步请示挂起
        await self._drain_inbox()

        # 同步请示挂起：请求已发出但指挥尚未回复 → 本 tick 不推进
        if self._awaiting_reply:
            return

        # 分析 agent：直接跑一次分析图即完成（无规划/无用户确认）
        if self.mode == "info_processor":
            try:
                self.state = await self.graph.ainvoke(self.state, {"recursion_limit": 50})
                self.last_output = self.state.get("output", "")
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
                and self.state.get("plan_generated") and self.state.get("macro_plan_confirmed")
                and not self.state.get("pending_question")):
            await self._execute_saved_action_step()
            return

        was_pg = self.state.get("plan_generated", False)

        try:
            self.state = await self.graph.ainvoke(self.state, {"recursion_limit": 50})
            self.last_output = self.state.get("output", "")
            self._sync_status()
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

        if idx >= len(actions):
            self.status = "done"
            self.progress = 1.0
            return

        action = dict(actions[idx])
        from agent.executors import dispatch_executor

        # 执行阶段：清空 tool_inputs 中的猜测值，让 tool_executor 走参数解析流程
        if action.get("executor") == "tool" and "tool_inputs" in action:
            action["tool_inputs"] = {}

        context = {
            "tools": self.deps.tools,
            "messages": self.state["messages"],
            "llm": self.deps.llm_no_tools,
            "state": self.state,
        }

        sys.stderr.write(f"[SubAgent {self.agent_id}] 执行步骤 {idx+1}/{len(actions)}: {action.get('action_name', '')}\n")
        sys.stderr.flush()

        try:
            result = await dispatch_executor(action, context)
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self.last_output = f"执行失败: {e}"
            return

        self.state["current_step_idx"] = idx + 1
        self.last_output = result.get("output", "")
        self.state["output"] = self.last_output
        
        # 执行成功后：位置更新仍由 UAV 直写 Redis；算法结果（扫频等）发给指挥，由指挥提取并写 Redis
        if result.get("success"):
            if action.get("post_action_update") and self.redis_mgr:
                await self._apply_post_action_update(action, result)
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
        else:
            self._sync_status()

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
        """执行后根据 post_action_update 更新 Redis 上下文"""
        post_action_update = action.get("post_action_update", "")
        if not post_action_update or not self.redis_mgr:
            return
        
        import json
        import requests
        
        from core.config import OLLAMA_BASE, MODEL
        
        tool_inputs = action.get("tool_inputs", {})
        goal = action.get("goal", "")
        
        current_context = self.redis_mgr.get_context(self._session_id) or {}
        
        # 获取工具的 input_schema 参数定义，帮助 LLM 理解参数含义
        tool_params = ""
        tool_name = action.get("tool_name", "")
        if tool_name:
            try:
                algo_path = __import__("core.config", fromlist=["BASE_DIR"]).BASE_DIR / "algorithms.json"
                with open(algo_path, encoding="utf-8") as f:
                    algos = {a["name"]: a for a in __import__("json").load(f).get("capabilities", [])}
                schema = algos.get(tool_name, {}).get("input_schema", {})
                if schema:
                    lines = ["【工具参数定义】"]
                    for pname, pinfo in schema.items():
                        lines.append(f"- {pname}: {pinfo.get('description', pname)}")
                    tool_params = "\n".join(lines) + "\n\n"
            except Exception:
                pass
        
        prompt_parts = [
            "你是一个任务执行系统的上下文更新模块。\n",
            "根据以下信息，确定执行完当前动作后需要更新 Redis 中的哪些字段。\n\n",
            f"【当前无人机】{self.agent_id}\n\n",
            f"【动作目标】{goal}\n\n",
            f"【需要执行的上下文更新】{post_action_update}\n\n",
            f"【已解析的工具参数】{json.dumps(tool_inputs, ensure_ascii=False)}\n\n",
            tool_params,
            f"【当前 Redis 上下文】{json.dumps(current_context, ensure_ascii=False)}\n\n",
            "【上下文字段规范（必须遵守）】\n"
            "1. 模板中 uavs/targets/base 的位置字段只有 Longitude 和 Latitude，不存在 position/x/y 字段。\n"
            "2. 更新位置时必须写入 Longitude 和 Latitude 两个字段，例如 uavs.0.Longitude、uavs.0.Latitude。\n"
            "3. 坐标值若是 \"70,15.01\" 这类用逗号分隔的字符串，必须拆成两条分别写入 Longitude（70）和 Latitude（15.01）。\n"
            "4. 禁止输出 position 字段；输出前检查一遍，确保没有生成模板之外的字段。\n\n"
            "分析思路：\n"
            "1. 先理解 post_action_update 指令要做什么（如「将无人机当前位置更新为目标位置坐标」）\n"
            "2. 在 tool_inputs 中找到对应的参数值（「目标位置」→destination 的值）\n"
            "3. 在 Redis 上下文中找到要更新的路径（「无人机当前位置」→uavs.0.Longitude 与 uavs.0.Latitude）\n"
            "4. 注意区分 tool_inputs 中哪些是输入（如 startPosition），哪些是动作结果（如 destination 是规划后的新位置）\n\n"
            "请输出一个 JSON 对象，表示要更新到 Redis 的字段映射。\n"
            'key 为点号分隔的路径（如 "uavs.0.Longitude" 或 "uavs.UAV_1.Latitude"），\n'
            "数组既可用数字索引也可用 id 值来定位元素。\n"
            "value 为具体的数值。"
        ]
        prompt = "".join(prompt_parts)
        
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": 4096},
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
        except Exception as e:
            sys.stderr.write(f"[SubAgent {self.agent_id}] post_action_update 调用失败: {e}\n")
            return
        
        try:
            updates = json.loads(content)
        except json.JSONDecodeError:
            sys.stderr.write(f"[SubAgent {self.agent_id}] post_action_update 解析失败: {content}\n")
            return
        
        if not updates or not isinstance(updates, dict):
            return
        
        nested = {}
        for key_path, val in updates.items():
            parts = key_path.split(".")
            d = nested
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
        
        self.redis_mgr.update_context(self._session_id, nested)
        sys.stderr.write(f"[SubAgent {self.agent_id}] post_action_update: {json.dumps(nested, ensure_ascii=False)}\n")
        sys.stderr.flush()

    async def _update_context_from_result(self, action: dict, result: dict):
        """（已废弃：算法结果改走指挥 agent 提取并写 Redis，见 fleet_manager._update_context_from_agent_result）"""
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
    
    def inject_context(self, context_msg: str):
        """向子agent注入上下文消息（用于跨机数据共享）"""
        self.state["messages"].append(SystemMessage(content=context_msg))
    
    def reset_for_next_task(self):
        """重置子agent状态，准备执行下一个任务"""
        self.state = self._init_state()
        self.status = "idle"
        self.progress = 0.0
        self.last_output = ""
        self.error = None
        self._current_task = None
        self._awaiting_reply = None
        
        # 保存重置后的状态到 Redis
        if self.redis_mgr:
            self.redis_mgr.save_agent_state(self.agent_id, self.state)
    
    @property
    def is_done(self) -> bool:
        return self.status in ("done", "error")
    
    @property
    def has_pending_question(self) -> bool:
        return bool(self.state.get("pending_question"))
