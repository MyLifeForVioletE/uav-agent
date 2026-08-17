"""execute_tools 节点：执行 LLM 请求的工具调用 或 详细规划的原子动作"""
import sys
import json

from core.state import AgentState, Deps
from core.redis_manager import get_redis_manager
from tools.executor import execute_tool_calls
from tools.script_writer import write_action_step
from agent.executors import dispatch_executor


def _append_analyst_script_line(state: AgentState, action: dict, result: dict):
    """分析 agent 直接把已执行动作写入统一任务脚本（JSON，与指挥侧同一文件）。

    走 Kafka 上报给指挥再回写存在消费竞态（调度循环提前结束即丢失），
    分析 agent 单步同步执行，故直接落盘最可靠。
    """
    session_id = state.get("session_id") or "default"
    agent_id = state.get("_agent_id") or "_processor_"
    output = str(result.get("output", "") or "")[:4000]
    output_fields = result.get("output_fields") or []
    tool_inputs = action.get("tool_inputs", {}) or {}
    write_action_step(session_id, agent_id, action, tool_inputs, output, output_fields)


def _apply_post_action_update(state: AgentState, action: dict, result: dict):
    """执行后根据 post_action_update 更新 Redis 上下文

    post_action_update 在 schema/技能中定义为自然语言指令字符串（执行时由 tool_executor
    确定性写 Redis），仅兼容遗留的 {key_path: param_name} dict 格式做旧式更新。
    """
    post_action_update = action.get("post_action_update", {})
    if not post_action_update or result.get("error"):
        return
    if not isinstance(post_action_update, dict):
        return
    
    session_id = state.get("session_id", "")
    if not session_id:
        return
    
    tool_inputs = action.get("tool_inputs", {})
    updates = {}
    for key_path, param_name in post_action_update.items():
        val = tool_inputs.get(param_name)
        if val is None:
            continue
        parts = key_path.split(".")
        d = updates
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    
    if updates:
        redis_mgr = get_redis_manager()
        redis_mgr.update_context(session_id, updates)
        sys.stderr.write(f"[Execute] post_action_update: {json.dumps(updates, ensure_ascii=False)}\n")
        sys.stderr.flush()


async def execute_tools(state: AgentState, deps: Deps) -> AgentState:
    """execute_tools 节点：执行 LLM 请求的所有工具调用（参数校验 → 去重 → 执行 → 展示结果）。"""
    # 取出上一轮的 LLM 响应和工具调用列表
    response = state.pop("_last_response", None)
    tcs = state.pop("_tool_calls", [])

    # ── 详细规划执行模式 ──
    # 当详细规划已确认，且有未执行的原子动作时，按顺序执行
    if state.get("detail_plan_confirmed") and state.get("detail_actions"):
        action_idx = state.get("current_step_idx", -1)
        actions = state.get("detail_actions", [])

        if action_idx < 0:
            action_idx = 0

        if action_idx < len(actions):
            action = actions[action_idx]
            sys.stderr.write(f"[Execute] 执行原子动作 {action_idx + 1}/{len(actions)}: {action.get('action_name', '')}\n")
            sys.stderr.flush()

            # 构建执行上下文
            context = {
                "tools": deps.tools,
                "messages": state["messages"],
                "llm": deps.llm_no_tools,
                "state": state,
                "agent_id": state.get("_agent_id") or "_processor_",
            }

            # 分发执行
            result = await dispatch_executor(action, context)

            # 缺少参数：不推进索引、不置错误，交由 SubAgent 缺参请求路径处理
            if result.get("missing_params"):
                missing = "、".join(p.get("name", p) for p in result["missing_params"])
                state["output"] = f"动作 {action.get('action_name', '')} 缺少参数: {missing}，将请求指挥解决"
                state["pending_question"] = ""
                sys.stderr.write(f"[Execute] 缺参: {missing}\n"); sys.stderr.flush()
                return state

            # 执行成功后应用 post_action_update
            if action.get("post_action_update"):
                _apply_post_action_update(state, action, result)

            # 更新状态
            state["current_step_idx"] = action_idx + 1
            state["output"] = result.get("output", "")

            if result.get("error"):
                sys.stderr.write(f"[Execute] 执行失败: {result['error']}\n")
                sys.stderr.flush()
                state["pending_question"] = f"执行失败: {result['error']}"
            elif result.get("pending"):
                # 需要等待用户输入
                pass
            else:
                # 执行成功，如果还有下一个动作，继续
                if action_idx + 1 < len(actions):
                    # 还有下一个动作，继续执行
                    pass
                else:
                    # 所有动作执行完毕
                    state["output"] = f"全部 {len(actions)} 个原子动作执行完毕"
                    state["pending_question"] = "详细规划执行完成"

            return state

    # ── LLM 工具调用模式 ──
    if not response or not tcs:
        return state

    state["output"] = ""  # 清空输出，工具执行结果通过 _print 直接输出

    # 分析 agent：每个工具调用走标准参数补齐管线（Redis 上下文 → knowledge 目录 → LLM 选择），
    # 与 UAV 执行路径一致，而非直接用 LLM 猜测的参数执行
    if state.get("_analyst_mode"):
        from langchain_core.messages import ToolMessage
        # 分析 agent 步骤序号：每次工具调用递增，脚本中连续展示该 agent 的分析流程
        analyst_step = state.get("_analyst_step", 0) + 1
        for tc in tcs:
            action = {
                "executor": "tool",
                "tool_name": tc["name"],
                "tool_inputs": tc["args"],
                "goal": state.get("original_scenario", ""),
            }
            context = {
                "tools": deps.tools,
                "messages": state["messages"],
                "llm": deps.llm_no_tools,
                "state": state,
                "agent_id": state.get("_agent_id") or "_processor_",
            }
            try:
                result = await dispatch_executor(action, context)
            except Exception as e:
                result = {"success": False, "output": "", "error": f"分析工具执行异常: {e}"}
            if result.get("tool_inputs"):
                action["tool_inputs"] = result["tool_inputs"]
            # 缺参：挂起，由 SubAgent 上报指挥解析（跨 agent 依赖 / 推导 / 用户提供）
            if result.get("missing_params"):
                missing = "、".join(p.get("name", p) for p in result["missing_params"])
                state["_analyst_param_pending"] = {"action": action, "missing": result["missing_params"]}
                state["pending_question"] = ""
                sys.stderr.write(f"[Execute] 分析缺参挂起: {missing}\n")
                sys.stderr.flush()
                state["_analyst_step"] = analyst_step
                return state
            # 自身工具能产出缺失字段 → 先执行产出动作，再执行本动作
            if result.get("self_produce"):
                guard = 0
                while result.get("self_produce") and guard < 5:
                    guard += 1
                    producer = result["self_produce"].get("action") or result["self_produce"]
                    try:
                        presult = await dispatch_executor(producer, context)
                    except Exception as e:
                        presult = {"success": False, "output": "", "error": f"产出工具执行异常: {e}"}
                    if presult.get("tool_inputs"):
                        producer["tool_inputs"] = presult["tool_inputs"]
                    if presult.get("missing_params"):
                        missing = "、".join(p.get("name", p) for p in presult["missing_params"])
                        state["_analyst_param_pending"] = {"action": action, "missing": presult["missing_params"]}
                        state["pending_question"] = ""
                        sys.stderr.write(f"[Execute] 产出工具缺参挂起: {missing}\n")
                        sys.stderr.flush()
                        state["_analyst_step"] = analyst_step
                        return state
                    _append_analyst_script_line(state, producer, presult)
                    result = await dispatch_executor(action, context)
                    if result.get("tool_inputs"):
                        action["tool_inputs"] = result["tool_inputs"]
                if result.get("missing_params"):
                    missing = "、".join(p.get("name", p) for p in result["missing_params"])
                    state["_analyst_param_pending"] = {"action": action, "missing": result["missing_params"]}
                    state["pending_question"] = ""
                    sys.stderr.write(f"[Execute] 产出后仍缺参挂起: {missing}\n")
                    sys.stderr.flush()
                    state["_analyst_step"] = analyst_step
                    return state
            result_text = result.get("output", "") or result.get("error", "")
            state["output"] = result_text
            state["messages"].append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
            # 分析 agent 步骤：直接写统一脚本（确定性，不依赖指挥回传）
            _append_analyst_script_line(state, action, result)
            analyst_step += 1
            if result.get("error"):
                state["pending_question"] = result_text
                return state
        state["_analyst_step"] = analyst_step - 1
        return state

    # 委托 tool_executor 执行具体的工具调用流程
    should_break, lct, lfp, plan_gen, _plan_steps = await execute_tool_calls(
        response, deps.tools, state["messages"], deps.llm_no_tools,
        state.get("calc_type"), state.get("file_path"), state.get("plan_generated", False),
        skip_summary=True, quiet=False,
    )

    # 更新状态中的计算类型和文件路径
    state["calc_type"] = lct
    state["file_path"] = lfp

    # should_break 表示参数校验失败，需要等待用户重新输入
    if should_break:
        state["pending_question"] = " "
        return state

    return state
