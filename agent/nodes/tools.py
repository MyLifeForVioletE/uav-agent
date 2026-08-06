"""execute_tools 节点：执行 LLM 请求的工具调用 或 详细规划的原子动作"""
import sys
import json

from core.state import AgentState, Deps
from core.redis_manager import get_redis_manager
from tools.executor import execute_tool_calls
from agent.executors import dispatch_executor


def _apply_post_action_update(state: AgentState, action: dict, result: dict):
    """执行后根据 post_action_update 更新 Redis 上下文"""
    post_action_update = action.get("post_action_update", {})
    if not post_action_update or result.get("error"):
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
            }
            try:
                result = await dispatch_executor(action, context)
            except Exception as e:
                result = {"success": False, "output": "", "error": f"分析工具执行异常: {e}"}
            result_text = result.get("output", "") or result.get("error", "")
            state["output"] = result_text
            state["messages"].append(ToolMessage(content=result_text, tool_call_id=tc["id"]))
            if result.get("error"):
                state["pending_question"] = result_text
                return state
        return state

    # 委托 tool_executor 执行具体的工具调用流程
    should_break, lct, lfp, plan_gen, plan_steps = await execute_tool_calls(
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
