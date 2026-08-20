"""call_llm 节点：调用 Ollama API 并处理响应"""
import asyncio
import json
import sys

from langchain_core.messages import HumanMessage, AIMessage

from core.state import AgentState, Deps
from core.ollama_utils import messages_to_ollama, tools_to_ollama, achat_ollama
from agent.nodes.planning import _extract_json


async def call_llm(state: AgentState, deps: Deps) -> AgentState:
    """call_llm 节点：直接调 Ollama API（带 tools），将回复存入消息列表。"""
    uid = state.get("user_input")
    if uid:
        state["messages"].append(HumanMessage(content=uid))
    state["user_input"] = None
    messages = state["messages"]
    o_messages = messages_to_ollama(messages)
    macro_phase = ("task_planning" in (state.get("_skill_injected") or set()) and not state.get("plan_generated"))
    detail_phase = (state.get("macro_plan_confirmed") and not state.get("detail_plan_done")
                    and state.get("current_phase_idx", -1) >= 0)
    # 宏观/详细规划阶段禁止工具调用，强制 LLM 只输出文本/JSON
    if macro_phase or \
       (state.get("macro_plan_confirmed") and not state.get("detail_plan_done")) or \
       (state.get("detail_plan_done") and not state.get("detail_plan_confirmed")):
        o_tools = []
    else:
        o_tools = tools_to_ollama(deps.tools)

    # 宏观规划响应含 macro_phases 标记但 JSON 无效/被截断 → 重试（模型偶发中途截断，重试通常可恢复）
    MAX_PLAN_ATTEMPTS = 3
    attempt = 0
    content = ""
    o_tcs = []
    while True:
        attempt += 1
        try:
            o_msg = await achat_ollama(o_messages, tools=o_tools or None)
            content = o_msg.get("content", "")
            o_tcs = o_msg.get("tool_calls", [])
        except Exception as e:
            detail = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = e.response.text[:500]
                except Exception:
                    pass
            sys.stderr.write(f"[LLM] Ollama 调用失败(第{attempt}次): {e}\n{detail}\n"); sys.stderr.flush()
            # 瞬时故障（超时/连接断开等）重试：Ollama 串行处理多 agent 请求，
            # 排队等待会把单次调用拖过超时阈值；重试时队列通常已排空，可成功。
            # 重试上限与下方 JSON 无效重试一致（MAX_PLAN_ATTEMPTS）。
            if attempt < MAX_PLAN_ATTEMPTS:
                sys.stderr.write(f"[LLM] 调用失败，{2.0 * attempt}s 后重试(第{attempt + 1}/{MAX_PLAN_ATTEMPTS}次)...\n"); sys.stderr.flush()
                await asyncio.sleep(2.0 * attempt)
                continue
            aim = AIMessage(content=f"（调用 Ollama 失败：{e}\n{detail}）")
            state["messages"].append(aim)
            state["output"] = aim.content
            state["_tool_calls"] = None
            state["_last_response"] = aim
            state["pending_question"] = aim.content
            return state

        if macro_phase:
            _norm = content.replace('"stages"', '"macro_phases"').replace('"stage_id"', '"phase_id"').replace('"stage_name"', '"phase_name"')
            has_marker = '"macro_phases"' in content or '"macro_phases"' in _norm
            if has_marker:
                parsed = _extract_json(_norm if '"macro_phases"' in _norm else content)
                if parsed and (parsed.get("macro_phases") or parsed.get("stages")):
                    break
                if attempt < MAX_PLAN_ATTEMPTS:
                    sys.stderr.write(f"[LLM] 宏观规划JSON无效或截断(第{attempt}次)，重试\n"); sys.stderr.flush()
                    continue
                sys.stderr.write(f"[LLM] 宏观规划JSON重试{MAX_PLAN_ATTEMPTS}次仍无效\n"); sys.stderr.flush()
                break
        elif detail_phase and "{" in content:
            # 详细规划阶段：内容像 JSON 但解析失败（截断/格式错误）→ 重试，避免该阶段动作丢失
            if _extract_json(content) is None:
                if attempt < MAX_PLAN_ATTEMPTS:
                    sys.stderr.write(f"[LLM] 详细规划JSON无效或截断(第{attempt}次)，重试\n"); sys.stderr.flush()
                    continue
                sys.stderr.write(f"[LLM] 详细规划JSON重试{MAX_PLAN_ATTEMPTS}次仍无效\n"); sys.stderr.flush()
        break

    # 将 Ollama tool_calls 转为 LangChain 格式
    tool_calls = []
    for tc in o_tcs:
        func = tc.get("function", {})
        name = func.get("name", "")
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args
        tool_calls.append({
            "name": name,
            "args": args,
            "id": f"call_{name}_{len(tool_calls)}",
        })

    # 构造 AIMessage
    additional_kwargs = {}
    if o_tcs:
        additional_kwargs["tool_calls"] = o_tcs
    aim_kw = {"content": content, "additional_kwargs": additional_kwargs}
    if tool_calls:
        aim_kw["tool_calls"] = tool_calls
    aim = AIMessage(**aim_kw)
    state["messages"].append(aim)
    state["output"] = content or ""
    state["_tool_calls"] = tool_calls if tool_calls else None
    state["_last_response"] = aim

    if not tool_calls and content:
        state["pending_question"] = content

    # 检测 LLM 是否输出了宏观规划（JSON 格式含 macro_phases 或其别名），标记 plan_generated
    # 规范化字段名：LLM 可能输出 stages/stage_id/stage_name 而非 macro_phases/phase_id/phase_name
    _normalized = content
    _normalized = _normalized.replace('"stages"', '"macro_phases"')
    _normalized = _normalized.replace('"stage_id"', '"phase_id"')
    _normalized = _normalized.replace('"stage_name"', '"phase_name"')
    if '"macro_phases"' in content:
        state["plan_generated"] = True
    elif '"macro_phases"' in _normalized:
        # 更新 content/output/pending_question 使用规范化后的版本
        content = _normalized
        state["output"] = content
        state["pending_question"] = content
        state["plan_generated"] = True
        # 修正消息历史中存储的 AIMessage content
        if state["messages"]:
            last = state["messages"][-1]
            if hasattr(last, "content") and isinstance(last.content, str) and '"stages"' in last.content:
                last.content = _normalized

    # 详细规划进行中：解析当前阶段的原子动作并累积
    if state.get("macro_plan_confirmed") and not state.get("detail_plan_done") and state.get("current_phase_idx", -1) >= 0:
        data = _extract_json(content)
        if data:
            actions = data.get("actions") or data.get("atomic_actions") or []
            if actions:
                state["detail_actions"] = state.get("detail_actions", []) + actions
                sys.stderr.write(f"[LLM] 累积 {len(actions)} 个原子动作，总计 {len(state['detail_actions'])} 个\n"); sys.stderr.flush()
        # 检查是否所有阶段都已拆解完毕
        phases = state.get("macro_phases", [])
        idx = state.get("current_phase_idx", 0)
        if phases and idx >= len(phases) - 1:
            # 归一化 action_id：逐阶段规划时 LLM 可能每阶段重新编号（重复 action_1/action_2），
            # 造成本 agent 内编号冲突、contingency 引用错乱；按本 agent 自己的动作列表
            # 从 1 开始重排（action_1..action_N），不跨 agent 累加。
            for _i, _a in enumerate(state.get("detail_actions", []), 1):
                _a["action_id"] = f"action_{_i}"
            state["detail_plan_done"] = True
            state["detail_plan_confirmed"] = True
            # 完整详细规划生成后：基于全部动作列表统一生成各动作的异常应对措施（contingency）
            from agent.nodes.planning import _run_contingency_pass
            await _run_contingency_pass(state, deps)
            # 整合输出全部阶段的原子动作
            all_actions = state.get("detail_actions", [])
            consolidated = {"actions": all_actions}
            consolidated_text = json.dumps(consolidated, ensure_ascii=False, indent=4)
            summary = f"全部 {len(phases)} 个阶段拆解完毕，共 {len(all_actions)} 个原子动作：\n\n```json\n{consolidated_text}\n```"
            state["output"] = summary
            state["pending_question"] = ""
            sys.stderr.write(f"[LLM] 所有 {len(phases)} 个阶段拆解完毕，总计 {len(all_actions)} 个原子动作\n"); sys.stderr.flush()

    # 用户修改意见后的单次重规划：解析新动作列表
    if state.get("_replanning_detail"):
        state["_replanning_detail"] = False
        data = _extract_json(content)
        actions = (data.get("actions") if data else None) or (data.get("atomic_actions") if data else None) or []
        if actions:
            state["detail_actions"] = actions
            # 归一化 action_id：重规划替换全部动作，同样按本 agent 从 1 开始重排
            for _i, _a in enumerate(actions, 1):
                _a["action_id"] = f"action_{_i}"
            state["detail_plan_done"] = True
            state["detail_plan_confirmed"] = True
            # 重规划替换了全部动作：重置标记并基于新计划重新生成 contingency
            state["_contingency_generated"] = False
            from agent.nodes.planning import _run_contingency_pass
            await _run_contingency_pass(state, deps)
            consolidated = json.dumps({"actions": data["actions"]}, ensure_ascii=False, indent=4)
            summary = f"已根据修改意见重新规划，共 {len(data['actions'])} 个原子动作：\n\n```json\n{consolidated}\n```"
            state["output"] = summary
            state["pending_question"] = ""
            sys.stderr.write(f"[LLM] 修改意见重规划完成，共 {len(data['actions'])} 个原子动作\n"); sys.stderr.flush()
        else:
            # 重规划失败：沿用原方案，避免挂起
            sys.stderr.write("[LLM] 修改意见重规划响应未包含有效 actions，沿用原方案\n"); sys.stderr.flush()
            state["detail_plan_done"] = True
            state["detail_plan_confirmed"] = True
            state["pending_question"] = ""

    return state
