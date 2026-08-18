"""planning_prep 节点：RAG 上下文注入（宏观/详细规划）"""
import asyncio
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage, AIMessage

from core.config import BASE_DIR
from core.ollama_utils import achat_ollama
from core.state import AgentState, Deps


def _repair_json(raw: str) -> str:
    """修复 LLM 生成的常见 JSON 格式错误，提高解析成功率"""
    # 1. 修复双冒号 "key":: → "key":（保留后随的值；若后随的是缺值的键则交给规则2补 null）
    raw = re.sub(r'":\s*:', '":', raw)
    # 2. 修复缺失值的键 "key":\n"next_key": ... → "key": null,\n"next_key": ...
    #    仅当换行后跟的是一个"键"（引号串后还有冒号）才补 null，避免误伤合法的换行值
    raw = re.sub(r'":\s*\n\s*"([^"\n]*)"\s*:', r'": null,\n"\g<1>":', raw)
    # 3. 修复 "key": } → "key": null }
    raw = re.sub(r'":\s*}', '": null }', raw)
    # 4. 修复 "key": , → "key": null,
    raw = re.sub(r'":\s*,', '": null,', raw)
    # 5. 去掉尾随逗号
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)
    # 6. 修复冒号后的杂散标点（如 "key":. "value" → "key": "value"）
    raw = re.sub(r'":\s*[.,;!](?=\s*["{\[\d])', '":', raw)
    return raw


def _close_json(text: str) -> str:
    """将截断的 JSON 文本补全为合法 JSON（尽力而为）：
    先应用常见错误修复，再补未闭合字符串、冒号后缺值、未闭合 { / ["""
    s = text.rstrip()
    s = s.rstrip("`").rstrip()
    s = s.strip()
    s = _repair_json(s)
    stack = []
    in_str = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "{":
            stack.append("o")
            i += 1
            continue
        if ch == "[":
            stack.append("a")
            i += 1
            continue
        if ch == "}":
            if stack and stack[-1] == "o":
                stack.pop()
            i += 1
            continue
        if ch == "]":
            if stack and stack[-1] == "a":
                stack.pop()
            i += 1
            continue
        i += 1
    if in_str:
        s += '"'
    s = s.rstrip().rstrip(",")
    if s.endswith(":"):
        s += " null"
    s = s.rstrip().rstrip(",")
    for c in reversed(stack):
        s += "}" if c == "o" else "]"
    return s


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON 对象（支持 markdown 代码块/裸 JSON，带常见错误修复与截断闭合）"""
    def _parse(blob: str) -> dict | None:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            try:
                return json.loads(_repair_json(blob))
            except json.JSONDecodeError:
                try:
                    return json.loads(_close_json(blob))
                except json.JSONDecodeError:
                    return None

    # 尝试 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        parsed = _parse(m.group(1))
        if parsed is not None:
            return parsed
    # 尝试裸 JSON：找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = _parse(text[start:end + 1])
        if parsed is not None:
            return parsed
    return None


def _find_macro_plan_json(messages: list) -> list | None:
    """从消息历史中找到宏观规划 JSON，提取 macro_phases 列表（支持别名 stages）"""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else ""
        # 规范化字段名：stages/stage_id/stage_name → macro_phases/phase_id/phase_name
        content = content.replace('"stages"', '"macro_phases"')
        content = content.replace('"stage_id"', '"phase_id"')
        content = content.replace('"stage_name"', '"phase_name"')
        if '"macro_phases"' not in content:
            continue
        data = _extract_json(content)
        if data and "macro_phases" in data:
            return data["macro_phases"]
    return None


def _load_detail_context(deps: Deps) -> list[str]:
    """加载详细规划所需的通用知识库上下文（schema + 工具列表 + RAG + executor 说明）"""
    parts = []

    # 详细规划 JSON schema
    atomic_schema_path = BASE_DIR / "schema" / "plan" / "atomic_action.json"
    if atomic_schema_path.is_file():
        atomic_schema_text = atomic_schema_path.read_text(encoding="utf-8")
        parts.append(f"【详细规划JSON Schema】\n```json\n{atomic_schema_text}\n```")

    # 算法工具列表（含 input_schema, output_schema, done_when, failure_policy）
    # 只列出本 agent 实际可用的算法（deps.tools 已按角色过滤），避免规划时选中其它角色的算法
    algo_json_path = BASE_DIR / "algorithms.json"
    if algo_json_path.is_file():
        algo_data = json.loads(algo_json_path.read_text(encoding="utf-8"))
        available_names = set(deps.tools.keys())
        lines = []
        for cap in algo_data.get("capabilities", []):
            name = cap["name"]
            if name not in available_names:
                continue
            desc = cap.get("description", "")
            input_schema = cap.get("input_schema", {})
            output_schema = cap.get("output_schema", {})
            done_when = cap.get("done_when", "")
            failure_policy = cap.get("failure_policy", "")
            params = ", ".join(f"{k}({v.get('type','')})" for k, v in input_schema.items())
            outputs = ", ".join(f"{k}({v.get('type','')})" if isinstance(v, dict) else str(v) for k, v in output_schema.items())
            lines.append(f"- {name}：{desc}\n"
                        f"  input_schema: {params if input_schema else '无'}\n"
                        f"  output_schema: {outputs if output_schema else '无'}\n"
                        f"  done_when: {done_when}\n"
                        f"  failure_policy: {failure_policy}")
        parts.append("【已注册算法工具列表】\n" + "\n".join(lines))

    # executor 类型说明
    executor_rule_path = BASE_DIR / "rag_docs" / "atomic_action" / "原子动作执行器类型说明.md"
    if executor_rule_path.is_file():
        parts.append(executor_rule_path.read_text(encoding="utf-8"))

    # task_planning skill
    skill_path = BASE_DIR / "skills" / "task_planning.md"
    if skill_path.is_file():
        parts.append(skill_path.read_text(encoding="utf-8"))

    return parts


async def _inject_rag_context(deps: Deps, original: str, phase_name: str, parts: list[str]) -> None:
    """检索阶段分解 RAG 并追加到 parts（同步 RAG 放入线程池，避免阻塞事件循环）"""
    try:
        query = f"{original} {phase_name}"
        context, phases = await asyncio.to_thread(deps.detail_planner.plan, query)
        if context:
            parts.append(context)
        sys.stderr.write(f"[RAG] 阶段分解检索({phase_name}): {len(context)}字\n"); sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[RAG] 阶段分解检索异常: {e}\n"); sys.stderr.flush()


def _build_phase_prompt(parts: list[str], phase: dict, original: str) -> str:
    """构造单个阶段的详细规划 prompt"""
    phase_block = (
        f"请对以下宏观阶段进行详细规划，将其拆分为原子动作列表：\n\n"
        f"阶段ID: {phase.get('phase_id', '')}\n"
        f"阶段名称: {phase.get('phase_name', '')}\n\n"
        #f"阶段目标: {phase.get('goal', '')}\n\n"
        f"原始场景：{original}"
    )
    if parts:
        return "知识库参考：\n" + "\n\n---\n\n".join(parts) + f"\n\n---\n{phase_block}"
    return phase_block


async def planning_prep(state: AgentState, deps: Deps = None) -> AgentState:
    """planning_prep 节点：注入 RAG 上下文（宏观规划检索 / 详细规划逐阶段注入）

    内部同步 RAG（chromadb + embedding HTTP）放入线程池，避免多 agent 并行时互相阻塞。
    """
    uid = state.get("user_input", "") or state.get("_last_input", "")

    # ── 宏观规划阶段（plan_generated=False）──
    if not state.get("plan_generated"):
        # 总是读取并注入 macro_plan.json schema（不依赖 RAG 分类结果）
        macro_schema_path = BASE_DIR / "schema" / "plan" / "macro_plan.json"
        macro_schema_text = ""
        if macro_schema_path.is_file():
            macro_schema_text = macro_schema_path.read_text(encoding="utf-8")

        parts = []
        if macro_schema_text:
            parts.append(f"【宏观规划JSON Schema（必须严格遵守，字段名不可更改）】\n```json\n{macro_schema_text}\n```\n警告：字段名必须与 schema 完全一致。禁止使用 stages、stage_id、step_phases 等替代名。必须使用 macro_phases、phase_id、phase_name。")

        # RAG 检索场景参考 + 约束条件
        try:
            scenario_context, scenario_phases = await asyncio.to_thread(
                deps.macro_planner.plan, uid, 3, 2
            )
            sys.stderr.write(f"[RAG] 场景检索: {len(scenario_context)}字, phases={scenario_phases}\n"); sys.stderr.flush()
        except Exception as e2:
            sys.stderr.write(f"[RAG] 场景检索异常: {e2}\n"); sys.stderr.flush()
        try:
            constraint_context, _ = await asyncio.to_thread(
                deps.constraint_planner.plan, uid, None, None, 0.4
            )
            constraint_srcs = [ln.split("来源: ", 1)[1] for ln in constraint_context.split("\n") if "来源: " in ln]
            sys.stderr.write(f"[RAG] 约束检索: {len(constraint_context)}字, 来源: {constraint_srcs}\n"); sys.stderr.flush()
        except Exception as e2:
            sys.stderr.write(f"[RAG] 约束检索异常: {e2}\n"); sys.stderr.flush()

        if scenario_context:
            # qwen3-4b-agent 在长上下文中易截断宏观 JSON：只注入最高分参考场景（第一个参考块）
            first_ref = scenario_context.split("[参考 2]", 1)[0].strip()
            parts.append(f"【参考场景】\n{first_ref}")
        if constraint_context:
            parts.append(f"【约束条件】\n{constraint_context}")

        # 只要 schema 或 RAG 有内容，就注入 user_input
        if parts:
            state["user_input"] = f"{chr(10).join(parts)}\n\n---\n{uid}"
        else:
            sys.stderr.write(f"[RAG] 无内容可注入\n"); sys.stderr.flush()

    # ── 用户驳回详细规划 → 基于现有动作+修改意见，单次重新规划 ──
    elif state.get("_replan_feedback") and state.get("macro_plan_confirmed"):
        feedback = state.get("_replan_feedback", "")
        state["_replan_feedback"] = ""  # 置空而非 pop：LangGraph 合并不会删除缺失键
        old_actions = state.get("detail_actions", [])
        state["detail_plan_done"] = False
        state["current_phase_idx"] = -1
        state["_replanning_detail"] = True
        parts = _load_detail_context(deps)
        parts.append(f"【现有详细规划动作列表】\n{json.dumps(old_actions, ensure_ascii=False)}")
        parts.append(f"【用户修改意见（重新规划时必须严格遵守）】\n{feedback}")
        state["user_input"] = (
            f"{chr(10).join(parts)}\n\n---\n"
            f"请基于以上【现有详细规划动作列表】与【用户修改意见】，重新规划该任务的详细原子动作列表，"
            f"不再拘泥于之前的宏观阶段划分。输出 JSON：{{\"actions\": [...]}}，"
            f"必须严格遵守 atomic_action JSON Schema，字段名不可更改。"
        )
        sys.stderr.write(f"[Planning] 用户修改详细规划：旧动作 {len(old_actions)} 个，意见: {feedback}\n"); sys.stderr.flush()

    # ── 宏观规划已确认 → 解析阶段 + 注入第一阶段详细规划上下文 ──
    elif state.get("plan_generated") and not state.get("macro_plan_confirmed") and state.get("_intent") == "confirm":
        state["macro_plan_confirmed"] = True
        original = state.get("original_scenario", uid)

        # 从消息历史中解析宏观规划 JSON
        macro_phases = _find_macro_plan_json(state.get("messages", []))
        if not macro_phases:
            sys.stderr.write(f"[Planning] 未找到宏观规划JSON，无法逐阶段拆解\n"); sys.stderr.flush()
            state["detail_plan_done"] = True
            return state

        state["macro_phases"] = macro_phases
        state["current_phase_idx"] = 0
        state["detail_actions"] = []
        sys.stderr.write(f"[Planning] 已解析 {len(macro_phases)} 个宏观阶段: {[p.get('phase_name','') for p in macro_phases]}\n"); sys.stderr.flush()

        # 加载通用知识库上下文 + RAG
        parts = _load_detail_context(deps)
        await _inject_rag_context(deps, original, macro_phases[0].get("phase_name", ""), parts)

        # 构造第一阶段 prompt
        state["user_input"] = _build_phase_prompt(parts, macro_phases[0], original)
        sys.stderr.write(f"[Planning] 注入阶段 1/{len(macro_phases)}: {macro_phases[0].get('phase_name','')}\n"); sys.stderr.flush()

    # ── 详细规划进行中 → 注入下一阶段上下文 ──
    elif state.get("macro_plan_confirmed") and not state.get("detail_plan_done") and state.get("current_phase_idx", -1) >= 0:
        phases = state.get("macro_phases", [])
        idx = state.get("current_phase_idx", 0)

        if idx < len(phases) - 1:
            # 还有下一阶段
            state["current_phase_idx"] = idx + 1
            original = state.get("original_scenario", uid)

            parts = _load_detail_context(deps)
            await _inject_rag_context(deps, original, phases[idx + 1].get("phase_name", ""), parts)

            state["user_input"] = _build_phase_prompt(parts, phases[idx + 1], original)
            sys.stderr.write(f"[Planning] 注入阶段 {idx + 2}/{len(phases)}: {phases[idx + 1].get('phase_name','')}\n"); sys.stderr.flush()
        else:
            # 所有阶段拆解完毕
            state["detail_plan_done"] = True
            sys.stderr.write(f"[Planning] 所有 {len(phases)} 个阶段拆解完毕\n"); sys.stderr.flush()

    return state


# ══════════════════════════════════════════════════════════════════
#  contingency 生成 pass：完整详细规划生成后，统一生成异常应对措施
#  （逐阶段规划时 LLM 无法引用尚未生成的后续阶段动作名，故必须基于完整列表生成）
# ══════════════════════════════════════════════════════════════════

def _build_contingency_prompt(actions: list) -> str:
    """构造 contingency 生成 prompt：给出完整动作列表，要求按真实 action_name 引用"""
    action_lines = "\n".join(
        f"- action_id: {a.get('action_id', '')}, action_name: {a.get('action_name', '')}, "
        f"tool_name: {a.get('tool_name', '')}, goal: {a.get('goal', '')}"
        for a in actions
    )
    example = (
        '{"contingencies": ['
        '{"action_id": "a1", "contingency": ['
        '{"condition": "在起点且任务失败", "action_name": "结束"},'
        '{"condition": "任务取消且不在起点", "action_name": "规划返航航线"}]},'
        '{"action_id": "a2", "contingency": ['
        '{"condition": "目标位置变更", "action_name": "规划去程航线"},'
        '{"condition": "航向偏移过大或遇障", "action_name": "规划去程航线"}]}'
        "]}"
    )
    return (
        "请为下面这架无人机的完整详细规划中的每个动作生成异常应对措施（contingency），"
        "contingency 在异常发生时决定切换到哪个动作节点重新执行。\n\n"
        f"【完整详细规划动作列表（action_name 必须严格按此列表引用）】\n{action_lines}\n\n"
        "【输出格式】只输出 JSON：\n"
        '{"contingencies": [{"action_id": "<动作id>", "contingency": [{"condition": "<异常情况>", "action_name": "<切换目标动作名>"}]}]}\n\n'
        "【规则】\n"
        "1. 只为可能出现异常的动作生成 contingency；无常见异常可省略该动作。\n"
        "2. action_name 必须引用【完整详细规划动作列表】中已存在的 action_name，表示异常发生时切换到该动作重新执行。\n"
        "3. 注意：action_name 字段填的是动作名称（如\"规划返航航线\"），**绝不能填 action_id**（如 action_1）；"
        "action_id（如 action_1）只出现在每项的\"action_id\"键里，表示哪个动作的应对措施。\n"
        "4. 也允许使用仅存在于状态机文件的系统/终端动作：\"结束\"（任务终止）、\"降落\"（落地）。\n"
        "5. condition 要具体，如\"目标位置变更\"、\"航向偏移过大或遇障\"、\"规划失败\"、\"扫频失败但信息不足\"等。\n\n"
        f"【参考示例】\n{example}"
    )


def _merge_contingency_results(actions: list, cons: list) -> int:
    """把 LLM 返回的 contingencies 按 action_id 合并进动作列表；返回成功合并的动作数"""
    by_id = {a.get("action_id", ""): a for a in actions}
    merged = 0
    for item in cons or []:
        if not isinstance(item, dict):
            continue
        aid = item.get("action_id", "")
        target = by_id.get(aid)
        if not target:
            continue
        lst = item.get("contingency") or []
        if not lst:
            continue
        target["contingency"] = lst
        merged += 1
    return merged


async def _run_contingency_pass(state: AgentState, deps: Deps = None) -> AgentState:
    """完整详细规划生成后，基于全部动作列表统一生成每个动作的 contingency（只跑一次）。

    LLM 调用失败/解析失败时静默降级（不生成 contingency），不影响规划主流程。
    LLM 调用经 achat_ollama 放入线程池，避免阻塞多 agent 并行的事件循环。
    """
    if state.get("_contingency_generated"):
        return state
    actions = state.get("detail_actions", [])
    if not actions:
        return state
    prompt = _build_contingency_prompt(actions)
    # 追加参考场景（phase_decmposition 场景文档中 contingency 的写法供参照）
    if deps is not None:
        try:
            original = state.get("original_scenario") or ""
            if original:
                scenario_context, _ = await asyncio.to_thread(
                    deps.detail_planner.plan, original, 2, 1
                )
                if scenario_context:
                    prompt += "\n\n【参考场景（异常应对写法参照此文档，但 action_name 仍须按上面动作列表引用）】\n" + scenario_context[:2000]
        except Exception as e:
            sys.stderr.write(f"[Planning] contingency RAG 参考检索失败: {e}\n"); sys.stderr.flush()
    content = ""
    try:
        o_msg = await achat_ollama(
            [{"role": "user", "content": prompt}], num_predict=4096
        )
        content = o_msg.get("content", "")
    except Exception as e:
        sys.stderr.write(f"[Planning] contingency 生成调用失败: {e}\n"); sys.stderr.flush()
        state["_contingency_generated"] = True
        return state
    data = _extract_json(content)
    cons = (data.get("contingencies") if data else None) or []
    if isinstance(cons, dict):
        cons = list(cons.values()) if cons else []
    # 统一覆盖：contingency 一律以本次（完整规划后）生成为准，
    # 清掉逐阶段规划阶段 LLM 可能照抄参考文档误生成的旧值（旧值引用不到真实动作名）
    for a in actions:
        a.pop("contingency", None)
    merged = _merge_contingency_results(actions, cons) if cons else 0
    # 规范化每个 contingency 目标：解析为 (action_id, 真实 action_name)，
    # 容错 LLM 把 action_id 误填进 action_name 字段（如 "action_4"）。
    # 规范化后写入 detail_actions，确认展示与输出文件均使用一致的值。
    from tools.script_writer import resolve_contingency_target
    for a in actions:
        lst = a.get("contingency") or []
        if not lst:
            continue
        canon = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            cond = item.get("condition", "")
            tid, tname = resolve_contingency_target(actions, item.get("action_name", ""))
            canon.append({"condition": cond, "action_id": tid, "action_name": tname})
        a["contingency"] = canon
    state["detail_actions"] = actions
    state["_contingency_generated"] = True
    sys.stderr.write(f"[Planning] contingency 生成完成: {merged}/{len(actions)} 个动作\n"); sys.stderr.flush()
    return state
