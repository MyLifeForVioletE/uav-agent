"""planning_prep 节点：RAG 上下文注入（宏观/详细规划）"""
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage, AIMessage

from core.config import OLLAMA_BASE, MODEL, BASE_DIR
from core.state import AgentState, Deps


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON 对象（支持 markdown 代码块和裸 JSON）"""
    # 尝试 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试裸 JSON：找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
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


def _inject_rag_context(deps: Deps, original: str, phase_name: str, parts: list[str]) -> None:
    """检索阶段分解 RAG 并追加到 parts"""
    try:
        query = f"{original} {phase_name}"
        context, phases = deps.detail_planner.plan(query)
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


def planning_prep(state: AgentState, deps: Deps = None) -> AgentState:
    """planning_prep 节点：注入 RAG 上下文（宏观规划检索 / 详细规划逐阶段注入）"""
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
            scenario_context, scenario_phases = deps.macro_planner.plan(uid, retrieve_k=3, rerank_n=2)
            sys.stderr.write(f"[RAG] 场景检索: {len(scenario_context)}字, phases={scenario_phases}\n"); sys.stderr.flush()
        except Exception as e2:
            sys.stderr.write(f"[RAG] 场景检索异常: {e2}\n"); sys.stderr.flush()
        try:
            constraint_context, _ = deps.constraint_planner.plan(uid, score_threshold=0.4)
            constraint_srcs = [ln.split("来源: ", 1)[1] for ln in constraint_context.split("\n") if "来源: " in ln]
            sys.stderr.write(f"[RAG] 约束检索: {len(constraint_context)}字, 来源: {constraint_srcs}\n"); sys.stderr.flush()
        except Exception as e2:
            sys.stderr.write(f"[RAG] 约束检索异常: {e2}\n"); sys.stderr.flush()

        if scenario_context:
            parts.append(f"【参考场景】\n{scenario_context}")
        if constraint_context:
            parts.append(f"【约束条件】\n{constraint_context}")

        # 只要 schema 或 RAG 有内容，就注入 user_input
        if parts:
            state["user_input"] = f"{chr(10).join(parts)}\n\n---\n{uid}"
        else:
            sys.stderr.write(f"[RAG] 无内容可注入\n"); sys.stderr.flush()

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
        _inject_rag_context(deps, original, macro_phases[0].get("phase_name", ""), parts)

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
            _inject_rag_context(deps, original, phases[idx + 1].get("phase_name", ""), parts)

            state["user_input"] = _build_phase_prompt(parts, phases[idx + 1], original)
            sys.stderr.write(f"[Planning] 注入阶段 {idx + 2}/{len(phases)}: {phases[idx + 1].get('phase_name','')}\n"); sys.stderr.flush()
        else:
            # 所有阶段拆解完毕
            state["detail_plan_done"] = True
            sys.stderr.write(f"[Planning] 所有 {len(phases)} 个阶段拆解完毕\n"); sys.stderr.flush()

    return state
