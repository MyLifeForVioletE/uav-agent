"""router 节点：LLM 意图分类与路由分派"""
import sys

import requests
from langchain_core.messages import SystemMessage

from core.config import OLLAMA_BASE, MODEL
from core.state import AgentState, Deps
from core.prompts import load_skill


def router_node(state: AgentState, deps: Deps = None) -> AgentState:
    """router 节点：LLM 纯意图分类，按管道分派到不同业务节点"""
    uid = state.get("user_input", "")
    uid = uid or state.get("_last_input", "")
    if not uid:
        state["_intent"] = "chat"
        return state

    # 已生成宏观规划但未确认 → 检测确认/修改意图
    if state.get("plan_generated") and not state.get("macro_plan_confirmed"):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "判断用户是否在确认、同意之前给出的规划方案。只回答 CONFIRM 或 OTHER。"},
                        {"role": "user", "content": uid},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=30,
            )
            is_confirm = resp.json()["message"]["content"].strip().upper() == "CONFIRM"
            sys.stderr.write(f"[Router] confirm={is_confirm}\n"); sys.stderr.flush()
            state["_intent"] = "confirm" if is_confirm else "planning"
        except Exception as e:
            sys.stderr.write(f"[Router] confirm error: {e}\n"); sys.stderr.flush()
            state["_intent"] = "planning"
        return state

    # 详细规划已生成但未确认 → 检测确认/修改意图
    if state.get("detail_plan_done") and not state.get("detail_plan_confirmed"):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "判断用户是否在确认、同意之前给出的详细规划方案。只回答 CONFIRM 或 OTHER。"},
                        {"role": "user", "content": uid},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=30,
            )
            is_confirm = resp.json()["message"]["content"].strip().upper() == "CONFIRM"
            sys.stderr.write(f"[Router] detail_confirm={is_confirm}\n"); sys.stderr.flush()
            if is_confirm:
                state["detail_plan_confirmed"] = True
                state["_intent"] = "chat"
            else:
                state["_intent"] = "planning"
        except Exception as e:
            sys.stderr.write(f"[Router] detail confirm error: {e}\n"); sys.stderr.flush()
            state["_intent"] = "planning"
        return state

    # 用户驳回详细规划 → 走 planning_prep 重规划路径（注入旧动作+修改意见）
    if state.get("_replan_feedback") and state.get("macro_plan_confirmed"):
        state["_intent"] = "planning"
        return state

    # 宏观刚确认、详细规划尚未生成 → 去详细规划
    if state.get("macro_plan_confirmed") and not state.get("detail_plan_done"):
        state["_intent"] = "detail_planning"
        return state

    # 首次进入：子agent 任务已由指挥分配，直接进入任务规划
    if not state.get("plan_generated"):
        state["_intent"] = "planning"
        sys.stderr.write("[Router] subtask -> intent=planning\n"); sys.stderr.flush()

        injected = state.get("_skill_injected") or set()

        # planning 意图：注入任务规划 skill
        if "task_planning" not in injected:
            skill_content = load_skill("task_planning.md")
            if skill_content:
                state["messages"].append(SystemMessage(content=skill_content))
                injected.add("task_planning")
                sys.stderr.write(f"[Skill] 已注入 task_planning.md\n"); sys.stderr.flush()

        state["_skill_injected"] = injected
        return state

    state["_intent"] = "chat"
    return state
