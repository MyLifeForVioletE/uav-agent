"""E2E driver: mirror main.py setup, drive coordinator graph with scripted confirmations, stop on completion"""
import asyncio
import sys
import uuid
from pathlib import Path

BASE = r"C:\Users\EHz\Desktop\uavAgentPlanning"
sys.path.insert(0, BASE)

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from core.config import OLLAMA_BASE, MODEL, BASE_DIR
from core.state import Deps, default_agent_state
from core.prompts import SYSTEM_PROMPT
from core.redis_manager import get_redis_manager
from agent.coordinator import build_coordinator_graph
from rag import TaskPlanner
from tools.stub_tools import merge_stub_tools


async def main():
    task = sys.argv[1] if len(sys.argv) > 1 else None
    if not task:
        print("usage: run_driver.py <task>")
        return

    redis_mgr = get_redis_manager()
    session_id = str(uuid.uuid4())[:8]

    client = MultiServerMCPClient({
        "mcp-server": {
            "command": sys.executable,
            "args": [str(BASE_DIR / "tools" / "mcp_server.py")],
            "transport": "stdio",
        }
    })

    async with client.session("mcp-server") as session:
        algo_tools = await load_mcp_tools(session)
        rag_base = str(BASE_DIR / "rag_docs")
        macro_planner = TaskPlanner(docs_dir=rag_base, collection_name="macro_rag", glob_include=["macro_examples/*.md"])
        constraint_planner = TaskPlanner(docs_dir=rag_base, collection_name="constraint_rag", glob_include=["constraint_sop/*.md"], no_chunk=True)
        detail_planner = TaskPlanner(docs_dir=rag_base, collection_name="detail_rag", glob_include=["phase_decmposition/*.md"])
        decomposer_planner = TaskPlanner(docs_dir=rag_base, collection_name="decomposer_rag", glob_include=["task_decomposition/*.md"])
        macro_planner.index_docs(); constraint_planner.index_docs(); detail_planner.index_docs(); decomposer_planner.index_docs()

        tool_map = {t.name: t for t in merge_stub_tools(algo_tools)}
        llm_plain = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_BASE)
        deps = Deps(tools=tool_map, llm_no_tools=llm_plain, macro_planner=macro_planner, constraint_planner=constraint_planner,
                    detail_planner=detail_planner, decomposer_planner=decomposer_planner)
        coordinator_graph = build_coordinator_graph(deps)

        state = default_agent_state()
        state["session_id"] = session_id
        state["messages"] = [SystemMessage(content=SYSTEM_PROMPT)]
        redis_mgr.save_session(session_id, state)
        redis_mgr.init_context(session_id)

        print(f"[DRIVER] session={session_id} task={task}", flush=True)

        inputs = [task, "确认", "确认", "确认", "确认", "确认", "确认", "确认", "确认", "确认", "确认", "确认", "确认"]
        round_no = 0
        for uid in inputs:
            round_no += 1
            state["user_input"] = uid
            if state.get("pending_question"):
                state["pending_question"] = ""
            state = await coordinator_graph.ainvoke(state)
            out = state.get("output", "")
            pq = state.get("pending_question", "")
            print(f"\n[DRIVER ROUND {round_no}] fed={uid!r}", flush=True)
            print(f"[DRIVER] out_len={len(out)} pq={pq!r} all_done={state.get('_fleet_all_done')} shared={list(state.get('shared_results', {}).keys())}", flush=True)
            if out:
                print(out, flush=True)
            if state.get("_fleet_all_done") and not pq:
                print("[DRIVER] COMPLETED", flush=True)
                break
            if not pq:
                print("[DRIVER] no pending_question -> stopping feed", flush=True)
                break
            if uid == "确认" and state.get("_dispatch_count", 0) == 0 and not out:
                print("[DRIVER] WARN: empty output", flush=True)

    redis_mgr.close()
    print("[DRIVER] DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
