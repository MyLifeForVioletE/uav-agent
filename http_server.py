"""HTTP 服务：把多机协同规划流程封装为 REST API，供前端调用（无用户确认自动模式）

启动：
    python -m uvicorn http_server:app --host 0.0.0.0 --port 8000

接口：
    POST /api/run  提交任务描述 {scenario: str}，同步阻塞执行，返回两个脚本下载链接
    GET  /api/files/{sid}/{kind}  下载脚本文件（kind=actions | contingency）
    GET  /api/health  健康检查

执行全程自动确认（不向用户提问），最终产物为两个脚本文件：
    output/actions_script_{sid}.json      动作脚本
    output/contingency_plan_{sid}.json    异常应对文件
"""
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from core.config import OLLAMA_BASE, MODEL, BASE_DIR
from core.prompts import SYSTEM_PROMPT
from core.redis_manager import get_redis_manager
from core.state import Deps, default_agent_state
from agent.coordinator import build_coordinator_graph
from rag import TaskPlanner
from tools.script_writer import contingency_plan_path, script_path
from tools.stub_tools import merge_stub_tools

AUTO_CONFIRM_WORD = "确认"
MAX_DRIVE_ROUNDS = 500
RECURSION_LIMIT = 2000


class RunRequest(BaseModel):
    scenario: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_mgr = get_redis_manager()

    client = MultiServerMCPClient({
        "mcp-server": {
            "command": sys.executable,
            "args": [str(BASE_DIR / "tools" / "mcp_server.py")],
            "transport": "stdio",
        }
    })

    async with client.session("mcp-server") as session:
        algo_tools = await load_mcp_tools(session)
        sys.stderr.write(f"[HTTP] MCP 真实工具 {len(algo_tools)} 个\n")
        sys.stderr.flush()

        rag_base = str(BASE_DIR / "rag_docs")
        macro_planner = TaskPlanner(
            docs_dir=rag_base, collection_name="macro_rag",
            glob_include=["macro_examples/*.md"],
        )
        constraint_planner = TaskPlanner(
            docs_dir=rag_base, collection_name="constraint_rag",
            glob_include=["constraint_sop/*.md"], no_chunk=True,
        )
        detail_planner = TaskPlanner(
            docs_dir=rag_base, collection_name="detail_rag",
            glob_include=["phase_decmposition/*.md"],
        )
        decomposer_planner = TaskPlanner(
            docs_dir=rag_base, collection_name="decomposer_rag",
            glob_include=["task_decomposition/*.md"],
        )
        for p in (macro_planner, constraint_planner, detail_planner, decomposer_planner):
            p.index_docs()

        all_tools = merge_stub_tools(algo_tools)
        tool_map = {t.name: t for t in all_tools}
        llm_plain = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_BASE, request_timeout=60)
        deps = Deps(
            tools=tool_map,
            llm_no_tools=llm_plain,
            macro_planner=macro_planner,
            constraint_planner=constraint_planner,
            detail_planner=detail_planner,
            decomposer_planner=decomposer_planner,
        )
        graph = build_coordinator_graph(deps)

        app.state.redis_mgr = redis_mgr
        app.state.deps = deps
        app.state.graph = graph
        sys.stderr.write("[HTTP] 服务就绪\n")
        sys.stderr.flush()
        yield

    redis_mgr.close()


app = FastAPI(title="UAV Agent Planning HTTP API", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "ready": hasattr(app.state, "graph")}


async def _run_task_auto(graph, redis_mgr, scenario: str) -> str:
    """无确认自动执行：创建会话 → 注入场景 → 自动注入确认驱动图跑完 → 返回 session_id。

    完成信号：result_agg 节点会把 session_id 轮换为 <base>-t<seq>（任务边界），
    因此 ainvoke 返回后 session_id 与初始值不一致即代表整个任务已执行完毕。
    """
    session_id = str(uuid.uuid4())[:8]
    state = default_agent_state()
    state["session_id"] = session_id
    state["messages"] = [SystemMessage(content=SYSTEM_PROMPT)]
    redis_mgr.save_session(session_id, state)
    redis_mgr.init_context(session_id)
    state["user_input"] = scenario

    for _ in range(MAX_DRIVE_ROUNDS):
        state = await graph.ainvoke(state, config={"recursion_limit": RECURSION_LIMIT})
        if state.get("session_id") != session_id:
            return session_id
        if state.get("pending_question") or state.get("_sub_awaiting"):
            state["pending_question"] = ""
            state["user_input"] = AUTO_CONFIRM_WORD
            continue
        break

    sys.stderr.write(f"[HTTP] 任务未在 {MAX_DRIVE_ROUNDS} 轮内完成，可能异常终止\n")
    sys.stderr.flush()
    return session_id


@app.post("/api/run")
async def run_task(req: RunRequest):
    scenario = (req.scenario or "").strip()
    if not scenario:
        raise HTTPException(status_code=400, detail="scenario 不能为空")
    if not hasattr(app.state, "graph"):
        raise HTTPException(status_code=503, detail="服务未就绪")
    sid = await _run_task_auto(app.state.graph, app.state.redis_mgr, scenario)
    return {
        "session_id": sid,
        "actions_script_url": f"/api/files/{sid}/actions",
        "contingency_plan_url": f"/api/files/{sid}/contingency",
    }


@app.get("/api/files/{sid}/{kind}")
async def download_script(sid: str, kind: str):
    if kind == "actions":
        path = script_path(sid)
    elif kind == "contingency":
        path = contingency_plan_path(sid)
    else:
        raise HTTPException(status_code=404, detail="kind 只能是 actions 或 contingency")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path.name}")
    return FileResponse(path, media_type="application/json", filename=path.name)