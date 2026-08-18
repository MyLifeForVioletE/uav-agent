"""主入口：MCP 连接 → 工具注册 → RAG 初始化 → 图构建 → 交互循环"""
import asyncio
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from core.config import OLLAMA_BASE, MODEL, BASE_DIR
from core.state import AgentState, Deps, default_agent_state
from core.prompts import SYSTEM_PROMPT
from core.redis_manager import get_redis_manager
from core.timing import timing
from agent.coordinator import build_coordinator_graph
from rag import TaskPlanner
from tools.stub_tools import merge_stub_tools


async def main():
    """主入口：启动 MCP 客户端 → 注册工具 → 构建图 → 进入交互循环。"""
    # 初始化 Redis 管理器
    redis_mgr = get_redis_manager()
    _t0 = time.perf_counter()
    
    # 生成会话ID
    session_id = str(uuid.uuid4())[:8]
    
    # 通过 stdio 启动 mcp_server.py 作为子进程
    client = MultiServerMCPClient({
        "mcp-server": {
            "command": sys.executable,
            "args": [str(BASE_DIR / "tools" / "mcp_server.py")],
            "transport": "stdio",
        }
    })

    print("=" * 60, flush=True)
    print("  [连接 MCP Server...]", flush=True)

    async with client.session("mcp-server") as session:
        # 从 MCP server 获得算法工具（仅注册真实执行 exe 的 path_planning）
        algo_tools = await load_mcp_tools(session)
        print(f"  MCP 真实工具 {len(algo_tools)} 个:", flush=True)
        for t in algo_tools:
            print(f"    - {t.name}", flush=True)

        # 初始化 RAG 任务规划器：场景 + 约束 + 详细规划三个独立索引
        rag_base = str(BASE_DIR / "rag_docs")
        macro_planner = TaskPlanner(
            docs_dir=rag_base,
            collection_name="macro_rag",
            glob_include=["macro_examples/*.md"],
        )
        constraint_planner = TaskPlanner(
            docs_dir=rag_base,
            collection_name="constraint_rag",
            glob_include=["constraint_sop/*.md"],
            no_chunk=True,
        )
        detail_planner = TaskPlanner(
            docs_dir=rag_base,
            collection_name="detail_rag",
            glob_include=["phase_decmposition/*.md"],
        )
        decomposer_planner = TaskPlanner(
            docs_dir=rag_base,
            collection_name="decomposer_rag",
            glob_include=["task_decomposition/*.md"],
        )
        macro_planner.index_docs()
        constraint_planner.index_docs()
        detail_planner.index_docs()
        decomposer_planner.index_docs()

        # 合并工具：MCP 仅提供真实执行 exe 的 path_planning，其余算法用本地 stub 工具补齐能力
        all_tools = merge_stub_tools(algo_tools)
        tool_map = {t.name: t for t in all_tools}
        print(f"  已注册 {len(tool_map)} 个算法工具（真实 exe: path_planning, 其余为本地 stub)", flush=True)
        print(f"  场景规划器 ({macro_planner.doc_count} 条) + 约束规划器 ({constraint_planner.doc_count} 条) + 详细规划器 ({detail_planner.doc_count} 条)", flush=True)
        print("=" * 60, flush=True)
        print("  模型随时根据你的需求选择合适的算法调用", flush=True)
        print("  复杂任务可以自动进行规划后分步执行", flush=True)
        print("  每个任务统一由指挥 agent 协调 UAV/信息处理子 agent 协同执行", flush=True)
        print("  如果缺少信息，模型会主动问你", flush=True)
        print('  输入 "exit" 退出', flush=True)
        print("=" * 60, flush=True)

        tool_map = {t.name: t for t in all_tools}
        llm_plain = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_BASE, request_timeout=60)
        deps = Deps(tools=tool_map, llm_no_tools=llm_plain, macro_planner=macro_planner, constraint_planner=constraint_planner, detail_planner=detail_planner, decomposer_planner=decomposer_planner)
        timing.init_time = time.perf_counter() - _t0
        
        # 构建 Coordinator 图（入口）；子 agent 内部复用 agent/graph.py 执行图
        coordinator_graph = build_coordinator_graph(deps)
        
        # 初始状态
        state: AgentState = default_agent_state()
        state["session_id"] = session_id
        state["messages"] = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # 保存初始会话到 Redis
        redis_mgr.save_session(session_id, state)
        
        # 初始化结构化上下文模板
        redis_mgr.init_context(session_id)
        
        print(f"\n  会话ID: {session_id}", flush=True)

        # 主循环：运行图 → 显示输出 → 读用户输入 → 注入状态
        while True:
            user_input = state.get("user_input")
            
            if user_input:
                # 追加用户消息到 Redis
                redis_mgr.append_message(session_id, {"type": "human", "content": user_input})
                
                _t1 = time.perf_counter()
                state = await coordinator_graph.ainvoke(state)
                timing.exec_wall += time.perf_counter() - _t1
                state["user_input"] = ""
                out = state.get("output", "")
                pq = state.get("pending_question", "")
                print(f"[DEBUG] out_len={len(out)} pq={pq!r}", flush=True)  # 清除，避免无限循环
                
                # 追加AI回复到 Redis
                if out:
                    redis_mgr.append_message(session_id, {"type": "ai", "content": out})
                
                # 保存会话状态到 Redis
                redis_mgr.save_session(session_id, state)
            
            # 显示输出
            out = state.get("output", "")
            pq = state.get("pending_question", "")
            
            if out:
                print(f"\n{out}", flush=True)
            if pq and pq.strip() and pq != out:
                print(f"\n{pq}", flush=True)

            # 读取用户输入
            _t2 = time.perf_counter()
            uid = await asyncio.to_thread(input, "\n>>> ")
            timing.user_wait += time.perf_counter() - _t2
            uid = uid.strip()
            if uid.lower() == "exit":
                print(timing.format_report(), flush=True)
                break
            
            # 如果有待确认问题，用户的输入是确认，清除 pending_question
            if state.get("pending_question"):
                state["pending_question"] = ""
            
            state["user_input"] = uid

    # 退出前关闭 Redis 连接
    redis_mgr.close()
    
    print("\n  退出成功", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
