"""主入口：MCP 连接 → 工具注册 → RAG 初始化 → 图构建 → 交互循环"""
import asyncio
import sys
import uuid
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field

from core.config import OLLAMA_BASE, MODEL, BASE_DIR
from core.state import AgentState, Deps, default_agent_state
from core.prompts import SYSTEM_PROMPT
from core.redis_manager import get_redis_manager
from agent.graph import build_graph
from agent.coordinator import build_coordinator_graph
from rag import TaskPlanner
from tools.scene_builder import build_scene


async def main():
    """主入口：启动 MCP 客户端 → 注册工具 → 构建图 → 进入交互循环。"""
    # 初始化 Redis 管理器
    redis_mgr = get_redis_manager()
    
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
        # 从 MCP server 获得算法工具（field_strength_calc, rf_model_training, rf_model_predict）
        algo_tools = await load_mcp_tools(session)
        print(f"  已注册 {len(algo_tools)} 个算法工具:", flush=True)
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

        # 场景构建工具
        class SceneBuildingInput(BaseModel):
            scene_description: str = Field(description="用户的自然语言场景描述，包含载体、装备、天线、项目等具体信息")

        scene_building_tool = Tool.from_function(
            name="scene_building",
            func=build_scene,
            args_schema=SceneBuildingInput,
            description="仅当用户明确说'生成场景'或'构建场景'时调用（这是电磁场景配置，包含载体、装备、天线、项目等），与任务规划（侦察/巡检/评估）无关。其他情况不要调用此工具。",
        )

        # 合并工具
        all_tools = algo_tools + [scene_building_tool]
        print(f"  已注册 {len(algo_tools)} 个算法工具 + 1 个辅助工具 (scene_building)", flush=True)
        print(f"  场景规划器 ({macro_planner.doc_count} 条) + 约束规划器 ({constraint_planner.doc_count} 条) + 详细规划器 ({detail_planner.doc_count} 条)", flush=True)
        print("=" * 60, flush=True)
        print("  模型随时根据你的需求选择合适的算法调用", flush=True)
        print("  复杂任务可以自动进行规划后分步执行", flush=True)
        print("  多机协同任务会自动检测并切换到多机模式", flush=True)
        print("  如果缺少信息，模型会主动问你", flush=True)
        print('  输入 "exit" 退出', flush=True)
        print("=" * 60, flush=True)

        tool_map = {t.name: t for t in all_tools}
        llm_plain = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_BASE)
        deps = Deps(tools=tool_map, llm_no_tools=llm_plain, macro_planner=macro_planner, constraint_planner=constraint_planner, detail_planner=detail_planner, decomposer_planner=decomposer_planner)
        
        # 构建两个图：单机图 + Coordinator图
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
                
                state["collaboration_mode"] = "multi"
                state = await coordinator_graph.ainvoke(state)
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
            uid = await asyncio.to_thread(input, "\n>>> ")
            uid = uid.strip()
            if uid.lower() == "exit":
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
