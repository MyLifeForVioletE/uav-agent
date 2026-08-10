"""分析 agent 管线验证：模拟 T2 采集完成后的执行，检查 signalAnalysis 是否被调用"""
import asyncio
import sys
import uuid
from pathlib import Path

BASE = r"C:\Users\EHz\Desktop\uavAgentPlanning"
sys.path.insert(0, BASE)

from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field

from core.config import OLLAMA_BASE, MODEL, BASE_DIR
from core.state import Deps
from core.redis_manager import get_redis_manager
from agent.sub_agent import SubAgent
from tools.stub_tools import merge_stub_tools


async def main():
    redis_mgr = get_redis_manager()
    if not redis_mgr._is_available():
        print("[TEST] redis unavailable")
        return
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
        tool_map = {t.name: t for t in merge_stub_tools(algo_tools)}
        llm_plain = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_BASE)
        deps = Deps(tools=tool_map, llm_no_tools=llm_plain, macro_planner=None,
                    constraint_planner=None, detail_planner=None, decomposer_planner=None)

        # 模拟 UAV_1 采集完成后的 Redis 上下文
        redis_mgr.init_context(session_id)
        redis_mgr.update_context(session_id, {
            "uavs": [{"id": "UAV_1", "Longitude": "70", "Latitude": "15.01", "path": "path_xxx.txt"}],
            "targets": [{"id": "T1", "Longitude": "70", "Latitude": "15.01",
                         "Frequency": "1000000000", "frequency_step": "500",
                         "recon_state": "已采集数据", "sweep_data": "0.3545"}],
        })

        task = {
            "task_id": "T2",
            "task_name": "信息处理agent分析扫频数据",
            "goal": "分析扫频数据以获取目标的频率、带宽、信号强度",
            "executor": "processor",
            "prerequisite_tasks": ["T1"],
            "prerequisite_results": {"T1": {"output": "外部动作 沿航线返航 执行成功（stub）"}},
        }

        agent = SubAgent("_processor_", deps, mode="processor",
                         redis_mgr=redis_mgr, session_id=session_id)
        agent.assign_task(task)
        await agent.run_step()

        print(f"\n[TEST] status={agent.status} output={agent.last_output[:400]!r}")

        # 检查上下文是否被分析结果更新
        ctx = redis_mgr.get_context(session_id)
        print(f"[TEST] context targets after: {ctx.get('targets')}")
        print("[TEST] signalAnalysis called:", "signalAnalysis" in str(ctx))
    redis_mgr.close()


if __name__ == "__main__":
    asyncio.run(main())
