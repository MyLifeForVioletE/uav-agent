"""结果回写流程验证：
1. UAV 算法结果 → 发 result 消息给指挥，UAV 不再直写 Redis
2. 指挥 _handle_agent_result：LLM 提取 → 写 Redis（sweep_data 等）
3. 分析 agent 结论 → 发消息 → 指挥写 Redis（analysis_result）
"""
import asyncio
import sys
import uuid

BASE = r"C:\Users\EHz\Desktop\uavAgentPlanning"
sys.path.insert(0, BASE)

from langchain_ollama import ChatOllama

from core.config import OLLAMA_BASE, MODEL, COORDINATOR_ID
from core.state import Deps, default_agent_state
from core.redis_manager import get_redis_manager
from core.kafka_bus import get_kafka_bus
from agent.sub_agent import SubAgent
from fleet.fleet_manager import FleetManager


async def test_uav_result_flow(deps, redis_mgr, session_id):
    print("=" * 40, "Test 1: UAV 结果→指挥→Redis", "=" * 40)
    bus = get_kafka_bus()
    await bus.flush_inbox(COORDINATOR_ID)
    agent = SubAgent("UAV_1", deps, mode="uav", redis_mgr=redis_mgr, session_id=session_id)
    action = {
        "executor": "tool",
        "tool_name": "signalAnalysis",
        "goal": "获取目标 T1 的扫频数据",
        "tool_inputs": {},
    }
    result = {
        "success": True,
        "output": "扫频分析完成：frequency=1.00003 GHz, bandwidth=7 MHz, signal_strength=-76 dB",
        "error": "",
    }
    agent._current_task = {"task_id": "T1", "task_name": "无人机执行扫频侦察", "goal": action["goal"]}

    # UAV 只发消息，不直写 Redis
    await agent._report_result_to_coordinator(action, result)
    ctx_before = redis_mgr.get_context(session_id)
    assert "sweep_data" not in str(ctx_before.get("targets")), "UAV 不应直写 sweep_data"
    print("PASS: UAV 未直写 Redis，仅发送消息")

    # 消息应携带任务/目标信息
    bus = get_kafka_bus()
    msgs = await bus.poll("_coordinator_")
    sent = [m for m in msgs if m.get("from") == "UAV_1"
            and m.get("payload", {}).get("session_id") == session_id]
    assert sent, "指挥应收到 UAV_1 的 result 消息"
    assert sent[0]["payload"]["task_id"] == "T1", "payload 应携带 task_id"
    assert sent[0]["payload"]["task_name"] == "无人机执行扫频侦察", "payload 应携带 task_name"
    print(f"PASS: 消息携带任务信息 task_id={sent[0]['payload']['task_id']}")

    # poll 已消费消息，重新发送一份供指挥处理
    await agent._report_result_to_coordinator(action, result)

    # 指挥处理收件箱
    state = default_agent_state()
    state["session_id"] = session_id
    fm = FleetManager(deps, redis_mgr=redis_mgr)
    await asyncio.sleep(0.5)
    await fm._process_coordinator_inbox(state)

    ctx = redis_mgr.get_context(session_id)
    print(f"Redis targets after: {ctx.get('targets')}")
    targets = ctx.get("targets") or []
    t1 = next((t for t in targets if t.get("id") == "T1"), None)
    assert t1 and any(k in str(t1) for k in ("sweep_data", "Frequency", "bandwidth", "signal_strength")), \
        "扫频结果应写入 target T1"
    assert "T1" in state.get("shared_results", {}), "结果应按 task_id 归档 shared_results"
    print(f"PASS: 指挥 LLM 提取并写入 Redis（归属 T1），shared_results keys={list(state['shared_results'].keys())}")


async def test_analysis_result_flow(deps, redis_mgr, session_id):
    print("=" * 40, "Test 2: 分析结论→指挥→Redis", "=" * 40)
    bus = get_kafka_bus()
    await bus.flush_inbox(COORDINATOR_ID)
    await bus.flush_inbox("_info_processor_")
    agent = SubAgent("_info_processor_", deps, mode="info_processor",
                     redis_mgr=redis_mgr, session_id=session_id)
    agent._current_task = {"task_id": "T2", "goal": "获取目标的频率、带宽、信号强度"}
    agent.last_output = "分析结论：目标频率 1.00003 GHz，带宽 7 MHz，信号强度 -76 dB"
    agent.status = "done"

    await agent._report_analysis_to_coordinator()

    state = default_agent_state()
    state["session_id"] = session_id
    fm = FleetManager(deps, redis_mgr=redis_mgr)
    await asyncio.sleep(0.5)
    await fm._process_coordinator_inbox(state)

    ctx = redis_mgr.get_context(session_id)
    print(f"Redis targets after analysis: {ctx.get('targets')}")
    t2 = next((t for t in (ctx.get("targets") or []) if t.get("id") == "T2"), None)
    assert t2 and any(k in str(t2) for k in ("analysis_result", "sweep_data", "Frequency", "bandwidth", "signal_strength")), \
        "分析结论应写入 target T2"
    print("PASS: 分析结论已由指挥写入 Redis（归属 T2）")


async def main():
    redis_mgr = get_redis_manager()
    if not redis_mgr._is_available():
        print("[TEST] redis unavailable")
        return
    session_id = str(uuid.uuid4())[:8]
    redis_mgr.init_context(session_id)
    redis_mgr.update_context(session_id, {
        "uavs": [{"id": "UAV_1", "Longitude": "70", "Latitude": "15"}],
        "targets": [{"id": "T1", "Longitude": "70", "Latitude": "15.01"},
                    {"id": "T2", "Longitude": "70.02", "Latitude": "15.03"}],
    })

    llm_plain = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_BASE)
    deps = Deps(tools={}, llm_no_tools=llm_plain, macro_planner=None,
                constraint_planner=None, detail_planner=None, decomposer_planner=None)

    try:
        await test_uav_result_flow(deps, redis_mgr, session_id)
        await test_analysis_result_flow(deps, redis_mgr, session_id)
    finally:
        await get_kafka_bus().close()
        redis_mgr.close()
    print("[TEST] DONE")


if __name__ == "__main__":
    asyncio.run(main())
