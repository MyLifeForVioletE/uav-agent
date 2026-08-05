"""Kafka 消息总线 E2E 验证：
1. 总线 send/poll 跨 topic 通信
2. 子 agent send_message 工具：request + await_reply → 挂起
3. 指挥收件箱处理：指挥 LLM 决策回复（同步请示闭环）
4. report/result 上报归档到 shared_results / Redis 上下文
"""
import asyncio
import sys
import uuid
from pathlib import Path

BASE = r"C:\Users\EHz\Desktop\uavAgentPlanning"
sys.path.insert(0, BASE)

from langchain_ollama import ChatOllama

from core.config import OLLAMA_BASE, MODEL, BASE_DIR, COORDINATOR_ID
from core.state import Deps, default_agent_state
from core.redis_manager import get_redis_manager
from core.kafka_bus import get_kafka_bus, new_msg_id
from agent.sub_agent import SubAgent
from fleet.fleet_manager import FleetManager


async def test_bus_basic():
    print("=" * 40, "Test 1: 总线 send/poll", "=" * 40)
    bus = get_kafka_bus()
    await bus.flush_inbox(COORDINATOR_ID)
    corr = new_msg_id()
    await bus.send("UAV_1", COORDINATOR_ID, "request", "请求目标坐标", correlation_id=corr)
    got = []
    for _ in range(10):
        await asyncio.sleep(0.5)
        got = await bus.poll(COORDINATOR_ID, timeout=1.0)
        if any(m.get("correlation_id") == corr for m in got):
            break
    matched = [m for m in got if m.get("correlation_id") == corr]
    assert matched, f"未收到带 corr 的消息: {got}"
    assert matched[0]["msg_type"] == "request"
    print(f"PASS: 收到消息 {matched[0]}")


async def test_sync_request(deps, redis_mgr, session_id):
    print("=" * 40, "Test 2: 同步请示闭环", "=" * 40)
    bus = get_kafka_bus()
    await bus.flush_inbox(COORDINATOR_ID)
    await bus.flush_inbox("UAV_1")
    agent = SubAgent("UAV_1", deps, mode="uav", redis_mgr=redis_mgr, session_id=session_id)
    agent.status = "running"

    # 通过 send_message 工具发送请示（LLM 驱动的入口）
    tool = agent.deps.tools.get("send_message")
    assert tool is not None, "send_message 工具未注册到 UAV agent"
    result = await tool.ainvoke({
        "to": COORDINATOR_ID,
        "msg_type": "request",
        "content": "请问 T1 目标的坐标是多少？我需要前往侦察。",
        "await_reply": True,
    })
    print(f"工具返回: {result}")
    assert agent._awaiting_reply is not None, "应进入等待回复状态"
    print(f"PASS: agent 已挂起，awaiting={agent._awaiting_reply}")

    # 指挥侧处理收件箱
    state = default_agent_state()
    state["session_id"] = session_id
    state["uav_states"] = {"UAV_1": agent.get_status_dict()}
    state["shared_results"] = {"T1": {"status": "done", "output": "扫频数据 0.3545"}}
    fm = FleetManager(deps, redis_mgr=redis_mgr)
    await fm._process_coordinator_inbox(state)

    # 请示回复应已进入 UAV_1 收件箱
    await asyncio.sleep(1.0)
    before = agent._awaiting_reply
    await agent._drain_inbox()
    assert agent._awaiting_reply is None, f"回复到达后应解除挂起，仍为 {before}"
    last = agent.state["messages"][-1]
    print(f"PASS: 收到指挥回复并解除挂起: {last.content[:100]}")
    print(f"指挥邮箱归档 {len(state.get('coordinator_mailbox', []))} 条")
    return state


async def test_report(deps, redis_mgr, session_id, state):
    print("=" * 40, "Test 3: 上报归档", "=" * 40)
    bus = get_kafka_bus()
    await bus.flush_inbox(COORDINATOR_ID)
    await bus.send("UAV_1", COORDINATOR_ID, "report",
                   "UAV_1 已完成 T1 扫频侦察",
                   payload={
                       "session_id": session_id,
                       "task_id": "T1",
                       "result": {"sweep_data": "0.3545", "frequency": 1e9},
                   })
    await asyncio.sleep(1.0)
    fm = FleetManager(deps, redis_mgr=redis_mgr)
    await fm._process_coordinator_inbox(state)
    sr = state.get("shared_results", {})
    assert "T1" in sr, f"shared_results 未归档 T1: {sr.keys()}"
    print(f"PASS: shared_results[{list(sr.keys())}] = {sr.get('T1')}")
    ctx = redis_mgr.get_context(session_id)
    print(f"Redis 上下文 targets: {ctx.get('targets')}")


async def main():
    redis_mgr = get_redis_manager()
    if not redis_mgr._is_available():
        print("[TEST] redis unavailable")
        return
    session_id = str(uuid.uuid4())[:8]
    redis_mgr.init_context(session_id)
    redis_mgr.update_context(session_id, {
        "uavs": [{"id": "UAV_1", "Longitude": "70", "Latitude": "15.01"}],
        "targets": [{"id": "T1", "Longitude": "70", "Latitude": "15.01", "sweep_data": "0.3545"}],
    })

    llm_plain = ChatOllama(model=MODEL, temperature=0, base_url=OLLAMA_BASE)
    deps = Deps(tools={}, llm_no_tools=llm_plain, macro_planner=None,
                constraint_planner=None, detail_planner=None, decomposer_planner=None)

    try:
        await test_bus_basic()
        state = await test_sync_request(deps, redis_mgr, session_id)
        await test_report(deps, redis_mgr, session_id, state)
    finally:
        bus = get_kafka_bus()
        await bus.close()
        redis_mgr.close()
    print("[TEST] DONE")


if __name__ == "__main__":
    asyncio.run(main())
