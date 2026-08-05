import sys, uuid, json
sys.path.insert(0, r"C:\Users\EHz\Desktop\uavAgentPlanning")

from core.config import BASE_DIR
from core.state import Deps, default_agent_state
from core.redis_manager import get_redis_manager
from agent.coordinator.coordinator_nodes import task_decomposer_node
from rag import TaskPlanner

task = "使用一架侦察无人机到目标位置侦察敌方的一个通信目标。期望得到敌方通信目标的频率、带宽、信号强度信息。该目标通信使用的频率在1GHz~2GHz范围内。"

redis_mgr = get_redis_manager()
session_id = str(uuid.uuid4())[:8]
redis_mgr.init_context(session_id)
redis_mgr.update_context(session_id, {
    "uavs": [{"id": "UAV_1", "Longitude": "70.0", "Latitude": "15.0"}],
    "targets": [{"id": "T1", "Longitude": "70.0", "Latitude": "15.01"}],
    "base": {"Longitude": "70.0", "Latitude": "15.0"},
})

decomposer_planner = TaskPlanner(docs_dir=str(BASE_DIR / "rag_docs"), collection_name="decomposer_rag",
                                 glob_include=["task_decomposition/*.md"])
decomposer_planner.index_docs()
print("RAG refs:", decomposer_planner.doc_count)

deps = Deps(tools={}, llm_no_tools=None, macro_planner=None, constraint_planner=None,
            detail_planner=None, decomposer_planner=decomposer_planner)

state = default_agent_state()
state["session_id"] = session_id
state["original_scenario"] = task
state["user_input"] = task

result = task_decomposer_node(state, deps)
print("pending_question:", result.get("pending_question"))
print("sub_tasks:", json.dumps(result.get("sub_tasks"), ensure_ascii=False, indent=2))
redis_mgr.close()
