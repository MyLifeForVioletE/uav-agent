"""
外部系统执行器：与外部无人机系统交互
（起飞、降落、飞行、载荷操作等）

飞行/返航动作：脚本中 input.route 由脚本写入器内联为具体航迹点列表；
执行时解析航迹点（优先级：内联列表 → Redis uavs.<id>.path → step:// 引用）
并模拟沿航线飞行、更新无人机位置；无可用航线时保持"等待航线"，不阻塞任务推进。
"""
import json
import re
import sys

from tools.script_writer import is_flight_action, load_steps


def _resolve_route_waypoints(inputs: dict, session_id: str, agent_id: str) -> list:
    """解析飞行/返航动作的航迹点列表，优先级：
    1. tool_inputs.route 已是航迹点列表（内联）
    2. Redis 上下文 uavs.<id>.path（最近一次实际规划，出航/返航自动关联各自航段）
    3. step://<id>.output.path 引用 → 读脚本
    解析失败/无引用时返回空列表。
    """
    route_val = inputs.get("route")
    if isinstance(route_val, list):
        return [dict(w) for w in route_val if isinstance(w, dict)]

    try:
        from core.redis_manager import get_redis_manager
        ctx = get_redis_manager().get_context(session_id) or {}
        uavs = ctx.get("uavs")
        if isinstance(uavs, list):
            for u in uavs:
                if isinstance(u, dict) and u.get("id") == agent_id:
                    p = u.get("path")
                    if isinstance(p, dict) and isinstance(p.get("waypoints"), list):
                        return [dict(w) for w in p["waypoints"]]
                    break
    except Exception as e:
        sys.stderr.write(f"[External] 读取 Redis 航迹失败: {e}\n")
        sys.stderr.flush()

    ref = str(route_val or "").strip()
    if ref.startswith("step://"):
        m = re.match(r"step://([^.]+)", ref)
        if m:
            step_id = m.group(1)
            try:
                for s in load_steps(session_id):
                    if s.get("step_id") == step_id and isinstance(s.get("output"), dict):
                        path = s["output"].get("path") or []
                        if isinstance(path, list):
                            return list(path)
                        break
            except Exception as e:
                sys.stderr.write(f"[External] 航线引用解析失败 {ref}: {e}\n")
                sys.stderr.flush()
    return []


def _simulate_flight(session_id: str, agent_id: str, waypoints: list):
    """模拟飞行：把无人机当前位置更新为航线终点（写回 Redis 上下文）"""
    if not waypoints or not agent_id:
        return
    try:
        from core.redis_manager import get_redis_manager
        redis_mgr = get_redis_manager()
        ctx = redis_mgr.get_context(session_id) or {}
        uavs = ctx.get("uavs")
        if not isinstance(uavs, list):
            uavs = []
            ctx["uavs"] = uavs
        last = waypoints[-1]
        if not isinstance(last, dict):
            return
        lon = last.get("x") if last.get("x") is not None else last.get("Longitude")
        lat = last.get("y") if last.get("y") is not None else last.get("Latitude")
        for u in uavs:
            if isinstance(u, dict) and u.get("id") == agent_id:
                if lon is not None:
                    u["Longitude"] = str(lon)
                if lat is not None:
                    u["Latitude"] = str(lat)
                break
        redis_mgr.save_context(session_id, ctx)
    except Exception as e:
        sys.stderr.write(f"[External] 飞行位置更新失败: {e}\n")
        sys.stderr.flush()


async def execute_external(action: dict, context: dict) -> dict:
    """
    执行外部系统类原子动作

    Args:
        action: 原子动作，包含 action_name, required_inputs 等
        context: 执行上下文

    Returns:
        执行结果：{"success": bool, "output": str, "error": str, "tool_inputs": dict}
    """
    action_name = action.get("action_name", "")
    state = context.get("state") or {}
    agent_id = state.get("_agent_id", "")
    session_id = state.get("session_id", "default")

    sys.stderr.write(f"[External] 执行外部动作: {action_name}\n")
    sys.stderr.flush()

    inputs = dict(action.get("tool_inputs") or {})
    waypoints = []

    if is_flight_action(action_name):
        waypoints = _resolve_route_waypoints(inputs, session_id, agent_id)
        if waypoints:
            _simulate_flight(session_id, agent_id, waypoints)
            sys.stderr.write(
                f"[External] {action_name} 沿航迹飞行（{len(waypoints)} 个航迹点），"
                f"已更新 {agent_id} 位置\n"
            )
            sys.stderr.flush()
        else:
            sys.stderr.write(f"[External] {action_name} 无可用航线（等待航线）\n")
            sys.stderr.flush()

    result = {
        "success": True,
        "output": "",
        "error": "",
        "tool_inputs": inputs,
    }
    if waypoints:
        result["resolved_route"] = waypoints
    return result
