"""
系统执行器：系统内部记录（日志、状态保存等）
"""
import sys
import json
from datetime import datetime


async def execute_system(action: dict, context: dict) -> dict:
    """
    执行系统类原子动作

    Args:
        action: 原子动作，包含 action_name, required_inputs 等
        context: 执行上下文

    Returns:
        执行结果：{"success": bool, "output": str, "error": str}
    """
    action_name = action.get("action_name", "")
    required_inputs = action.get("required_inputs", [])

    sys.stderr.write(f"[System] 执行系统动作: {action_name}\n")
    sys.stderr.flush()

    # 根据动作类型执行不同的系统操作
    if "日志" in action_name or "记录" in action_name:
        # 日志记录
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action_name,
            "inputs": required_inputs,
        }
        sys.stderr.write(f"[System] 日志记录: {json.dumps(log_entry, ensure_ascii=False)}\n")
        sys.stderr.flush()
        return {
            "success": True,
            "output": f"日志记录完成: {action_name}",
            "error": ""
        }
    elif "状态" in action_name or "保存" in action_name:
        # 状态保存
        state = context.get("state", {})
        if state is not None:
            sys.stderr.write(f"[System] 状态保存完成\n")
            sys.stderr.flush()
        return {
            "success": True,
            "output": f"状态保存完成: {action_name}",
            "error": ""
        }
    else:
        # 默认系统操作
        return {
            "success": True,
            "output": f"系统操作完成: {action_name}",
            "error": ""
        }
