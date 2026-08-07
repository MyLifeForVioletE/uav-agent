"""
外部系统执行器：与外部无人机系统交互
（起飞、降落、飞行、载荷操作等）
"""
import sys


async def execute_external(action: dict, context: dict) -> dict:
    """
    执行外部系统类原子动作

    Args:
        action: 原子动作，包含 action_name, required_inputs 等
        context: 执行上下文

    Returns:
        执行结果：{"success": bool, "output": str, "error": str}
    """
    action_name = action.get("action_name", "")

    sys.stderr.write(f"[External] 执行外部动作: {action_name}\n")
    sys.stderr.flush()

    # TODO: 实现与外部无人机系统的实际交互
    # 目前返回 stub 结果
    return {
        "success": True,
        "output": f"外部动作 {action_name} 执行成功（stub）",
        "error": ""
    }
