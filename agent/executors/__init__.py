"""
执行器分发模块：根据 executor 类型分发到对应的执行器
"""
from .tool_executor import execute_tool_action
from .external_executor import execute_external
from .system_executor import execute_system

# 执行器映射表
EXECUTORS = {
    "tool": execute_tool_action,
    "external": execute_external,
    "system": execute_system,
}


async def dispatch_executor(action: dict, context: dict) -> dict:
    """
    分发执行器：根据 action 中的 executor 字段调用对应的执行器

    Args:
        action: 原子动作，包含 executor, action_name, required_inputs 等
        context: 执行上下文，包含 tools, messages, llm 等

    Returns:
        执行结果：{"success": bool, "output": str, "error": str}
    """
    executor_type = action.get("executor", "")

    if executor_type not in EXECUTORS:
        return {
            "success": False,
            "output": "",
            "error": f"未知的执行器类型: {executor_type}"
        }

    executor_fn = EXECUTORS[executor_type]

    try:
        result = await executor_fn(action, context)
        return result
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"执行器异常: {str(e)}"
        }
