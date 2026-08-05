"""Ollama API 格式转换工具函数"""
import json
from pathlib import Path

from core.config import BASE_DIR


def messages_to_ollama(messages: list) -> list:
    """将 LangChain 消息列表转为 Ollama API 格式"""
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    result = []
    for msg in messages:
        role = role_map.get(msg.type, "user")
        entry = {"role": role, "content": msg.content or ""}
        # AIMessage 可能带有 tool_calls
        if role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
            o_tcs = []
            for tc in msg.tool_calls:
                o_tcs.append({
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["args"],  # 保持 dict，不要转 string
                    }
                })
            if o_tcs:
                entry["tool_calls"] = o_tcs
        result.append(entry)
    return result


def tools_to_ollama(tool_map: dict) -> list:
    """将工具映射表转为 Ollama tools 格式"""
    o_tools = []
    for name, tool in tool_map.items():
        schema_json = {"type": "object", "properties": {}}
        as_ = tool.args_schema
        if as_ is not None:
            try:
                if hasattr(as_, "model_json_schema"):       # Pydantic v2
                    schema_json = as_.model_json_schema()
                elif hasattr(as_, "schema"):                # Pydantic v1
                    schema_json = as_.schema()
                elif isinstance(as_, dict):
                    schema_json = as_
                elif isinstance(as_, str):
                    schema_json = json.loads(as_)
            except Exception:
                pass
        # 提取 required 字段
        if "properties" not in schema_json:
            schema_json["properties"] = {}
        if "required" not in schema_json:
            schema_json["required"] = list(schema_json.get("properties", {}).keys())
        o_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description or "",
                "parameters": schema_json,
            }
        })
    return o_tools


# ── 按角色过滤工具表 ────────────────────────────────────

def _load_capability_roles() -> dict[str, list[str]]:
    """从 algorithms.json 加载 算法名 → 允许角色 映射"""
    path = Path(BASE_DIR) / "algorithms.json"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return {a["name"]: a.get("roles", []) for a in data.get("capabilities", [])}


_CAP_ROLES = _load_capability_roles()


def filter_tools_for_role(tool_map: dict, role: str) -> dict:
    """按 agent 角色过滤工具表

    - role="coordinator"：协调者可用全部工具（含非注册工具）
    - role="info_processor"：分析 agent 只能用 algorithms.json 中 roles 含 "info_processor" 的算法工具
    - role="uav"：只能用 roles 含 "uav" 的算法工具 + 非注册工具（如 scene_building）
    """
    if role == "coordinator":
        return dict(tool_map)

    result = {}
    for name, tool in tool_map.items():
        roles = _CAP_ROLES.get(name)
        if roles is None:
            # 非 algorithms.json 注册的工具（如 scene_building）：UAV/协调者可用，分析 agent 不可用
            if role != "info_processor":
                result[name] = tool
        elif role in roles:
            result[name] = tool
    return result
