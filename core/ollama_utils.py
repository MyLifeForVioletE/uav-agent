"""Ollama API 格式转换工具函数"""
import json
from pathlib import Path

from core.config import BASE_DIR, OLLAMA_BASE, MODEL


async def achat_ollama(messages: list, *, model: str = MODEL, tools: list = None,
                       temperature: float = 0, num_predict: int = 4096,
                       timeout: float = 120.0) -> dict:
    """异步调用 Ollama /api/chat（非流式）：把同步 requests.post 放入线程池执行。

    多 agent 并行（asyncio.gather）时，若在事件循环里直接发起同步 HTTP 请求，
    单个 agent 的 LLM 调用会阻塞整个循环，其它 agent 全部被卡死。
    放入线程池后阻塞只发生在 worker 线程，事件循环可继续调度其它 agent。
    返回响应中的 message dict（含 content / tool_calls）。
    """
    import asyncio
    import requests
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if tools:
        body["tools"] = tools
    resp = await asyncio.to_thread(
        requests.post, f"{OLLAMA_BASE}/api/chat", json=body, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json().get("message", {})


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


def _load_output_producers() -> dict[str, list[str]]:
    """字段名 → 能产出该字段的算法名列表（依据 algorithms.json 的 output_schema）"""
    path = Path(BASE_DIR) / "algorithms.json"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    result: dict[str, list[str]] = {}
    for a in data.get("capabilities", []):
        out = a.get("output_schema") or {}
        for field in out:
            result.setdefault(field, []).append(a["name"])
    return result


_OUTPUT_PRODUCERS = _load_output_producers()


def producer_algos_for_field(field: str) -> list[str]:
    """返回能产出该字段的算法名列表（用于缺参数据依赖解析）"""
    return list(_OUTPUT_PRODUCERS.get(field, []))


def filter_tools_for_role(tool_map: dict, role: str) -> dict:
    """按 agent 角色过滤工具表

    - role="processor"：分析 agent 只能用 algorithms.json 中 roles 含 "processor" 的算法工具
    - role="uav"：只能用 roles 含 "uav" 的算法工具 + 非注册工具
    """
    result = {}
    for name, tool in tool_map.items():
        roles = _CAP_ROLES.get(name)
        if roles is None:
            # 非 algorithms.json 注册的工具：UAV 可用，分析 agent 不可用
            if role != "processor":
                result[name] = tool
        elif role in roles:
            result[name] = tool
    return result
