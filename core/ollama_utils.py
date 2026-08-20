"""Ollama API 格式转换工具函数"""
import asyncio
import json
from pathlib import Path

from core.config import BASE_DIR, OLLAMA_BASE, MODEL

# Ollama 单进程串行处理请求：多 agent 并行调用时若同时涌入，
# 排队的请求会把等待时间计入自身耗时，120s 超时容易被挤爆。
# 这里用信号量限制并发（与 Ollama 实际串行一致，只是让等待计入自身调度），
# 并把超时放宽到 300s，避免详细规划阶段长生成被超时打断。
_OLLAMA_MAX_CONCURRENT = 2
_OLLAMA_TIMEOUT = 300.0

_ollama_sem: asyncio.Semaphore | None = None


def _get_ollama_sem() -> asyncio.Semaphore:
    global _ollama_sem
    if _ollama_sem is None:
        _ollama_sem = asyncio.Semaphore(_OLLAMA_MAX_CONCURRENT)
    return _ollama_sem


async def achat_ollama(messages: list, *, model: str = MODEL, tools: list = None,
                       temperature: float = 0, num_predict: int = 4096,
                       timeout: float = _OLLAMA_TIMEOUT, num_ctx: int = 32768) -> dict:
    """异步调用 Ollama /api/chat（非流式）：把同步 requests.post 放入线程池执行。

    多 agent 并行（asyncio.gather）时，若在事件循环里直接发起同步 HTTP 请求，
    单个 agent 的 LLM 调用会阻塞整个循环，其它 agent 全部被卡死。
    放入线程池后阻塞只发生在 worker 线程，事件循环可继续调度其它 agent。
    返回响应中的 message dict（含 content / tool_calls）。

    num_ctx 显式传 32768：qwen3-4b-agent 的 Modelfile 默认 num_ctx=16384，
    详细规划逐阶段累积后请求极易超限（Ollama 返回 exceed_context_size_error），
    这里统一放宽到 32768（模型上限 40960 内）。

    并发受信号量限制（_OLLAMA_MAX_CONCURRENT），超时默认 300s：Ollama 串行处理，
    多 agent 并行时排队的请求会把等待计入耗时，120s 容易 Read timed out。
    """
    import requests
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if tools:
        body["tools"] = tools
    async with _get_ollama_sem():
        resp = await asyncio.to_thread(
            requests.post, f"{OLLAMA_BASE}/api/chat", json=body, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("message", {})


def _compact_history(messages: list, max_exchanges: int = 2) -> list:
    """压缩消息历史：保留全部 system 消息 + 含宏观规划（macro_phases）的 assistant 消息
    + 最近 max_exchanges 轮（human+assistant）对话。

    详细规划逐阶段注入时，每个阶段 prompt 都重复携带 schema/算法表/RAG 上下文，
    消息列表只增不减，累积几轮后极易超出模型上下文窗口导致 Ollama 报错。
    阶段 prompt 自包含（含完整 schema 与参考上下文），模型只需当前阶段即可，
    故历史可安全压缩到最近 1-2 轮。仅影响发送给 Ollama 的副本，不改 state["messages"]。
    """
    if max_exchanges <= 0:
        return list(messages)
    system_msgs = [m for m in messages if m.type == "system"]
    non_system = [m for m in messages if m.type != "system"]
    macro_plan_msgs = [
        m for m in non_system
        if m.type == "ai" and isinstance(m.content, str) and "macro_phases" in m.content
    ]
    keep = non_system[-max_exchanges * 2:] if non_system else []
    merged = []
    seen = set()
    for m in system_msgs + macro_plan_msgs + keep:
        if id(m) not in seen:
            seen.add(id(m))
            merged.append(m)
    return merged


def messages_to_ollama(messages: list, max_exchanges: int = 2) -> list:
    """将 LangChain 消息列表转为 Ollama API 格式（默认压缩历史，防止累积超出上下文窗口）"""
    if max_exchanges > 0:
        messages = _compact_history(messages, max_exchanges)
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
