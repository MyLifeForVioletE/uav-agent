"""
Stub 工具：为不真实执行 exe 的算法生成本地 mock 工具。

仅 path_planning 通过 MCP 真正调用 exe 并注册为真实工具；
其余算法（扫频侦察、signalAnalysis 等）不需要真实调用，只需让 LLM 知道有此能力。
这些工具以本地 stub 形式放入 deps.tools，返回与旧 MCP mock 相同的输出格式
（"输出;<字段>"），供上层建立 Redis 输出占位与脚本记录，但不经过 MCP 往返、不调用 exe。
"""
import json
import sys
from pathlib import Path

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field, create_model

from core.config import BASE_DIR

REAL_EXE_ALGORITHMS = {"path_planning"}


def _make_stub_schema(name: str, props: dict):
    """根据 input_schema 的 JSON Schema props 动态构造 pydantic 模型，供 StructuredTool 使用"""
    fields = {}
    for pname, pinfo in props.items():
        t = pinfo.get("type", "string")
        if t == "array":
            py_type = list
        elif t in ("integer", "number"):
            py_type = float if t == "number" else int
        elif t == "boolean":
            py_type = bool
        else:
            py_type = str
        fields[pname] = (py_type, Field(default=..., description=pinfo.get("description", pname)))
    return create_model(f"Stub_{name}", **fields)


def _load_algorithms():
    path = BASE_DIR / "algorithms.json"
    with open(path, encoding="utf-8") as f:
        return {a["name"]: a for a in json.load(f).get("capabilities", [])}


def stub_tool_names():
    """返回应生成本地 stub 的算法名（algorithms.json 中非真实 exe 的算法）"""
    algos = _load_algorithms()
    return [name for name in algos if name not in REAL_EXE_ALGORITHMS]


def build_stub_tools():
    """构建所有非 exe 算法的本地工具列表"""
    algos = _load_algorithms()
    tools = []
    for name in stub_tool_names():
        algo = algos[name]
        input_schema = algo.get("input_schema", {})
        props = {}
        for pname, pinfo in input_schema.items():
            t = pinfo.get("type", "string")
            props[pname] = {
                "description": pinfo.get("description", pname),
                "type": "array" if t == "tuple" else t,
            }
            if "enum" in pinfo:
                props[pname]["enum"] = pinfo["enum"]
        try:
            tool = StructuredTool.from_function(
                name=name,
                description=algo.get("description", name),
                coroutine=_tool_with_name(name),
                args_schema=_make_stub_schema(name, props),
            )
            tools.append(tool)
        except Exception as e:
            sys.stderr.write(f"[StubTools] 构建 {name} 失败: {e}\n")
            sys.stderr.flush()
    return tools


def _tool_with_name(name: str):
    """构造闭包捕获算法名，转发到 _call_algo"""
    async def wrapper(**kwargs):
        return await _call_algo(name, kwargs)
    return wrapper


async def _call_algo(name: str, kwargs: dict):
    """非 exe 算法 mock 执行：返回与旧 MCP mock 相同的 content 格式（内嵌 JSON body）。"""
    algos = _load_algorithms()
    out_schema = algos.get(name, {}).get("output_schema", {})
    # 有输出字段时 stdout 为 "输出;字段1, 字段2" 占位；无输出 schema（如 flight）则不给
    # 伪造输出，stdout 为空字符串，脚本中该步骤 output 记录为空（""），不出现 "输出;无"。
    output_files = ", ".join(out_schema.keys())
    body = {
        "returncode": 0,
        "stdout": f"输出;{output_files}" if output_files else "",
        "stderr": "",
    }
    return [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]


def merge_stub_tools(algo_tools):
    """合并真实 MCP 算法工具与本地 stub 工具，返回去重后的列表"""
    merged = list(algo_tools)
    existing = {t.name for t in merged}
    for st in build_stub_tools():
        if st.name not in existing:
            merged.append(st)
            existing.add(st.name)
    return merged