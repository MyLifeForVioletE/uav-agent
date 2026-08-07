"""
MCP Server - 动态读取 algorithms.json，注册所有算法工具
轻量 JSON-RPC over stdio，零外部依赖
"""
import json
import subprocess
import sys
from pathlib import Path

# 子进程需要能找到项目根目录的包
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import BASE_DIR


def load_algorithms():
    """从 algorithms.json 加载算法配置列表（名称、执行路径、参数定义等）"""
    path = BASE_DIR / "algorithms.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("capabilities", [])


def _convert_type(raw_type):
    """将 algorithms.json 中的自定义类型映射为 JSON Schema 标准类型"""
    if raw_type == "tuple":
        return {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    if raw_type == "file":
        return {"type": "string", "format": "uri"}
    return raw_type


# 需要真正调用 exe 的算法（路径规划），仅此算法注册为 MCP 工具；
# 其余算法不提供 MCP 注册，能力由本地 stub 工具（tools/stub_tools.py）对外暴露。
REAL_EXE_ALGORITHMS = {"path_planning"}


def build_tool_definitions(algorithms):
    """将 algorithms.json 中的每个算法转为 MCP tools/list 响应格式的 tool schema

    仅注册需要真实执行 exe 的算法（REAL_EXE_ALGORITHMS）；其余算法无需 MCP 调用，
    交由上层本地 stub 工具处理，故不在此注册。
    """
    tools = []
    for algo in algorithms:
        if algo["name"] not in REAL_EXE_ALGORITHMS:
            continue
        props = {}
        required = []
        input_schema = algo.get("input_schema", {})
        for pname, pinfo in input_schema.items():
            ptype = pinfo.get("type", "string")
            prop = {"description": pinfo.get("description", "")}
            converted = _convert_type(ptype)
            if isinstance(converted, dict):
                prop.update(converted)
            else:
                prop["type"] = converted
            if "enum" in pinfo:
                prop["enum"] = pinfo["enum"]
            if "default" in pinfo:
                prop["default"] = pinfo["default"]
            if pinfo.get("required") == "true":
                required.append(pname)
            props[pname] = prop

        desc = algo.get("description", "")
        tools.append({
            "name": algo["name"],
            "description": desc,
            "inputSchema": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        })
    return tools


def execute_algorithm(algo, arguments):
    """执行算法：仅 path_planning 真正调用 exe；其余算法跳过 exe，返回模拟结果。

    非路径规划算法不实际执行外部程序，仅确认参数已确定，返回结构化模拟输出，
    供上层据此建立 Redis 输出占位字段（即便当前没有真实值）。
    """
    sys.stderr.write(f"[MCP] execute_algorithm: {algo['name']}, args={arguments}\n")
    sys.stderr.flush()
    algo_name = algo.get("name", "")

    # 非真实执行的算法：不调用 exe，返回模拟成功结果（含输出 schema，供占位）
    if algo_name not in REAL_EXE_ALGORITHMS:
        sys.stderr.write(f"[MCP] {algo_name} 跳过真实 exe 调用，返回模拟输出\n")
        sys.stderr.flush()
        out_schema = algo.get("output_schema", {})
        output_files = ", ".join(out_schema.keys()) or "无"
        mock = {
            "note": f"算法 {algo_name} 未调用真实 exe，仅记录参数并建立输出占位",
            "output_fields": list(out_schema.keys()),
        }
        return {
            "returncode": 0,
            "stdout": f"输出;{output_files}",
            "stderr": "",
        }

    exe_path = Path(algo["executable"])
    if not exe_path.is_file():
        return {"error": f"可执行文件不存在: {algo['executable']}"}

    # 确定工作目录：优先用配置的 working_dir，否则用 exe 所在目录
    work_dir = Path(algo.get("working_dir", exe_path.parent))
    if not work_dir.is_dir():
        work_dir = exe_path.parent

    # 按参数定义顺序构建命令行参数（保证与算法 exe 的期望顺序一致）
    args = [str(exe_path)]
    for pname, pinfo in algo.get("input_schema", {}).items():
        prefix = pinfo.get("prefix", "")
        if pname in arguments:
            val = str(arguments[pname])
            fmt = pinfo.get("format", "")
            if prefix:
                args.append(prefix)
            if fmt == "space":
                val = val.strip("()").replace(",", " ").replace("，", " ")
                args.extend(val.split())
            elif fmt == "comma":
                val = val.strip("()").replace(" ", ",").replace("，", ",")
                args.append(val)
            else:
                args.append(val)
        elif "default" in pinfo:
            if prefix:
                args.append(prefix)
            args.append(str(pinfo["default"]))
        elif pinfo.get("required") == "true":
            return {"error": f"缺少必需参数: {pname}"}

    # 调用外部 exe，捕获标准输出和错误输出
    try:
        sys.stderr.write(f"[MCP] 执行命令: {' '.join(args)}\n")
        sys.stderr.flush()
        result = subprocess.run(
            args, cwd=str(work_dir),
            capture_output=True, text=True, timeout=120,
        )
        sys.stderr.write(f"[MCP] 执行完成: returncode={result.returncode}, stdout={result.stdout[:200]}\n")
        sys.stderr.flush()
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"error": "程序执行超时（120秒）"}
    except Exception as e:
        return {"error": str(e)}


def handle_request(request, algorithms, tool_defs):
    """
    处理 JSON-RPC 请求的调度中心。
    支持的 method：initialize, notifications/initialized, tools/list, tools/call
    """
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "initialize":
        # MCP 初始握手：返回服务器能力信息（协议版本、工具支持等）
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "algo-mcp-server", "version": "1.0.0"},
            },
        }

    elif method == "notifications/initialized":
        # 客户端已初始化通知（无需响应 body）
        return None

    elif method == "tools/list":
        # 返回所有可用工具的定义（名称、描述、参数 schema）
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": tool_defs},
        }

    elif method == "tools/call":
        # 调用指定工具：查找算法定义 → 执行 → 返回结果
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        algo = next((a for a in algorithms if a["name"] == tool_name), None)
        if not algo:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"未知工具: {tool_name}"},
            }

        try:
            result = execute_algorithm(algo, tool_args)
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}
                    ]
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps({"error": str(e)}, ensure_ascii=False)}
                    ]
                },
            }

    else:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"未知方法: {method}"},
        }


def main():
    """
    主循环：从 algorithms.json 加载算法配置，
    通过 stdin/stdout 与客户端进行 JSON-RPC 通信（MCP stdio 传输）。
    启动日志写入 stderr 避免污染 stdout 协议流。
    """
    algorithms = load_algorithms()
    tool_defs = build_tool_definitions(algorithms)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # 注册信息写入 stderr，避免干扰 stdout 的 JSON-RPC 数据流
    import io
    info_out = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    for t in tool_defs:
        props = list(t["inputSchema"]["properties"].keys())
        info_out.write(f"  [注册] {t['name']} ({', '.join(props)})\n")
        info_out.flush()

    # 逐行读取 stdin 上的 JSON-RPC 请求并处理
    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        response = handle_request(request, algorithms, tool_defs)
        if response is not None:
            msg = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
            stdout.write(msg)
            stdout.flush()


if __name__ == "__main__":
    main()
