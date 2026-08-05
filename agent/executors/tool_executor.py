"""
工具执行器：执行算法工具调用
包含参数查找逻辑：Redis缓存 → knowledge目录扫描 → LLM选择 → 询问用户
"""
import json
import sys
from pathlib import Path
from typing import Optional

from core.config import BASE_DIR
from core.redis_manager import get_redis_manager


# knowledge 目录支持的文件类型
KNOWLEDGE_EXTENSIONS = {".txt", ".json", ".ldf"}


def _load_algorithms():
    """加载算法配置"""
    path = BASE_DIR / "algorithms.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {a["name"]: a for a in data.get("capabilities", [])}


# 模块级缓存
_ALGORITHMS = None


def _get_algorithms():
    global _ALGORITHMS
    if _ALGORITHMS is None:
        _ALGORITHMS = _load_algorithms()
    return _ALGORITHMS


def _scan_knowledge_dir() -> list[dict]:
    """
    扫描 knowledge/ 目录，返回所有支持类型的文件列表
    Returns: [{"name": "文件名", "path": "完整路径", "ext": ".json"}, ...]
    """
    knowledge_dir = BASE_DIR / "knowledge"
    if not knowledge_dir.is_dir():
        return []
    
    files = []
    for f in knowledge_dir.iterdir():
        if f.is_file() and f.suffix.lower() in KNOWLEDGE_EXTENSIONS:
            files.append({
                "name": f.name,
                "path": str(f),
                "ext": f.suffix.lower(),
            })
    return files


def _validate_param_value(value: str, param_type: str) -> bool:
    """校验参数值是否匹配预期类型"""
    import re
    v = value.strip()
    if not v:
        return False
    if param_type == "file":
        return v.endswith((".json", ".txt", ".ldf", ".csv", ".xml", ".yaml", ".yml")) or "\\" in v or "/" in v
    if param_type == "tuple":
        return bool(re.match(r'^\s*\(\s*[\d.]+\s*,\s*[\d.]+\s*\)\s*$', v))
    if param_type == "number":
        try:
            float(v)
            return True
        except ValueError:
            return False
    return True


def _normalize_param_value(value: str, param_type: str, param_format: str = "") -> str:
    """
    将 LLM 提取的参数值规范化为算法需要的格式。
    主要处理坐标参数：LLM 常返回 {"Longitude": x, "Latitude": y} 之类的 JSON 对象，
    需要转为 "x,y" 形式再传给 exe。
    """
    import re
    v = str(value).strip()
    if not v:
        return value
    # JSON 对象形式：提取经纬度键（无论声明类型如何）
    if v.startswith("{"):
        try:
            obj = json.loads(v)
            if isinstance(obj, dict):
                lon = (obj.get("Longitude") or obj.get("longitude")
                       or obj.get("lon") or obj.get("Lon") or obj.get("x"))
                lat = (obj.get("Latitude") or obj.get("latitude")
                       or obj.get("lat") or obj.get("Lat") or obj.get("y"))
                if lon is not None and lat is not None:
                    return f"{lon},{lat}"
        except Exception:
            pass
    is_coord = (param_type == "tuple") or param_format in ("comma", "space")
    if is_coord:
        # 已是 "x,y" 或 "(x,y)" 形式，直接规范分隔符
        if re.match(r'^[\(\[]?\s*-?\d+\.?\d*\s*[,，]\s*-?\d+\.?\d*\s*[\)\]]?$', v):
            return v.strip("()[]").replace("，", ",").replace(" ", "")
        # 兜底：按出现顺序取前两个数字
        nums = re.findall(r"-?\d+\.?\d*", v)
        if len(nums) >= 2:
            return f"{nums[0]},{nums[1]}"
    return value


def _build_file_list_text(files: list[dict]) -> str:
    """将文件列表格式化为文本"""
    if not files:
        return "（knowledge 目录下无可用文件）"
    
    lines = []
    for f in files:
        lines.append(f"- {f['name']} ({f['ext']})")
    return "\n".join(lines)


async def _llm_select_file(llm, files: list[dict], scene_description: str, 
                           param_name: str, param_desc: str) -> Optional[str]:
    """
    调用 LLM 从文件列表中选择最合适的文件
    Returns: 选中的文件路径，或 None
    """
    if not files or not llm:
        return None
    
    file_list_text = _build_file_list_text(files)
    
    prompt = f"""你是一个文件选择助手。根据场景描述和参数需求，从文件列表中选择最合适的文件。

场景描述：{scene_description}

需要填写的参数：{param_name}
参数描述：{param_desc}

可用文件列表：
{file_list_text}

请只输出选中的文件名（不包含路径），如果无法确定则输出 "NONE"。"""

    try:
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        selected = response.content.strip()
        
        # 验证选择
        if selected == "NONE" or not selected:
            return None
        
        # 检查选中的文件是否在列表中
        for f in files:
            if f["name"] == selected:
                return f["path"]
        
        # 尝试模糊匹配
        for f in files:
            if selected in f["name"] or f["name"] in selected:
                return f["path"]
        
        return None
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] LLM选择文件失败: {e}\n")
        sys.stderr.flush()
        return None


async def _llm_extract_value_from_file(llm, file_path: str, file_ext: str,
                                       scene_description: str, param_name: str,
                                       param_desc: str) -> Optional[str]:
    """
    读取文件内容，用 LLM 提取指定参数的值
    适用于 tuple/string/number 等非 file 类型参数
    Returns: 提取到的值字符串，或 None
    """
    if not llm:
        return None

    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        try:
            content = Path(file_path).read_text(encoding="gbk")
        except Exception as e:
            sys.stderr.write(f"[ToolExecutor] 读取文件失败 {file_path}: {e}\n")
            sys.stderr.flush()
            return None

    # 截断过长内容
    if len(content) > 3000:
        content = content[:3000] + "\n...(已截断)"

    prompt = f"""你是一个数据提取助手。根据场景和参数需求，从文件内容中提取对应值。

场景描述：{scene_description}

需要提取的参数：{param_name}
参数描述：{param_desc}
参数类型：非文件类型（如坐标、数值、字符串等）

文件内容：
{content}

请从文件内容中提取该参数的值。如果是坐标，输出格式如 x y；如果是数值，只输出数字；如果是字符串，输出原始字符串。
如果文件中找不到相关信息，输出 "NONE"。"""

    try:
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        value = response.content.strip()

        if value == "NONE" or not value:
            return None

        # 去除可能的引号包裹
        value = value.strip('"').strip("'")
        return value
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] LLM提取值失败: {e}\n")
        sys.stderr.flush()
        return None


async def _llm_extract_from_context(llm, param_name: str, param_desc: str,
                                     param_type: str, context: dict, goal: str = "") -> Optional[str]:
    """
    用 LLM 从结构化上下文中提取指定参数的值
    Returns: 提取到的值，或 None
    """
    if not llm or not context:
        sys.stderr.write(f"[ToolExecutor] _llm_extract_from_context: llm={llm is not None}, context={'有' if context else '空'}\n")
        sys.stderr.flush()
        return None
    
    context_text = json.dumps(context, ensure_ascii=False, indent=2)
    
    sys.stderr.write(f"[ToolExecutor] LLM提取: {param_name}({param_desc}), goal={goal[:50] if goal else ''}, 上下文有 {len(context)} 个分组\n")
    sys.stderr.flush()
    
    goal_hint = f"\n动作目标：{goal}\n根据动作目标理解该参数的含义。" if goal else ""
    
    prompt = f"""从结构化上下文中提取指定参数的值。
{goal_hint}
结构化上下文：
{context_text}

需要提取的参数：
- 参数名: {param_name}
- 参数描述: {param_desc}
- 参数类型: {param_type}

请从上下文中找到与该参数语义最匹配的值，只返回值本身（不要参数名、不要引号、不要其他内容）。
例如 goal 说"从起始点到目标位置1"，那么 uavPosition 应该返回起始点坐标，targetPosition 应该返回目标位置1坐标。
如果没有找到匹配的值，返回 "NONE"。"""

    try:
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        
        sys.stderr.write(f"[ToolExecutor] LLM提取结果: '{result}'\n")
        sys.stderr.flush()
        
        if result == "NONE" or not result:
            return None
        
        return result
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] LLM提取失败: {e}\n")
        sys.stderr.flush()
        return None


async def execute_tool_action(action: dict, context: dict) -> dict:
    """
    执行工具类原子动作

    Args:
        action: 原子动作，包含 tool_name, tool_inputs 等
        context: 执行上下文，包含 tools, messages, llm, state 等

    Returns:
        执行结果：{"success": bool, "output": str, "error": str, "pending": bool}
    """
    tools = context.get("tools", {})
    messages = context.get("messages", [])
    llm = context.get("llm", None)
    state = context.get("state", {})
    
    tool_name = action.get("tool_name", action.get("action_name", ""))
    tool_inputs = action.get("tool_inputs", {})

    # 查找匹配的工具
    tool = tools.get(tool_name)
    if tool is None:
        # 模糊匹配
        for name, tool_obj in tools.items():
            if tool_name == name or tool_name in name:
                tool = tool_obj
                tool_name = name
                break

    if tool is None:
        return {
            "success": False,
            "output": "",
            "error": f"未找到工具: {tool_name}",
            "pending": False,
        }

    # 获取算法配置中的参数定义
    algorithms = _get_algorithms()
    algo_config = algorithms.get(tool_name, {})
    input_schema = algo_config.get("input_schema", {})
    
    # 获取场景描述
    scene_description = state.get("original_scenario", "")
    
    # 获取 session_id
    session_id = state.get("session_id", "default")
    
    # 获取 Redis 管理器
    redis_mgr = get_redis_manager()
    
    sys.stderr.write(f"[ToolExecutor] session_id={session_id}\n")
    sys.stderr.flush()
    
    # 调试：检查 Redis 中的 context key
    raw_ctx = redis_mgr.get(session_id)
    sys.stderr.write(f"[ToolExecutor] get({session_id})={raw_ctx}\n")
    sys.stderr.flush()
    raw_ctx2 = redis_mgr.get(f"context:{session_id}")
    sys.stderr.write(f"[ToolExecutor] get(context:{session_id})={raw_ctx2}\n")
    sys.stderr.flush()
    
    # 获取 knowledge 目录文件列表（只扫描一次）
    knowledge_files = None

    # 从 input_schema 获取所有需要的参数名，遍历检查
    required_params = list(input_schema.keys()) if input_schema else list(tool_inputs.keys())
    goal = action.get("goal", "")
    for param_name in required_params:
        param_value = tool_inputs.get(param_name, "")
        # 如果参数已有有效值，跳过
        if param_value and str(param_value).strip():
            continue
        
        sys.stderr.write(f"[ToolExecutor] 参数 {param_name} 为空，开始查找...\n")
        sys.stderr.flush()
        
        param_info = input_schema.get(param_name, {})
        param_desc = param_info.get("description", param_name)
        param_type = param_info.get("type", "string")
        param_meta = {"type": param_type, "description": param_desc}
        
        if param_type == "file":
            # file 类型：扫描 knowledge 目录选择文件
            if knowledge_files is None:
                knowledge_files = _scan_knowledge_dir()
            
            if knowledge_files:
                selected_path = await _llm_select_file(
                    llm, knowledge_files, scene_description, param_name, param_desc
                )
                if selected_path:
                    sys.stderr.write(f"[ToolExecutor] LLM选择文件: {param_name}={selected_path}\n")
                    sys.stderr.flush()
                    tool_inputs[param_name] = selected_path
                    continue
        else:
            # 非 file 类型：从结构化上下文中 LLM 提取
            context = redis_mgr.get_context(session_id)
            sys.stderr.write(f"[ToolExecutor] 结构化上下文: {'有' if context else '空'}\n")
            sys.stderr.flush()
            if context and llm:
                extracted_value = await _llm_extract_from_context(
                    llm, param_name, param_desc, param_type, context, goal
                )
                if extracted_value:
                    sys.stderr.write(f"[ToolExecutor] LLM提取: {param_name}={extracted_value}\n")
                    sys.stderr.flush()
                    tool_inputs[param_name] = extracted_value
                    continue
                else:
                    sys.stderr.write(f"[ToolExecutor] LLM提取失败: {param_name}\n")
                    sys.stderr.flush()
            elif not context:
                sys.stderr.write(f"[ToolExecutor] 无结构化上下文\n")
                sys.stderr.flush()
            elif not llm:
                sys.stderr.write(f"[ToolExecutor] LLM不可用\n")
                sys.stderr.flush()

    # 规范化参数值（JSON对象 → 坐标、数字等），再做缺失检查
    if input_schema:
        for p, pinfo in input_schema.items():
            v = tool_inputs.get(p, "")
            if v and str(v).strip():
                tool_inputs[p] = _normalize_param_value(v, pinfo.get("type", "string"), pinfo.get("format", ""))

    # 检查必需参数是否齐全：input_schema 中仍为空的参数，不允许带空值调用算法
    if input_schema:
        missing = [p for p in input_schema if not str(tool_inputs.get(p, "")).strip()]
        if missing:
            names = "、".join(f"{p}({input_schema[p].get('description', p)})" for p in missing)
            sys.stderr.write(f"[ToolExecutor] 缺少必需参数: {names}\n")
            sys.stderr.flush()
            return {
                "success": False,
                "output": "",
                "error": f"缺少必需参数: {names}",
                "pending": False,
            }

    # 所有参数就绪，执行工具
    try:
        sys.stderr.write(f"[ToolExecutor] 执行工具 {tool_name}，参数: {tool_inputs}\n")
        sys.stderr.flush()
        
        result = await tool.ainvoke(tool_inputs)
        result_text = str(result)
        sys.stderr.write(f"[ToolExecutor] tool.ainvoke 返回: {result_text[:500]}\n")
        sys.stderr.flush()

        # 将算法返回结果追加到消息历史，让 LLM 后续能看到算法输出
        try:
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=result_text, tool_call_id=f"{tool_name}_{len(messages)}"))
        except Exception:
            pass

        # MCP 工具返回 content 列表，内嵌 JSON：{"returncode": ..., "stdout": ..., "stderr": ...}
        returncode = None
        try:
            for item in result if isinstance(result, list) else []:
                if isinstance(item, dict) and item.get("type") == "text":
                    inner = json.loads(item.get("text", ""))
                    returncode = inner.get("returncode")
                    if returncode is not None:
                        break
        except Exception:
            pass

        if returncode is not None and returncode != 0:
            return {
                "success": False,
                "output": result_text,
                "error": f"算法 {tool_name} 执行失败（returncode={returncode}）: {result_text[:500]}",
                "pending": False,
            }

        return {
            "success": True,
            "output": result_text,
            "error": "",
            "pending": False,
        }
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] tool.ainvoke 异常: {e}\n")
        sys.stderr.flush()
        return {
            "success": False,
            "output": "",
            "error": f"工具执行失败: {str(e)}",
            "pending": False,
        }
