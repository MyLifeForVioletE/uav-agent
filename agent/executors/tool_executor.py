"""
工具执行器：执行算法工具调用
包含参数查找逻辑：Redis缓存 → knowledge目录扫描 → LLM选择 → 询问用户
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from core.config import BASE_DIR
from core.redis_manager import get_redis_manager


# knowledge 目录支持的文件类型
KNOWLEDGE_EXTENSIONS = {".txt", ".json", ".ldf"}

# LLM 单次调用超时（秒）：Ollama 无响应/排队时兜底，避免永久挂起
LLM_CALL_TIMEOUT = 60.0


async def _ainvoke_with_timeout(llm, prompt, timeout: float = LLM_CALL_TIMEOUT):
    """
    带超时的 LLM 调用：超时抛出 TimeoutError，由调用方决定降级策略。
    返回 LLM 响应的 content 字符串。
    """
    from langchain_core.messages import HumanMessage
    response = await asyncio.wait_for(
        llm.ainvoke([HumanMessage(content=prompt)]),
        timeout=timeout,
    )
    return response.content.strip()


def _is_coord_param(param_name: str, param_desc: str, param_type: str) -> bool:
    """判断参数是否为坐标类（名/描述含坐标词，或类型为 tuple 坐标、format=comma/space）"""
    text = f"{param_name} {param_desc}".lower()
    if param_type == "tuple":
        return True
    if any(k in text for k in ("坐标", "位置", "经纬度", "longitude", "latitude", "coord", "x,y", "x、y")):
        return True
    return False


def _group_to_coord_str(group) -> Optional[str]:
    """把单个分组（dict 或 dict 列表）中的 Longitude/Latitude 拼成 x,y 坐标；找不到返回 None"""
    if not isinstance(group, list):
        group = [group]
    for item in group:
        if not isinstance(item, dict):
            continue
        lon = (item.get("Longitude") or item.get("longitude")
               or item.get("lon") or item.get("Lon") or item.get("x"))
        lat = (item.get("Latitude") or item.get("latitude")
               or item.get("lat") or item.get("Lat") or item.get("y"))
        if lon is not None and lat is not None and str(lon).strip() and str(lat).strip():
            return f"{lon},{lat}"
    return None


def _local_resolve_param(context: dict, param_name: str, param_desc: str, param_type: str) -> Optional[str]:
    """
    LLM 不可用/超时后的本地降级：键值直取 → 语义启发式 → 坐标按分组拼接。
    返回候选值字符串（坐标转为 x,y 形式），找不到返回 None。
    """
    if not context:
        return None

    def to_coord_str(v):
        s = str(v).strip()
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    lon = (obj.get("Longitude") or obj.get("longitude")
                           or obj.get("lon") or obj.get("Lon") or obj.get("x"))
                    lat = (obj.get("Latitude") or obj.get("latitude")
                           or obj.get("lat") or obj.get("Lat") or obj.get("y"))
                    if lon is not None and lat is not None:
                        return f"{lon},{lat}"
            except Exception:
                pass
        return s

    pname_lower = param_name.lower()

    is_coord = _is_coord_param(param_name, param_desc, param_type)

    # 1) 键名直取：上下文各分组里与参数名完全同名的字段
    for group in context.values():
        if not isinstance(group, list):
            group = [group]
        for item in group:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if str(k).lower() == pname_lower and str(v).strip():
                    return to_coord_str(v)

    # 2) 坐标参数：按语义分组拼接 Longitude/Latitude（起点→uav/base 分组；终点→target/base 分组）
    desc_lower = (param_desc or "").lower()
    is_start = any(k in pname_lower or k in desc_lower for k in ("start", "起点", "from", "当前位置", "开始", "uav", "outpoint"))
    is_end = any(k in pname_lower or k in desc_lower for k in ("dest", "终点", "目标", "target", "end", "to"))
    if is_coord:
        if is_end:
            end_candidates = []
            target_g = context.get("targets")
            base_g = context.get("base")
            for g in (target_g, base_g):
                if g:
                    c = _group_to_coord_str(g)
                    if c:
                        end_candidates.append(c)
            # 终点优先 target；无法区分时返回第一个候选，否则返回 None 交给更上层
            if end_candidates:
                return end_candidates[0]
        # 起点：UAV 分组优先，其次 base
        for gname in ("uavs", "uav", "base"):
            g = context.get(gname)
            if g:
                c = _group_to_coord_str(g)
                if c:
                    return c
        # 兜底：任一分组含 kind = "start" 类坐标
        for gname, g in context.items():
            if any(k in str(gname).lower() for k in ("start", "uav", "pos", "航")):
                c = _group_to_coord_str(g)
                if c:
                    return c

    # 3) 语义启发式：按参数含义匹配单个键
    aliases = {
        "start": ["start", "起点", "startposition", "start_position", "uavposition", "uav_position", "当前位置", "uav", "from"],
        "end": ["end", "终点", "endposition", "end_position", "targetposition", "target_position", "目标", "target", "base", "baseposition"],
        "frequency": ["frequency", "频率", "freq", "target_frequency"],
        "bandwidth": ["bandwidth", "带宽", "bw"],
        "signal": ["signal_strength", "信号强度", "signal", "场强"],
        "power": ["power", "功率", "w", "dbm"],
    }
    for kind, keys in aliases.items():
        matched = False
        if pname_lower in keys:
            matched = True
        if any(k in desc_lower for k in keys):
            matched = True
        if not matched:
            continue
        for group in context.values():
            if not isinstance(group, list):
                group = [group]
            for item in group:
                if not isinstance(item, dict):
                    continue
                for k, v in item.items():
                    if str(k).lower() in keys and str(v).strip():
                        return to_coord_str(v)

    return None


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
    # 数值型参数带频率单位（GHz/MHz/kHz/Hz）时，统一换算为 MHz（与算法 exe 约定单位一致）
    if param_type == "number":
        converted = _convert_frequency_to_mhz(v)
        if converted is not None:
            return converted
    return value


def _convert_frequency_to_mhz(value: str):
    """将带单位频率字符串换算为 MHz 数值字符串；纯数字视为已是 MHz 返回；未知单位返回 None"""
    import re
    s = value.strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)?$", s)
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("ghz", "g"):
        mhz = num * 1000
    elif unit in ("mhz", "m"):
        mhz = num
    elif unit in ("khz", "kh", "k"):
        mhz = num / 1000.0
    elif unit in ("hz", "h"):
        mhz = num / 1000000.0
    elif unit == "":
        return s  # 纯数字，按约定视为 MHz
    else:
        return None  # 未知单位（如 W/dBm），不强行换算
    return str(int(mhz)) if mhz == int(mhz) else str(mhz)


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
        response = await _ainvoke_with_timeout(llm, prompt)
        selected = response.strip()

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
    except asyncio.TimeoutError:
        sys.stderr.write(f"[ToolExecutor] LLM选择文件超时({LLM_CALL_TIMEOUT}s)，降级为无选定文件: {param_name}\n")
        sys.stderr.flush()
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
        response = await _ainvoke_with_timeout(llm, prompt)
        value = response.strip()

        if value == "NONE" or not value:
            return None

        # 去除可能的引号包裹
        value = value.strip('"').strip("'")
        return value
    except asyncio.TimeoutError:
        sys.stderr.write(f"[ToolExecutor] LLM文件取值超时({LLM_CALL_TIMEOUT}s): {param_name}\n")
        sys.stderr.flush()
        return None
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] LLM提取值失败: {e}\n")
        sys.stderr.flush()
        return None


def _resolve_context_path(context: dict, path: str) -> Optional[str]:
    """按点号路径从上下文对象中取真实值；数组支持数字索引或 id 定位。

    值一律来自上下文原文（LLM 只负责给路径，禁止编造数值）；
    值缺失或为空字符串返回 None。
    """
    if not context or not path:
        return None
    parts = [p.strip() for p in path.strip().strip('"').strip("'").split(".") if p.strip()]
    current = context
    for p in parts:
        if isinstance(current, list):
            idx = None
            if p.isdigit():
                idx = int(p)
            else:
                for i, item in enumerate(current):
                    if isinstance(item, dict) and str(item.get("id")) == p:
                        idx = i
                        break
            if idx is None or not (0 <= idx < len(current)):
                return None
            current = current[idx]
        elif isinstance(current, dict) and p in current:
            current = current[p]
        else:
            return None
    if current is None:
        return None
    if isinstance(current, str):
        return current.strip() or None
    if isinstance(current, (dict, list)):
        s = json.dumps(current, ensure_ascii=False)
        return s if s not in ("{}", "[]") else None
    return str(current)


async def _llm_extract_from_context(llm, param_name: str, param_desc: str,
                                     param_type: str, context: dict, goal: str = "") -> Optional[str]:
    """
    用 LLM 从结构化上下文中定位与参数语义最匹配的字段路径，
    值一律从上下文原文读取（禁止 LLM 自行生成/编造数值）。
    Returns: 上下文中的真实值，或 None
    """
    if not llm or not context:
        sys.stderr.write(f"[ToolExecutor] _llm_extract_from_context: llm={llm is not None}, context={'有' if context else '空'}\n")
        sys.stderr.flush()
        return None
    
    context_text = json.dumps(context, ensure_ascii=False, indent=2)
    
    sys.stderr.write(f"[ToolExecutor] LLM定位键: {param_name}({param_desc}), goal={goal[:50] if goal else ''}, 上下文有 {len(context)} 个分组\n")
    sys.stderr.flush()
    
    goal_hint = f"\n动作目标：{goal}\n根据动作目标理解该参数的含义。" if goal else ""
    
    prompt = f"""在结构化上下文中找到与指定参数语义最匹配的字段。
{goal_hint}
结构化上下文：
{context_text}

需要匹配的参数：
- 参数名: {param_name}
- 参数描述: {param_desc}
- 参数类型: {param_type}

【要求】
1. 只输出该字段的完整 JSON 点号路径，禁止输出字段的值或任何编造的数据。
2. 路径格式：数组元素用数字索引或 id 定位（如 targets.0.Frequency 或 targets.T1.Frequency），
   其他逐级用点号拼接（如 base.Longitude、uavs.UAV_1.Latitude）。
3. 若上下文中不存在与参数语义匹配的字段，输出 "NONE"。

只输出一行路径或 "NONE"，不要输出其他任何内容。"""

    try:
        from langchain_core.messages import HumanMessage
        response = await _ainvoke_with_timeout(llm, prompt)
        path = response.strip().strip('"').strip("'")
        
        sys.stderr.write(f"[ToolExecutor] LLM键路径: '{path}'\n")
        sys.stderr.flush()
        
        if not path or path.upper() == "NONE":
            return None
        
        value = _resolve_context_path(context, path)
        if value is None:
            sys.stderr.write(f"[ToolExecutor] 路径未命中或值为空: '{path}'\n")
            sys.stderr.flush()
            return None
        
        sys.stderr.write(f"[ToolExecutor] 命中 {path} = {str(value)[:80]}\n")
        sys.stderr.flush()
        return value
    except asyncio.TimeoutError:
        sys.stderr.write(f"[ToolExecutor] LLM定位键超时({LLM_CALL_TIMEOUT}s): {param_name}\n")
        sys.stderr.flush()
        return None
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] LLM定位键失败: {e}\n")
        sys.stderr.flush()
        return None


def _param_placeholder_names(context: dict, tool_inputs: dict) -> set:
    """识别应保持为空占位的参数字段。

    规则：某一参数名若在原上下文中已存在，且其值为空占位
    （由不执行 exe 的算法建立，如扫频侦察的 sweepData），则后续无论 LLM
    如何猜测，都不允许用无关上下文数据（如航迹点）覆盖，保持"只留参数名"。

    Returns: 应清空为占位参数名的参数名集合。
    """
    if not context:
        return set()
    pnames = {k for k in tool_inputs if str(k).strip()}
    matched = set()
    for group in context.values():
        if not isinstance(group, list):
            group = [group]
        for item in group:
            if not isinstance(item, dict):
                continue
            for k in pnames:
                if k in item and not str(item[k] or "").strip():
                    matched.add(k)
    return matched


def _write_output_placeholders(redis_mgr, session_id: str, tool_name: str, output_schema: dict):
    """按算法的输出 schema 在 Redis 上下文中建立输出占位字段。

    即使算法未真实执行（值暂为空），也保证后续任务可按输出名读取到该字段。
    """
    if not output_schema or not redis_mgr:
        return
    session_id = session_id or "default"
    try:
        context = redis_mgr.get_context(session_id) or {}
        targets = context.get("targets")
        if isinstance(targets, list) and targets:
            # 输出归属到第一个目标条目（多数算法输出是对目标的分析结果）
            update = {"targets": {0: {}}}
            for fname in output_schema:
                # 已存在的输出名保留原值，否则置为空字符串作为占位
                update["targets"][0][fname] = targets[0].get(fname, "")
        else:
            update = {}
            for fname in output_schema:
                update[fname] = context.get(fname, "")
        redis_mgr.update_context(session_id, update)
        sys.stderr.write(f"[ToolExecutor] 已建立输出占位: {tool_name} => {json.dumps(update, ensure_ascii=False)}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] 输出占位写入失败: {e}\n")
        sys.stderr.flush()


def _parse_waypoints(stdout: str) -> list:
    """从路径规划 stdout 解析航迹点列表。

    输出形如 (lng,lat,alt,time),(lng,lat,alt,time),... ，每个四元组：
    经度, 纬度, 高度, 时间。
    """
    if not stdout:
        return []
    import re as _re
    waypoints = []
    for m in _re.finditer(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", stdout):
        try:
            waypoints.append({
                "Longitude": float(m.group(1)),
                "Latitude": float(m.group(2)),
                "Altitude": float(m.group(3)),
                "Time": float(m.group(4)),
            })
        except (ValueError, TypeError):
            continue
    return waypoints


def _extract_output_fields(output: str) -> list:
    """从 stub 占位输出（"输出;field1, field2"）提取输出字段名列表；非 stub 输出返回空列表。

    stub 工具不调用 exe，stdout 仅为 "输出;field" 字段名列表（值未知），
    字段名直接交给上层使用，无需保留 "输出;" 前缀。
    """
    import re as _re
    m = _re.fullmatch(r"输出;[\s,，、]*(.*)", str(output or "").strip())
    if not m:
        return []
    return _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1))


def _write_path_to_redis(redis_mgr, session_id: str, state: dict, tool_inputs: dict, output: str):
    """路径规划成功后，将航迹点写入 Redis 上下文（关联到对应的无人机）。

    记录该航迹由哪架无人机、从哪个起点到哪个终点。
    """
    if not redis_mgr:
        return
    session_id = session_id or "default"
    waypoints = _parse_waypoints(str(output or ""))
    if not waypoints:
        sys.stderr.write("[ToolExecutor] 未从路径规划输出中解析到航迹点\n")
        sys.stderr.flush()
        return
    try:
        agent_id = (state or {}).get("_agent_id") or "UAV_1"
        # 起点/终点：优先取经规范化的 tool_inputs 坐标，其次尝试逗号拆分的组合字段
        start_lon = str((tool_inputs.get("startPositionLong") or "")).strip()
        start_lat = str((tool_inputs.get("startPositionLat") or "")).strip()
        end_lon = str((tool_inputs.get("destinationLong") or "")).strip()
        end_lat = str((tool_inputs.get("destinationLat") or "")).strip()
        # 去除可能残留的括号
        start_lon = start_lon.strip("()").strip()
        start_lat = start_lat.strip("()").strip()
        end_lon = end_lon.strip("()").strip()
        end_lat = end_lat.strip("()").strip()

        route = {
            "agent": agent_id,
            "from_lon": start_lon,
            "from_lat": start_lat,
            "to_lon": end_lon,
            "to_lat": end_lat,
            "waypoints": waypoints,
        }
        # 直接构建完整最新上下文并整体保存，避免 update_context 按 id 深合并时丢弃新增 uav
        ctx = redis_mgr.get_context(session_id) or {}
        uavs = ctx.get("uavs")
        if not isinstance(uavs, list):
            uavs = []
            ctx["uavs"] = uavs
        for u in uavs:
            if isinstance(u, dict) and u.get("id") == agent_id:
                u.update({"Longitude": start_lon, "Latitude": start_lat, "path": route})
                break
        else:
            uavs.append({"id": agent_id, "Longitude": start_lon, "Latitude": start_lat,
                         "path": route})
        redis_mgr.save_context(session_id, ctx)
        sys.stderr.write(
            f"[ToolExecutor] 已写入航迹: {agent_id} "
            f"[{start_lon},{start_lat}] -> [{end_lon},{end_lat}] 共{len(waypoints)}个航迹点\n"
        )
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] 航迹写入 Redis 失败: {e}\n")
        sys.stderr.flush()


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
    tool_inputs = dict(action.get("tool_inputs", {}))

    # 是否需真实调用 exe：只有路径规划算法调用，其余算法模拟执行
    real_exec = tool_name == "path_planning"

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
    output_schema = algo_config.get("output_schema", {})
    
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

    # 空占位保护：上下文中已存在且值为空的参数字段（如未执行 exe 的扫频侦察建立的
    # sweepData 占位），不允许被 LLM 用无关上下文数据（航迹点等）填充，保持"只留参数名"。
    if not real_exec:
        ctx_for_placeholder = redis_mgr.get_context(session_id)
        placeholder_params = _param_placeholder_names(ctx_for_placeholder, tool_inputs)
        if placeholder_params:
            for p in placeholder_params:
                tool_inputs[p] = p  # 只留参数名
            sys.stderr.write(f"[ToolExecutor] 空占位保护: {tool_name} 参数 {sorted(placeholder_params)} 保持为参数名占位\n")
            sys.stderr.flush()

    for param_name in required_params:
        param_value = tool_inputs.get(param_name, "")
        # 空占位保护已把该参数置为参数名（值为参数名），跳过查找
        if str(param_value).strip() == param_name:
            continue
        # 非 exe 算法：LLM 猜测的值不可信（可能是参数描述/编造文本），
        # 统一清空后走 Redis 查找，保证值一定来自上下文原文。
        if param_value and str(param_value).strip():
            if real_exec:
                continue  # 真实 exe：已有值直接用
            sys.stderr.write(f"[ToolExecutor] 非exe 清除 LLM 猜测值: {param_name}={param_value}\n")
            sys.stderr.flush()
            tool_inputs[param_name] = ""

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
                elif not real_exec:
                    # 不执行 exe 的算法：LLM 未查到值属于正常现象，不需本地降级，
                    # 直接留空，交给末尾"非 exe 缺参用参数名占位"处理（写脚本时显示参数名）。
                    sys.stderr.write(f"[ToolExecutor] LLM提取失败(非exe)，{param_name} 保持为空，交由参数名占位\n")
                    sys.stderr.flush()
                else:
                    # 真实 exe 算法（如 path_planning）：LLM 超时/失败/返回 NONE
                    # → 本地降级（键值直取 + 语义启发式），仍缺才进入 missing_params
                    local_value = _local_resolve_param(context, param_name, param_desc, param_type)
                    if local_value:
                        sys.stderr.write(f"[ToolExecutor] LLM提取失败，本地降级补齐: {param_name}={local_value}\n")
                        sys.stderr.flush()
                        tool_inputs[param_name] = local_value
                        continue
                    sys.stderr.write(f"[ToolExecutor] LLM提取失败且本地无法补齐: {param_name}\n")
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

    # 检查必需参数是否齐全
    if input_schema:
        missing = [p for p in input_schema if not str(tool_inputs.get(p, "")).strip()]
        if missing and not real_exec:
            # 非真实执行的算法：缺参时用参数名填充，保证能记录到脚本与 Redis 输出占位
            sys.stderr.write(f"[ToolExecutor] {tool_name} 非 exe 执行，缺参用参数名填充: {missing}\n")
            sys.stderr.flush()
            for p in missing:
                tool_inputs[p] = p
            missing = []
        if missing:
            names = "、".join(f"{p}({input_schema[p].get('description', p)})" for p in missing)
            sys.stderr.write(f"[ToolExecutor] 缺少必需参数: {names}\n")
            sys.stderr.flush()
            missing_params = [
                {
                    "name": p,
                    "description": input_schema[p].get("description", p),
                    "type": input_schema[p].get("type", "string"),
                }
                for p in missing
            ]
            return {
                "success": False,
                "output": "",
                "error": f"缺少必需参数: {names}",
                "missing_params": missing_params,
                "tool_inputs": tool_inputs,
                "pending": False,
            }

    # 所有参数就绪，执行工具
    # 无论是否真实调用 exe，都根据该算法的输出 schema 在 Redis 中建立输出占位字段（值暂为空）
    _write_output_placeholders(redis_mgr, session_id, tool_name, output_schema)
    try:
        sys.stderr.write(f"[ToolExecutor] 执行工具 {tool_name}，参数: {tool_inputs}\n")
        sys.stderr.flush()
        
        result = await tool.ainvoke(tool_inputs)
        result_text = str(result)
        sys.stderr.write(f"[ToolExecutor] tool.ainvoke 返回: {result_text[:500]}\n")
        sys.stderr.flush()

        # MCP 工具返回 content 列表，内嵌 JSON：{"returncode": ..., "stdout": ..., "stderr": ...}
        # 解析出可读的 stdout 作为脚本输出（兼容 dict / Pydantic 对象）
        returncode = None
        readable = ""
        inner_text = ""
        try:
            for item in result if isinstance(result, (list, tuple)) else [result]:
                if isinstance(item, dict):
                    item_text = item.get("text", "")
                elif hasattr(item, "text"):
                    item_text = getattr(item, "text", "")
                else:
                    item_text = ""
                if item_text:
                    inner_text = str(item_text)
                    break
        except Exception:
            pass

        import re as _re
        try:
            if inner_text:
                parsed = json.loads(inner_text)
                if isinstance(parsed, dict):
                    if parsed.get("returncode") is not None:
                        returncode = parsed.get("returncode")
                    out = parsed.get("stdout", "")
                    if out:
                        readable = out if isinstance(out, str) else str(out)
            if not readable:
                # 兜底：从原始文本中正则提取 "stdout": "xxx"
                m = _re.search(r'"stdout"\s*:\s*"((?:[^"\\]|\\.)*)"', inner_text or result_text)
                if m:
                    readable = m.group(1).encode().decode("unicode_escape", errors="ignore")
        except Exception:
            pass
        if not readable:
            readable = result_text

        # 将算法返回结果追加到消息历史，让 LLM 后续能看到算法输出
        try:
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=result_text, tool_call_id=f"{tool_name}_{len(messages)}"))
        except Exception:
            pass

        if returncode is not None and returncode != 0:
            return {
                "success": False,
                "output": readable,
                "error": f"算法 {tool_name} 执行失败（returncode={returncode}）: {readable[:500]}",
                "tool_inputs": tool_inputs,
                "pending": False,
            }

        # 路径规划成功：把航迹点写入 Redis（关联无人机 + 起终点），并同步到脚本输出
        if real_exec and tool_name == "path_planning":
            _write_path_to_redis(redis_mgr, session_id, state, tool_inputs, readable)

        return {
            "success": True,
            "output": readable,
            "output_fields": _extract_output_fields(readable),
            "tool_inputs": tool_inputs,
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
