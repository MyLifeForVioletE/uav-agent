"""
工具执行器：执行算法工具调用
包含参数查找逻辑：Redis缓存 → knowledge目录扫描 → LLM选择 → 询问用户
"""
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

from core.config import BASE_DIR
from core.ollama_utils import producer_algos_for_field
from core.redis_manager import get_redis_manager
from core.timing import timing


# knowledge 目录支持的文件类型
KNOWLEDGE_EXTENSIONS = {".txt", ".json", ".ldf"}

# LLM 单次调用超时（秒）：Ollama 无响应/排队时兜底，避免永久挂起
LLM_CALL_TIMEOUT = 60.0

# 文件选择缓存：同一 session 内 knowledge 文件不变，避免重复调 LLM
# key = (param_name, param_desc) → value = 文件路径
_file_select_cache: dict[tuple, str] = {}


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
    if any(k in text for k in ("坐标", "位置", "经纬度", "经度", "纬度", "longitude", "latitude", "coord", "x,y", "x、y")):
        return True
    return False


def _pick_coord_component(value, param_name: str, param_desc: str) -> str:
    """坐标分量拆分：参数为经度/纬度单分量（如 destinationLat），而值却是 "x,y" 坐标对时，
    取对应分量（经度→第一个，纬度→第二个），防止 "70,15.02" 被当作单值纬度传给 exe。
    参数不是经纬度单分量或值不含坐标对时，原样返回。"""
    v = str(value or "").strip()
    m = re.fullmatch(r"[\(\[]?\s*(-?\d+(?:\.\d+)?)\s*(?:[,，]\s*|\s+)(-?\d+(?:\.\d+)?)\s*[\)\]]?", v)
    if not m:
        return v
    text = f"{param_name} {param_desc}".lower()
    is_lon = any(k in text for k in ("经度", "longitude"))
    is_lat = any(k in text for k in ("纬度", "latitude"))
    if is_lon:
        return m.group(1)
    if is_lat:
        return m.group(2)
    return v


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


def _local_resolve_param(context: dict, param_name: str, param_desc: str, param_type: str,
                         agent_id: str = "", goal: str = "", scene_text: str = "") -> Optional[str]:
    """
    LLM 不可用/超时后的本地降级：键值直取 → 语义启发式 → 坐标按分组拼接。
    返回候选值字符串（坐标转为 x,y 形式，单分量坐标参数返回对应分量），找不到返回 None。
    agent_id: 执行该动作的无人机 id（用于解析观测位置 mission_position）
    goal: 原子动作目标文本（用于区分出航/返航语义）
    scene_text: 子任务目标原文（含具体坐标，如"飞行至位置70,15.01..."）
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
    desc_lower = (param_desc or "").lower()

    is_coord = _is_coord_param(param_name, param_desc, param_type)

    # 坐标分量识别：参数语义是经度还是纬度（destinationLong/起点经度 → 取经度分量）
    comp_text = f"{pname_lower} {desc_lower}"
    is_lon = any(k in comp_text for k in ("经度", "longitude"))
    is_lat = any(k in comp_text for k in ("纬度", "latitude"))

    def pick_component(coord: str) -> str:
        parts = re.split(r"[,，]", coord)
        if len(parts) == 2:
            if is_lon:
                return parts[0].strip()
            if is_lat:
                return parts[1].strip()
        return coord

    # 1) 键名直取：上下文各分组里与参数名完全同名的字段
    for group in context.values():
        if not isinstance(group, list):
            group = [group]
        for item in group:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if str(k).lower() == pname_lower and str(v).strip():
                    return _freq_component(pick_component(to_coord_str(v)), pname_lower, desc_lower)

    # 1.5) 别名/后缀匹配：参数名与字段名互为后缀（如 step ↔ frequency_step、
    #      targetMinFreq ↔ minFreq）。值一律来自上下文原文，不编造；
    #      仅当较短一侧长度 >=3 时启用，避免过短关键词误匹配。
    for group in context.values():
        if not isinstance(group, list):
            group = [group]
        for item in group:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                kl = str(k).lower().strip()
                if not kl or not str(v).strip():
                    continue
                if kl == pname_lower:
                    continue
                if (kl.endswith(pname_lower) or pname_lower.endswith(kl)) \
                        and min(len(kl), len(pname_lower)) >= 3:
                    return _freq_component(pick_component(to_coord_str(v)), pname_lower, desc_lower)

    # 2) 坐标参数：按语义分组拼接 Longitude/Latitude（起点→uav/base 分组；终点→目标原文/观测位置/target/base）
    is_start = any(k in pname_lower or k in desc_lower for k in ("start", "起点", "from", "当前位置", "开始", "uav", "outpoint"))
    is_end = any(k in pname_lower or k in desc_lower for k in ("dest", "终点", "目标", "target", "end", "to"))
    goal_lower = (goal or "").lower()
    is_return = any(k in goal_lower for k in ("返航", "返回", "回基地", "起始点", "home", "return")) or \
                any(k in pname_lower or k in desc_lower for k in ("返航", "返回", "回基地", "起始点"))
    if is_coord:
        # 子任务目标原文中的坐标：取最后一个坐标对（任务 goal 通常以"飞至位置X,Y"结尾）
        goal_coord = None
        if scene_text:
            m = re.findall(r"(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)", scene_text)
            if m:
                goal_coord = f"{m[-1][0]},{m[-1][1]}"
        # 执行无人机的观测位置（mission_position）
        mp_coord = None
        if agent_id:
            for g in (context.get("uavs"), context.get("uav")):
                if not isinstance(g, list):
                    g = [g]
                for u in g:
                    if isinstance(u, dict) and u.get("id") == agent_id:
                        mp = str(u.get("mission_position") or "").strip()
                        parts = re.split(r"[,，]", mp)
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            mp_coord = f"{parts[0].strip()},{parts[1].strip()}"
                        break

        if is_end:
            if not is_return:
                for c in (goal_coord, mp_coord):
                    if c:
                        return pick_component(c)
            # 返航优先回基地；出航兜底 target → base
            groups = (context.get("base"), context.get("targets")) if is_return \
                else (context.get("targets"), context.get("base"))
            for g in groups:
                if g:
                    c = _group_to_coord_str(g)
                    if c:
                        return pick_component(c)
            return None
        # 起点：执行无人机的当前坐标优先，其次任意 UAV / base
        if agent_id:
            for g in (context.get("uavs"), context.get("uav")):
                if not isinstance(g, list):
                    g = [g]
                for u in g:
                    if isinstance(u, dict) and u.get("id") == agent_id:
                        c = _group_to_coord_str([u])
                        if c:
                            return pick_component(c)
        for gname in ("uavs", "uav", "base"):
            g = context.get(gname)
            if g:
                c = _group_to_coord_str(g)
                if c:
                    return pick_component(c)
        # 兜底：任一分组含 kind = "start" 类坐标
        for gname, g in context.items():
            if any(k in str(gname).lower() for k in ("start", "uav", "pos", "航")):
                c = _group_to_coord_str(g)
                if c:
                    return pick_component(c)

    # 3) 语义启发式：按参数含义匹配单个键
    aliases = {
        "start": ["start", "起点", "startposition", "start_position", "uavposition", "uav_position", "当前位置", "uav", "from"],
        "end": ["end", "终点", "endposition", "end_position", "targetposition", "target_position", "目标", "target", "base", "baseposition"],
        "min_frequency": ["minfreq", "最低频率", "最小频率", "min_frequency", "下限"],
        "max_frequency": ["maxfreq", "最高频率", "最大频率", "max_frequency", "上限"],
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
                        # 频率范围兜底：min/max 频率参数命中范围字符串（如旧 Frequency="2~3GHz"）时
                        # 拆分出对应分量，防止整个范围被当作单值传给算法。
                        return _freq_component(to_coord_str(v), pname_lower, desc_lower)

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


def _split_freq_range(value: str):
    """将频率范围字符串（如 "2~3GHz"、"1GHz到2GHz"、"1-2GHz"）拆成 (最低, 最高)；
    单位缺失的一侧继承另一侧单位（"1~2GHz" → ("1GHz","2GHz")）；非范围返回 None。"""
    import re as _re
    s = str(value or "").strip()
    m = _re.fullmatch(r"\s*([^~～至到－\-]+?)\s*[~～至到－\-]\s*([^~～至到－\-]+?)\s*", s)
    if not m:
        return None
    lo, hi = m.group(1).strip(), m.group(2).strip()

    def has_unit(x):
        return bool(_re.search(r"[a-zA-Zμµ]", x))

    if has_unit(hi) and not has_unit(lo):
        lo = lo + _re.search(r"[a-zA-Zμµ]+", hi).group()
    elif has_unit(lo) and not has_unit(hi):
        hi = hi + _re.search(r"[a-zA-Zμµ]+", lo).group()
    return lo, hi


def _freq_component(value, param_name: str, param_desc: str):
    """min/max 频率参数命中范围字符串时拆出对应分量（min→下限，max→上限）；否则原样返回。"""
    if not value:
        return value
    parts = _split_freq_range(value)
    if not parts:
        return value
    text = f"{param_name} {param_desc}".lower()
    if any(k in text for k in ("maxfreq", "最高频率", "最大频率", "max_frequency", "上限")):
        return parts[1]
    if any(k in text for k in ("minfreq", "最低频率", "最小频率", "min_frequency", "下限")):
        return parts[0]
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
    调用 LLM 从文件列表中选择最合适的文件。
    带 session 级缓存（knowledge 文件不变）和重试（Windows socket 瞬时耗尽恢复）。
    Returns: 选中的文件路径，或 None
    """
    global _file_select_cache
    if not files or not llm:
        return None
    
    # 缓存命中：同一参数名+描述 → 同一文件（knowledge 目录不变）
    cache_key = (param_name, param_desc)
    if cache_key in _file_select_cache:
        cached_path = _file_select_cache[cache_key]
        sys.stderr.write(f"[ToolExecutor] LLM选择文件(缓存): {param_name}={cached_path}\n")
        sys.stderr.flush()
        return cached_path
    
    file_list_text = _build_file_list_text(files)
    
    prompt = f"""你是一个文件选择助手。根据场景描述和参数需求，从文件列表中选择最合适的文件。

场景描述：{scene_description}

需要填写的参数：{param_name}
参数描述：{param_desc}

可用文件列表：
{file_list_text}

请只输出选中的文件名（不包含路径），如果无法确定则输出 "NONE"。"""

    # 重试：Windows socket 瞬时耗尽（WSAENOBUFS）时短延迟后重试即可恢复
    max_retries = 3
    for attempt in range(max_retries):
        try:
            from langchain_core.messages import HumanMessage
            response = await _ainvoke_with_timeout(llm, prompt)
            selected = response.strip()

            # 验证选择
            if selected == "NONE" or not selected:
                return None
            
            # 检查选中的文件是否在列表中
            result_path = None
            for f in files:
                if f["name"] == selected:
                    result_path = f["path"]
                    break
            
            # 尝试模糊匹配
            if result_path is None:
                for f in files:
                    if selected in f["name"] or f["name"] in selected:
                        result_path = f["path"]
                        break
            
            if result_path:
                _file_select_cache[cache_key] = result_path
            return result_path
        except asyncio.TimeoutError:
            sys.stderr.write(f"[ToolExecutor] LLM选择文件超时({LLM_CALL_TIMEOUT}s)，降级为无选定文件: {param_name}\n")
            sys.stderr.flush()
            return None
        except OSError as e:
            # Windows socket 耗尽 (WSAENOBUFS) 等瞬时网络错误：短延迟后重试
            if attempt < max_retries - 1:
                delay = 0.5 * (attempt + 1)
                sys.stderr.write(f"[ToolExecutor] LLM选择文件网络错误(重试 {attempt+1}/{max_retries}): {e}\n")
                sys.stderr.flush()
                await asyncio.sleep(delay)
                continue
            sys.stderr.write(f"[ToolExecutor] LLM选择文件失败(已重试{max_retries}次): {e}\n")
            sys.stderr.flush()
            return None
        except Exception as e:
            err_str = str(e)
            # 含 socket/buffer/bind 关键词的也视为瞬时错误，重试
            is_transient = any(kw in err_str.lower() for kw in ("buffer", "socket", "bind", "eno", "connectionrefused", "reset"))
            if is_transient and attempt < max_retries - 1:
                delay = 0.5 * (attempt + 1)
                sys.stderr.write(f"[ToolExecutor] LLM选择文件瞬时错误(重试 {attempt+1}/{max_retries}): {e}\n")
                sys.stderr.flush()
                await asyncio.sleep(delay)
                continue
            sys.stderr.write(f"[ToolExecutor] LLM选择文件失败: {e}\n")
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
    return_hint = ""
    if goal and any(k in goal for k in ("返航", "返回", "回基地", "起始点")):
        return_hint = (
            "\n注意：该动作是【返航】，终点坐标必须取 base（基地）的 Longitude/Latitude，"
            "禁止取 uavs.*.path.to_lon/to_lat（那是上次出航的终点/目标点坐标）。"
        )
    
    prompt = f"""在结构化上下文中找到与指定参数语义最匹配的字段。
{goal_hint}
{return_hint}
结构化上下文：
{context_text}

需要匹配的参数：
- 参数名: {param_name}
- 参数描述: {param_desc}
- 参数类型: {param_type}

【要求】
1. 只输出该字段的完整 JSON 点号路径，禁止输出字段的值或任何编造的数据。
2. 路径格式：数组元素用数字索引或 id 定位（如 targets.0.minFreq 或 targets.T1.minFreq），
   其他逐级用点号拼接（如 base.Longitude、uavs.UAV_1.Latitude）。
3. 若上下文中不存在与参数语义匹配的字段，输出 "NONE"。

只输出一行路径或 "NONE"，不要输出其他任何内容。"""

    # 重试：Windows socket 瞬时耗尽时短延迟后重试
    max_retries = 3
    for attempt in range(max_retries):
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
        except (OSError, Exception) as e:
            err_str = str(e)
            is_transient = any(kw in err_str.lower() for kw in ("buffer", "socket", "bind", "eno", "connectionrefused", "reset"))
            if is_transient and attempt < max_retries - 1:
                delay = 0.5 * (attempt + 1)
                sys.stderr.write(f"[ToolExecutor] LLM定位键瞬时错误(重试 {attempt+1}/{max_retries}): {e}\n")
                sys.stderr.flush()
                await asyncio.sleep(delay)
                continue
            sys.stderr.write(f"[ToolExecutor] LLM定位键失败: {e}\n")
            sys.stderr.flush()
            return None


def _build_producer_action(algo_name: str, field: str) -> dict:
    """构造产出缺失字段的原子动作（参数留空，由执行器从上下文解析）"""
    return {
        "executor": "tool",
        "tool_name": algo_name,
        "tool_inputs": {},
        "action_name": f"{algo_name} 产出 {field}",
        "goal": f"调用算法 {algo_name} 产出缺失参数 {field}，供后续动作使用",
    }


def _find_self_producer(tools_map: dict, truly_missing: list, current_tool: str) -> dict | None:
    """在自身工具集中查找能产出缺失字段的算法。

    Returns: {"field": 字段名, "action": 产出动作} 或 None
    """
    for field in truly_missing:
        for aname in producer_algos_for_field(field):
            if aname == current_tool or aname not in tools_map:
                continue
            return {"field": field, "action": _build_producer_action(aname, field)}
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


def _placeholder_fields_in_context(context: dict, fields: list) -> set:
    """在结构化上下文中已存在且值为空的字段集合。

    用于缺参检查：上游非 exe 算法已执行并写入空占位（如扫频侦察的 sweepData=''），
    即使当前 tool_inputs 未携带该参数键，也应视为占位字段而非真正缺失。
    """
    matched = set()
    for f in fields:
        for group in context.values():
            if not isinstance(group, list):
                group = [group]
            for item in group:
                if isinstance(item, dict) and f in item and not str(item[f] or "").strip():
                    matched.add(f)
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


def _write_path_to_redis(redis_mgr, session_id: str, agent_id: str, tool_inputs: dict, output: str):
    """路径规划成功后，将航迹点写入 Redis 上下文（关联到对应的无人机）。

    按实体存储：目的地坐标不重复写进无人机航迹（那是 target/base 的职责），
    航迹只记录 agent 与 waypoints，避免"上次出航终点"残留干扰后续返航解析。
    """
    if not redis_mgr:
        return
    session_id = session_id or "default"
    agent_id = str(agent_id or "UAV_1").strip()
    waypoints = _parse_waypoints(str(output or ""))
    if not waypoints:
        sys.stderr.write("[ToolExecutor] 未从路径规划输出中解析到航迹点\n")
        sys.stderr.flush()
        return
    try:
        # 起点/终点仅用于日志与 UAV 当前坐标更新，不写入航迹记录
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
                u.update({"Longitude": end_lon, "Latitude": end_lat, "path": route})
                break
        else:
            uavs.append({"id": agent_id, "Longitude": end_lon, "Latitude": end_lat,
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


def _resolve_flight_waypoints(redis_mgr, session_id: str, agent_id: str) -> list:
    """flight 工具输入解析：取该无人机最近一次 path_planning 输出的航迹点。

    Redis 中 uavs.<id>.path.waypoints 存储为 {Longitude, Latitude, Altitude, Time} 字典，
    转为 (时间,经度,纬度,高度) 四元组列表；无可用航迹点时返回空列表。
    """
    if not redis_mgr or not agent_id:
        return []
    try:
        ctx = redis_mgr.get_context(session_id) or {}
        uavs = ctx.get("uavs")
        if isinstance(uavs, list):
            for u in uavs:
                if isinstance(u, dict) and u.get("id") == agent_id:
                    p = u.get("path")
                    if isinstance(p, dict) and isinstance(p.get("waypoints"), list):
                        return [
                            [w.get("Time", 0), w.get("Longitude"), w.get("Latitude"), w.get("Altitude", 0)]
                            for w in p["waypoints"] if isinstance(w, dict)
                        ]
                    break
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] flight 航迹解析失败: {e}\n")
        sys.stderr.flush()
    return []


def _simulate_flight_redis(redis_mgr, session_id: str, agent_id: str, waypoints: list):
    """flight 工具模拟执行：把无人机当前位置更新为航线终点（写回 Redis 上下文）"""
    if not waypoints or not agent_id or not redis_mgr:
        return
    try:
        last = waypoints[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 3:
            lon, lat = last[1], last[2]
        elif isinstance(last, dict):
            lon = last.get("Longitude") if last.get("Longitude") is not None else last.get("x")
            lat = last.get("Latitude") if last.get("Latitude") is not None else last.get("y")
        else:
            return
        if lon is None or lat is None:
            return
        ctx = redis_mgr.get_context(session_id) or {}
        uavs = ctx.get("uavs")
        if not isinstance(uavs, list):
            uavs = []
            ctx["uavs"] = uavs
        for u in uavs:
            if isinstance(u, dict) and u.get("id") == agent_id:
                u["Longitude"] = str(lon)
                u["Latitude"] = str(lat)
                break
        redis_mgr.save_context(session_id, ctx)
        sys.stderr.write(f"[ToolExecutor] flight 模拟飞行完成，{agent_id} 位置更新为 [{lon},{lat}]\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[ToolExecutor] flight 位置更新失败: {e}\n")
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
    # flight 工具：输入航迹点直接取 Redis（path_planning 输出），成功后模拟飞行更新位置
    is_flight = tool_name == "flight"

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
    
    # 执行该动作的无人机 id（用于解析观测位置 mission_position）
    agent_id = str(context.get("agent_id") or (state or {}).get("_agent_id") or "").strip()
    
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
    if not real_exec and not is_flight:
        ctx_for_placeholder = redis_mgr.get_context(session_id)
        placeholder_params = _param_placeholder_names(ctx_for_placeholder, tool_inputs)
        if placeholder_params:
            for p in placeholder_params:
                tool_inputs[p] = p  # 只留参数名
            sys.stderr.write(f"[ToolExecutor] 空占位保护: {tool_name} 参数 {sorted(placeholder_params)} 保持为参数名占位\n")
            sys.stderr.flush()

    # flight 特殊处理：航迹点不依赖 LLM 猜测，直接取该无人机最近一次 path_planning 输出
    # （写入 uavs.<id>.path.waypoints），转为 (时间,经度,纬度,高度) 四元组列表。
    if is_flight:
        wp = _resolve_flight_waypoints(redis_mgr, session_id, agent_id)
        if wp:
            tool_inputs["waypoints"] = wp
            sys.stderr.write(f"[ToolExecutor] flight 已取 Redis 航迹点 {len(wp)} 个\n")
            sys.stderr.flush()

    # 预处理：跳过占位/已有值参数，非 exe 清空 LLM 猜测值（顺序执行，无 I/O）
    resolve_params = []
    for param_name in required_params:
        param_value = tool_inputs.get(param_name, "")
        # 空占位保护已把该参数置为参数名（值为参数名），跳过查找
        if str(param_value).strip() == param_name:
            continue
        # 非 exe 算法：LLM 猜测的值不可信（可能是参数描述/编造文本），
        # 统一清空后走 Redis 查找，保证值一定来自上下文原文。
        if param_value and str(param_value).strip():
            if real_exec or is_flight:
                continue  # 真实 exe / flight：已有值直接用
            sys.stderr.write(f"[ToolExecutor] 非exe 清除 LLM 猜测值: {param_name}={param_value}\n")
            sys.stderr.flush()
            tool_inputs[param_name] = ""
        resolve_params.append(param_name)

    # file 类型参数依赖 knowledge 目录扫描，若存在先扫描一次（并发任务共享，只读）
    if knowledge_files is None and any(
        input_schema.get(p, {}).get("type", "string") == "file" for p in resolve_params
    ):
        knowledge_files = _scan_knowledge_dir()

    # 哨兵：区分 file 参数（不重绑定 context）与非 file 参数（即使 Redis 返回 None 也重绑定）
    _NO_CTX = object()

    async def _resolve_param(param_name: str):
        """解析单个参数：file 类型走 LLM 选文件，其余走 Redis 上下文 + LLM 提取 + 本地降级。
        各参数互相独立（只读上下文、写各自 tool_inputs key），可安全并发。
        Returns: (param_name, 解析值或 None, 本次获取的 Redis 上下文，file 参数为 _NO_CTX)"""
        param_info = input_schema.get(param_name, {})
        param_desc = param_info.get("description", param_name)
        param_type = param_info.get("type", "string")

        sys.stderr.write(f"[ToolExecutor] 参数 {param_name} 为空，开始查找...\n")
        sys.stderr.flush()

        if param_type == "file":
            if knowledge_files:
                selected_path = await _llm_select_file(
                    llm, knowledge_files, scene_description, param_name, param_desc
                )
                if selected_path:
                    sys.stderr.write(f"[ToolExecutor] LLM选择文件: {param_name}={selected_path}\n")
                    sys.stderr.flush()
                    return param_name, selected_path, _NO_CTX
            return param_name, None, _NO_CTX

        ctx = redis_mgr.get_context(session_id)
        sys.stderr.write(f"[ToolExecutor] 结构化上下文: {'有' if ctx else '空'}\n")
        sys.stderr.flush()
        if not ctx:
            sys.stderr.write(f"[ToolExecutor] 无结构化上下文\n")
            sys.stderr.flush()
            return param_name, None, ctx
        if not llm:
            sys.stderr.write(f"[ToolExecutor] LLM不可用\n")
            sys.stderr.flush()
            return param_name, None, ctx

        # 真实 exe（当前只有 path_planning）的坐标参数语义固定：出航终点=目标点，
        # 返航终点=基地。_local_resolve_param 含返航优先 base 的逻辑，可确定性区分；
        # 而 LLM 容易被 uavs.*.path.to_lon/to_lat（上次出航终点）干扰，返航时会把
        # 旧目标点当终点。因此真实 exe 先走本地语义解析，LLM 仅作兜底。
        local_tried = False
        if real_exec:
            local_value = _local_resolve_param(ctx, param_name, param_desc, param_type,
                                               agent_id=agent_id, goal=goal, scene_text=scene_description)
            local_tried = True
            if local_value:
                sys.stderr.write(f"[ToolExecutor] path_planning 本地语义解析: {param_name}={local_value}\n")
                sys.stderr.flush()
                return param_name, local_value, ctx

        extracted_value = await _llm_extract_from_context(
            llm, param_name, param_desc, param_type, ctx, goal
        )
        if extracted_value:
            extracted_value = _pick_coord_component(extracted_value, param_name, param_desc)
            # 频率范围兜底：min/max 频率参数命中范围字符串时拆出对应分量
            extracted_value = _freq_component(extracted_value, param_name, param_desc)
            sys.stderr.write(f"[ToolExecutor] LLM提取: {param_name}={extracted_value}\n")
            sys.stderr.flush()
            return param_name, extracted_value, ctx

        if not real_exec:
            # 不执行 exe 的算法：LLM 未查到值 → 本地键值/别名直取（值必须来自上下文原文，
            # 与 targetMinFreq 命中 targets.T1.minFreq 同理），仍找不到才留空，
            # 交给末尾"非 exe 缺参用参数名占位"处理（写脚本时显示参数名）。
            local_value = _local_resolve_param(ctx, param_name, param_desc, param_type,
                                               agent_id=agent_id, goal=goal, scene_text=scene_description)
            if local_value:
                sys.stderr.write(f"[ToolExecutor] LLM提取失败(非exe)，本地键值补齐: {param_name}={local_value}\n")
                sys.stderr.flush()
                return param_name, local_value, ctx
            sys.stderr.write(f"[ToolExecutor] LLM提取失败(非exe)，{param_name} 保持为空，交由参数名占位\n")
            sys.stderr.flush()
        else:
            # 真实 exe（path_planning 已在上方优先尝试本地解析，失败才走到这里）：
            # LLM 超时/失败/返回 NONE → 本地降级，仍缺才进入 missing_params
            if not local_tried:
                local_value = _local_resolve_param(ctx, param_name, param_desc, param_type,
                                                   agent_id=agent_id, goal=goal, scene_text=scene_description)
                if local_value:
                    sys.stderr.write(f"[ToolExecutor] LLM提取失败，本地降级补齐: {param_name}={local_value}\n")
                    sys.stderr.flush()
                    return param_name, local_value, ctx
            sys.stderr.write(f"[ToolExecutor] LLM提取失败且本地无法补齐: {param_name}\n")
            sys.stderr.flush()
        return param_name, None, ctx

    # 并发解析所有待填参数（互相独立）：N 次 LLM/上下文往返从串行 N× 降为 ~1×
    if resolve_params:
        with timing.track("参数解析LLM"):
            results = await asyncio.gather(*(_resolve_param(p) for p in resolve_params))
        last_ctx = _NO_CTX
        for param_name, value, ctx in results:
            if value:
                tool_inputs[param_name] = value
            if ctx is not _NO_CTX:
                last_ctx = ctx
        # 保持原语义：context 重绑定为最后一次非 file 参数获取的 Redis 上下文
        # （即使为 None，也走原"无结构化上下文"路径；供缺参占位判断使用）
        if last_ctx is not _NO_CTX:
            context = last_ctx

    # 规范化参数值（JSON对象 → 坐标、数字等），再做缺失检查
    if input_schema:
        for p, pinfo in input_schema.items():
            v = tool_inputs.get(p, "")
            if v and str(v).strip():
                tool_inputs[p] = _normalize_param_value(v, pinfo.get("type", "string"), pinfo.get("format", ""))

    # 检查必需参数是否齐全
    if input_schema:
        missing = [p for p in input_schema if not str(tool_inputs.get(p, "")).strip()]
        if missing:
            # 区分「占位字段」（上游已产出但值仍为空）与「真正缺失」（完全不存在）：
            # 占位字段保持原行为；真正缺失走数据依赖解析（自身产出工具 → 指挥跨 agent 解析）
            placeholder_fields = _placeholder_fields_in_context(context, missing)
            truly_missing = [p for p in missing if p not in placeholder_fields]
            if truly_missing:
                missing_params = [
                    {
                        "name": p,
                        "description": input_schema[p].get("description", p),
                        "type": input_schema[p].get("type", "string"),
                    }
                    for p in truly_missing
                ]
                # 自身工具集能产出缺失字段 → 先执行产出工具，再继续本动作
                # 注意：context 已被下方参数解析重绑定为 Redis 结构化上下文，须用 tools 变量（执行上下文传入的工具表）
                producer = _find_self_producer(tools, truly_missing, tool_name)
                if producer:
                    sys.stderr.write(
                        f"[ToolExecutor] {tool_name} 真正缺参 {truly_missing}，"
                        f"自身工具 {producer['action']['tool_name']} 可产出 {producer['field']}，先执行产出动作\n"
                    )
                    sys.stderr.flush()
                    return {
                        "success": False,
                        "output": "",
                        "error": f"缺少参数 {truly_missing}，先执行产出工具 {producer['action']['tool_name']}",
                        "missing_params": missing_params,
                        "self_produce": producer,
                        "tool_inputs": tool_inputs,
                        "pending": False,
                    }
                # 自身无法产出 → 上报指挥跨 agent 解析（exe 与非 exe 一致，不再静默占位）
                names = "、".join(f"{p}({input_schema[p].get('description', p)})" for p in truly_missing)
                sys.stderr.write(f"[ToolExecutor] {tool_name} 真正缺参: {names}，自身无产出工具，上报指挥解析\n")
                sys.stderr.flush()
                return {
                    "success": False,
                    "output": "",
                    "error": f"缺少必需参数: {names}",
                    "missing_params": missing_params,
                    "tool_inputs": tool_inputs,
                    "pending": False,
                }
            # 仅占位字段：非真实执行算法用参数名填充，保证能记录到脚本与 Redis 输出占位
            if missing and not real_exec:
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
        
        with timing.track(f"工具调用[{tool_name}]"):
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
        body_parsed = False  # 是否成功解析出内嵌 JSON body（stdout 为空也视为解析成功）
        try:
            if inner_text:
                parsed = json.loads(inner_text)
                if isinstance(parsed, dict):
                    body_parsed = True
                    if parsed.get("returncode") is not None:
                        returncode = parsed.get("returncode")
                    if "stdout" in parsed:
                        out = parsed.get("stdout", "")
                        readable = out if isinstance(out, str) else str(out)
            if not readable and not body_parsed:
                # 兜底：从原始文本中正则提取 "stdout": "xxx"
                m = _re.search(r'"stdout"\s*:\s*"((?:[^"\\]|\\.)*)"', inner_text or result_text)
                if m:
                    readable = m.group(1).encode().decode("unicode_escape", errors="ignore")
        except Exception:
            body_parsed = False
        if not readable and not body_parsed:
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
            _write_path_to_redis(redis_mgr, session_id, agent_id, tool_inputs, readable)

        # flight 成功：模拟沿航迹飞行，把无人机当前位置更新为航线终点
        if is_flight:
            _simulate_flight_redis(redis_mgr, session_id, agent_id, tool_inputs.get("waypoints") or [])

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
