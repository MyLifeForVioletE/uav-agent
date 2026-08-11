"""统一任务脚本写入（JSON 格式）：每个已执行动作落盘为一条 step

文件：output/actions_script_{session_id}.json
结构：
{
  "steps": [
    {
      "step_id": "UAV-1-1",
      "subject_id": "UAV-1",
      "step_type": "path_planning",
      "step_name": "规划航线",
      "input": {"startPositionLong": 70, ...},
      "output": {"path": [{"t": 0, "x": 70, "y": 15, "z": 0}, ...]}
    }
  ],
  "dependencies": [
    {"predecessor": "UAV-1-1", "successor": "UAV-1-2"},
    {"predecessor": "UAV-1-6", "successor": "_processor_-1"}
  ]
}

output 处理规则：
- stub 工具：执行层直接给出 output_fields（如 ["sweepData"]），写成 {field: field} 占位
- 数组类型字段（如 path_planning 的 path）从输出文本解析为结构化列表
- 其余（object 单值）保留原始文本

dependencies 处理规则（每次 _save 自动重算，文件任何时刻完整）：
- 顺序边：同一 subject 相邻 step（跳过 __pause__ 标记，暂停不参与节点/边）
- 血缘边：step 的消费字段 → 往前找最近一条产出该字段（字段名归一化匹配）的 step，
  任意 subject 均可（如分析 agent 消费 sweepData → 依赖产出 sweepData 的扫频动作）
- LLM 兜底：确定性匹配不到的消费字段，在任务完成时由 LLM 一次性语义映射，
  结果存于进程内存（{ "step_id.field": "producer_step_id" 或 null }）用于去重防重复调用，
  每次重算时保留并合并进 dependencies；仅接受跨步骤的真实映射（过滤自环）
- 条目按 successor 分组：单一依赖时 predecessor 为字符串，多个依赖时为字符串列表
"""
import asyncio
import json
import re
import sys
from pathlib import Path

from core.config import BASE_DIR

# LLM 单次调用总超时（秒）：与 fleet/tool_executor 保持一致，防止流式响应慢导致永久挂起
SCRIPT_WRITER_LLM_TIMEOUT = 60.0


def _load_tool_schemas() -> dict:
    """从 algorithms.json 加载各工具的输入/输出 schema：{tool_name: {"input": dict, "output": dict}}"""
    path = BASE_DIR / "algorithms.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            a["name"]: {
                "input": a.get("input_schema", {}) or {},
                "output": a.get("output_schema", {}) or {},
            }
            for a in data.get("capabilities", [])
        }
    except Exception:
        return {}


_SCHEMAS = None


def _get_schema(tool_name: str) -> dict:
    """获取工具的 input/output schema；无定义时返回空 dict"""
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = _load_tool_schemas()
    return _SCHEMAS.get(tool_name, {}) or {}


def _get_input_schema(tool_name: str) -> dict:
    return _get_schema(tool_name).get("input", {})


def _get_output_schema(tool_name: str) -> dict:
    return _get_schema(tool_name).get("output", {})


# 可去除 t 前缀的目标字段基名（tFreq→freq, tBand→band, tStre→stre, tlat→lat, tlon→lon）
_TARGET_PREFIX_BASES = {"freq", "band", "stre", "lat", "lon"}

# 飞行/返航 external 动作关键词：命中则回填/解析航线引用（起飞/降落等不沿航线的动作除外）
_FLIGHT_KEYWORDS = ("飞行", "飞往", "飞向", "飞至", "飞抵", "飞回", "返航", "返回", "巡航", "航线")


def is_flight_action(action_name: str) -> bool:
    """判断动作是否为沿航线飞行的飞行/返航动作（起飞/降落不含关键词，返回 False）"""
    return any(k in (action_name or "") for k in _FLIGHT_KEYWORDS)


def recent_path_ref(steps: list, subject_id: str) -> str:
    """同 subject 最近一次 path_planning 步骤的 step:// 引用；无则返回空串"""
    ref = ""
    for s in steps:
        if s.get("subject_id") == subject_id and s.get("step_type") == "path_planning":
            ref = f"step://{s.get('step_id')}.output.path"
    return ref


def _recent_path_waypoints(steps: list, subject_id: str) -> list:
    """同 subject 最近一次 path_planning 步骤 output.path 的航迹点列表；无则返回空列表"""
    for s in reversed(steps):
        if s.get("subject_id") == subject_id and s.get("step_type") == "path_planning":
            out = s.get("output")
            if isinstance(out, dict) and isinstance(out.get("path"), list):
                return list(out["path"])
            return []
    return []


def _norm_field(name) -> str:
    """字段名归一化，供血缘匹配：
    - 小写
    - 去 t 前缀（tFreq/tBand/tStre/tlat/tlon 视为目标侧字段）
    - 去尾部数字索引（positionAnalysis 的 lat1/lat2/stre1 → lat/stre）
    """
    n = str(name).strip().lower()
    if n.startswith("t") and len(n) > 1 and n[1:] in _TARGET_PREFIX_BASES:
        n = n[1:]
    return re.sub(r"\d+$", "", n)


def _tool_produced_fields(step: dict) -> set:
    """step 实际产出的字段（归一化）：output 为 dict 取键；否则回退到工具 output_schema"""
    output = step.get("output")
    if isinstance(output, dict):
        return {_norm_field(k) for k in output.keys()}
    schema = _get_output_schema(step.get("step_type", ""))
    if schema:
        return {_norm_field(k) for k in schema.keys()}
    return set()


def _tool_consumed_fields(step: dict) -> set:
    """step 消费的字段（归一化）：优先工具 input_schema；无 schema（external/system）回退到实际 input 键。
    step:// 数据引用（如 input.route）不算消费字段，由 _edge_predecessors 单独建边。
    """
    schema = _get_input_schema(step.get("step_type", ""))
    if schema:
        return {_norm_field(k) for k in schema.keys()}
    inputs = step.get("input")
    if isinstance(inputs, dict):
        return {_norm_field(k) for k, v in inputs.items()
                if not (isinstance(v, str) and v.startswith("step://"))}
    return set()


def _edge_predecessors(steps: list, llm_field_map: dict = None) -> dict:
    """计算 successor → [predecessor, ...]（保序去重）：
    - 顺序边：同一 subject 相邻 step（跳过 __pause__ 标记，暂停不参与节点/边）
    - 血缘边：step 消费字段 → 往前找最近一条产出该字段（归一化匹配）的 step，任意 subject
    - LLM 兜底边：llm_field_map 中已解析的字段映射（key 为 successor_step_id.field）
    """
    preds = {}

    def add(successor, predecessor):
        if not predecessor or predecessor == successor:
            return
        lst = preds.setdefault(successor, [])
        if predecessor not in lst:
            lst.append(predecessor)

    # 顺序边
    prev_by_subject = {}
    for step in steps:
        if step.get("step_type") == "__pause__":
            continue
        subject = step.get("subject_id", "")
        step_id = step.get("step_id", "")
        if subject in prev_by_subject:
            add(step_id, prev_by_subject[subject])
        prev_by_subject[subject] = step_id

    # 血缘边：按序扫描，produced_at 记录各归一化字段最近的产出 step 下标
    produced_at = {}
    for i, step in enumerate(steps):
        if step.get("step_type") == "__pause__":
            continue
        step_id = step.get("step_id", "")
        # 先匹配消费字段（避免自产自销，如 communication 同时消费/产出 time）
        for f in _tool_consumed_fields(step):
            j = produced_at.get(f)
            if j is not None:
                add(step_id, steps[j].get("step_id", ""))
        for f in _tool_produced_fields(step):
            produced_at[f] = i

    # 数据引用边：input 中 step://<id>.output.path 引用 → 显式数据依赖（如飞行动作依赖航线产出）
    for step in steps:
        if step.get("step_type") == "__pause__":
            continue
        inputs = step.get("input")
        if not isinstance(inputs, dict):
            continue
        step_id = step.get("step_id", "")
        for v in inputs.values():
            if isinstance(v, str) and v.startswith("step://"):
                m = re.match(r"step://([^.]+)", v)
                if m:
                    add(step_id, m.group(1))

    # LLM 兜底边（持久化映射；自环已在 add 中过滤）
    for field_key, from_id in (llm_field_map or {}).items():
        if not from_id:
            continue
        add(field_key.partition(".")[0], from_id)

    return preds


def compute_dependencies(steps: list) -> list:
    """计算动作间依赖关系（确定性部分）：
    返回 [{"predecessor": str|list, "successor": str}, ...]，按 successor 分组，
    单一依赖时 predecessor 为字符串，多个依赖时为字符串列表。
    """
    preds = _edge_predecessors(steps)
    return [
        {"predecessor": preds[s] if len(preds[s]) > 1 else preds[s][0], "successor": s}
        for s in preds
    ]


def script_path(session_id: str) -> Path:
    out_dir = BASE_DIR / "output"
    return out_dir / f"actions_script_{session_id}.json"


# LLM 血缘映射去重缓存（进程内存，不落盘）：session_id → {"step_id.field": producer_step_id 或 None}
_FIELD_MAP_CACHE: dict = {}


def _get_llm_field_map(session_id: str) -> dict:
    return _FIELD_MAP_CACHE.setdefault(session_id, {})


def _read_script(session_id: str) -> dict:
    """读取脚本完整内容：{steps}；文件缺失/损坏时返回空壳"""
    path = script_path(session_id)
    if not path.is_file():
        return {"steps": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"steps": []}
        steps = data.get("steps")
        return {"steps": steps if isinstance(steps, list) else []}
    except Exception:
        return {"steps": []}


def load_steps(session_id: str) -> list:
    """读取当前脚本已有的 steps 列表"""
    return _read_script(session_id)["steps"]


def _save(session_id: str, steps: list, llm_field_map: dict = None):
    """写入脚本；每次重算依赖并保留进程内存中的 LLM 血缘映射（文件任何时刻完整）。
    公开脚本只含 steps + dependencies，不产生任何额外文件。
    """
    if llm_field_map is None:
        llm_field_map = _get_llm_field_map(session_id)
    try:
        preds = _edge_predecessors(steps, llm_field_map)
        deps = [
            {"predecessor": preds[s] if len(preds[s]) > 1 else preds[s][0], "successor": s}
            for s in preds
        ]
        script_path(session_id).write_text(
            json.dumps({"steps": steps, "dependencies": deps}, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
    except Exception as e:
        sys.stderr.write(f"[ScriptWriter] 脚本写入失败: {e}\n")
        sys.stderr.flush()


def _parse_waypoints(output: str) -> list:
    """从输出文本解析 (v1,v2,v3,v4) 元组序列 → [{t,x,y,z}, ...]"""
    waypoints = []
    for m in re.finditer(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", str(output)):
        try:
            waypoints.append({
                "t": float(m.group(4)),
                "x": float(m.group(1)),
                "y": float(m.group(2)),
                "z": float(m.group(3)),
            })
        except (ValueError, TypeError):
            continue
    return waypoints


def _parse_output(tool_name: str, output: str):
    """按算法 output_schema 解析输出：
    - 数组类型字段（如 path_planning 的 path）解析为结构化列表
    - 其余（object 单值，且无 output_fields 提供）保留原始文本
    """
    text = (output or "").strip()
    if not text:
        return ""
    schema = _get_output_schema(tool_name)
    array_fields = [n for n, info in schema.items()
                    if isinstance(info, dict) and info.get("type") == "array"]
    if array_fields:
        elements = _parse_waypoints(text)
        if elements:
            return {array_fields[0]: elements}
    return text


def write_action_step(session_id: str, subject_id: str, action: dict,
                      tool_inputs: dict, output: str, output_fields: list = None):
    """写一条已执行动作 step；step_id 按 subject 在该文件内递增（跨任务唯一）

    output_fields: 工具实际产出的字段名列表（stub 工具从执行层直接给出，
    如 ["sweepData"]），此时 output 直接写为 {field: field} 占位。
    """
    action_name = action.get("action_name", "") or ""
    tool_name = action.get("tool_name", "") or action_name
    executor = action.get("executor", "") or ""

    # step_type：tool 步骤用算法名；external/system 用执行器类型；兜底用工具/动作名
    if executor == "tool" and tool_name:
        step_type = tool_name
    elif executor:
        step_type = executor
    else:
        step_type = tool_name or "unknown"

    inputs = {k: v for k, v in (tool_inputs or {}).items() if str(v).strip()}

    steps = load_steps(session_id)

    # 飞行/返航 external 动作：内联最近一次同 subject 的 path_planning 输出航迹点。
    # 写入时刻该 path_planning 步骤必然已落盘，出航/返航自动关联各自航段；
    # 无可用航迹点时标注"等待航线"。
    if executor == "external" and is_flight_action(action_name):
        waypoints = _recent_path_waypoints(steps, subject_id)
        inputs["route"] = waypoints if waypoints else "等待航线"

    if output_fields:
        step_output = {f: f for f in output_fields}
    else:
        step_output = _parse_output(tool_name, output)

    seq = sum(1 for s in steps if s.get("subject_id") == subject_id) + 1
    step = {
        "step_id": f"{subject_id}-{seq}",
        "subject_id": subject_id,
        "step_type": step_type,
        "step_name": action_name or tool_name or "未知动作",
        "input": inputs,
        "output": step_output,
    }
    steps.append(step)
    _save(session_id, steps)
    sys.stderr.write(f"[ScriptWriter] 已写入步骤 {step['step_id']}: {step['step_name']}\n")
    sys.stderr.flush()


def write_pause_marker(session_id: str, reason: str = ""):
    """写入暂停标记（缺参等待用户/指挥等待等场景）"""
    steps = load_steps(session_id)
    steps.append({
        "step_id": f"pause-{len(steps) + 1}",
        "subject_id": "_coordinator_",
        "step_type": "__pause__",
        "step_name": reason or "暂停",
        "input": {},
        "output": "",
    })
    _save(session_id, steps)
    sys.stderr.write(f"[ScriptWriter] 已写入暂停标记: {reason}\n")
    sys.stderr.flush()


def _find_unmatched_consumed(steps: list) -> dict:
    """确定性血缘未覆盖的消费字段：{step_id: [normalized_field, ...]}
    （与 compute_dependencies 同规则：只认更早的产出者，且匹配发生在记录自身产出之前）
    """
    produced_at = {}
    unmatched = {}
    for i, step in enumerate(steps):
        if step.get("step_type") == "__pause__":
            continue
        miss = []
        for f in _tool_consumed_fields(step):
            if produced_at.get(f) is None:
                miss.append(f)
        if miss:
            unmatched[step.get("step_id", "")] = miss
        for f in _tool_produced_fields(step):
            produced_at[f] = i
    return unmatched


async def refresh_dependencies_with_llm(session_id: str, llm) -> bool:
    """任务完成时异步 LLM 兜底：对确定性血缘未匹配的消费字段做一次性语义映射。

    - 已处理过（进程内存 field_map 中存在）的字段不再询问，避免重复调用
    - 无未匹配字段/无候选产出步骤/无 LLM 时直接返回 False（不发起调用）
    - 仅接受跨步骤映射（过滤自环）；无产出/无效产出记为 null 防重复询问
    - 映射存于进程内存，每次 _save 重算时合并进 dependencies（不产生任何文件）

    Returns:
        bool: 是否发起了 LLM 调用并成功更新
    """
    data = _read_script(session_id)
    steps = data["steps"]
    field_map = _get_llm_field_map(session_id)
    unmatched = _find_unmatched_consumed(steps)
    if not unmatched:
        return False

    pending = {}
    for step_id, fields in unmatched.items():
        new = [f for f in fields if f"{step_id}.{f}" not in field_map]
        if new:
            pending[step_id] = new
    if not pending or llm is None:
        return False

    # 候选产出（step_id, 原始字段, 归一化字段, step_name），按 (step_id, norm) 去重
    producers = []
    seen_prod = set()
    for step in steps:
        if step.get("step_type") == "__pause__":
            continue
        step_id = step.get("step_id", "")
        output = step.get("output")
        if isinstance(output, dict):
            fields = [(k, _norm_field(k)) for k in output.keys()]
        else:
            schema = _get_output_schema(step.get("step_type", ""))
            fields = [(k, _norm_field(k)) for k in schema.keys()]
        for orig, norm in fields:
            key = (step_id, norm)
            if key in seen_prod:
                continue
            seen_prod.add(key)
            producers.append((step_id, orig, norm, step.get("step_name", "")))
    if not producers:
        return False

    consumer_text = "\n".join(
        f"- {sid}.{f}" for sid, fields in pending.items() for f in fields
    )
    producer_text = "\n".join(
        f"- {sid}（{name or sid}）产出 {orig}（归一化 {norm}）"
        for sid, orig, norm, name in producers
    )

    prompt = (
        "你是多无人机电磁侦察任务的依赖推断模块。任务脚本中的动作步骤之间存在数据血缘依赖。\n"
        "以下消费字段无法按字段名直接匹配到产出步骤，请根据字段语义、工具输入输出描述与动作目标，"
        "为每个消费字段选择它数据来源的产出步骤。\n\n"
        f"【未匹配的消费字段（consumer_step_id.field_name）】\n{consumer_text}\n\n"
        f"【候选产出步骤（producer_step_id）】\n{producer_text}\n\n"
        "【规则】\n"
        "1. 每个消费字段最多选择一个产出步骤；必须在候选产出步骤中选取（匹配 step_id 前缀）。\n"
        "2. 产出步骤必须与消费字段属于**不同**的步骤（禁止自产自销，如把 path_planning 的输入\n"
        "   坐标/地图文件映射到它自己的 path 输出）。\n"
        "3. 只能建立真实存在的数据依赖，禁止为了凑边而臆造。\n"
        "4. 若某消费字段的数据来自任务配置/外部系统而非任何步骤产出（如起飞点坐标、地图/盲区\n"
        "   文件、扫频目标频率与步进），则该字段输出空字符串。\n"
        "5. 语义匹配示例：directionFinding 输入 freq ← signalAnalysis 输出 tFreq（目标频率）；\n"
        "   positionAnalysis 输入 lat1/lon1 ← directionFinding 输出 lat/lon（测量点经纬度）。\n\n"
        "只输出 JSON 对象，key 为 consumer_step_id.field_name，value 为 producer_step_id 或空字符串：\n"
        '{"step_id.field": "producer_step_id"}'
    )

    try:
        from langchain_core.messages import HumanMessage
        raw = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=SCRIPT_WRITER_LLM_TIMEOUT,
        )
        raw = str(raw.content).strip()
    except Exception as e:
        sys.stderr.write(f"[ScriptWriter] LLM 依赖推断调用失败: {e}\n")
        sys.stderr.flush()
        return False

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        mapping = json.loads(m.group()) if m else {}
    except Exception as e:
        sys.stderr.write(f"[ScriptWriter] LLM 依赖推断解析失败: {raw[:200]}\n")
        sys.stderr.flush()
        return False

    if not isinstance(mapping, dict):
        return False

    # 有效产出步骤 id（排除暂停标记）
    valid_producers = {
        s.get("step_id", "") for s in steps if s.get("step_type") != "__pause__"
    }

    # 本次涉及的全部待处理字段一律落账（有跨步骤产出→producer，否则→None），
    # 避免下次任务完成时对相同字段重复发起 LLM 调用
    updated = False
    for step_id, fields in pending.items():
        for field in fields:
            key = f"{step_id}.{field}"
            if key in field_map:
                continue
            prod = mapping.get(key)
            if isinstance(prod, str) and prod and prod != step_id and prod in valid_producers:
                field_map[key] = prod
            else:
                field_map[key] = None  # 自环/无效/无产出 → 记为已处理
            updated = True

    if updated:
        _save(session_id, steps, field_map)
        sys.stderr.write(f"[ScriptWriter] LLM 依赖兜底完成，新增/确认映射 {len(field_map)} 条\n")
        sys.stderr.flush()
    return updated
