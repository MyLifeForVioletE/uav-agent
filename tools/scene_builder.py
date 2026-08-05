"""
场景构建工具：加载 schema/scene/types/*.json → LLM 提取实体 → 按显式映射分发到 schema/scene/excel/*.json 定义的表 → 写入 Excel
"""
import json
import os
import re
from pathlib import Path
from datetime import datetime
import requests
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

from core.config import OLLAMA_BASE, MODEL, BASE_DIR

TYPES_DIR = BASE_DIR / "schema" / "scene" / "types"
EXCEL_DIR = BASE_DIR / "schema" / "scene" / "excel"


# ── 加载 ────────────────────────────────────────────────

def load_type_schemas(types_dir: str) -> dict:
    """加载 schema/scene/types/*.json（每个文件一个实体类型）"""
    types = {}
    for fpath in sorted(Path(types_dir).glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        for type_name, type_body in raw.items():
            params = type_body.get("参数", {})
            types[type_name] = {
                "name": type_name,
                "desc": type_body.get("描述", ""),
                "params": params,
            }
    return types


def load_excel_schemas(excel_dir: str) -> dict:
    """加载 schema/scene/excel/*.json（每个文件一个 Excel 表）"""
    schemas = {}
    for fpath in sorted(Path(excel_dir).glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        for schema_name, schema_body in raw.items():
            fields = []
            for pname, pinfo in schema_body.get("参数", {}).items():
                fields.append({
                    "name": pname,
                    "type": pinfo.get("类型", "string"),
                    "desc": pinfo.get("描述", ""),
                })
            schemas[schema_name] = {"name": schema_name, "fields": fields}
    return schemas


# ── 阶段1：LLM 提取实体 ──────────────────────────────

def _format_param(name: str, pinfo: dict, depth: int = 0) -> str:
    """格式化一个参数为提示词中的一行"""
    prefix = "  " * (depth + 1)
    t = pinfo.get("类型", "string")
    desc = pinfo.get("描述", "")
    unit = pinfo.get("单位", "")
    extra = f" ({t})"
    if unit:
        extra += f" 单位:{unit}"
    if desc:
        extra += f"  {desc}"
    return f"{prefix}- {name}{extra}"


def _format_type_section(type_name: str, type_def: dict, depth: int = 0) -> list[str]:
    """递归格式化一个实体类型的所有字段（含嵌套 struct/array）"""
    lines = []
    indent = "  " * depth
    for pname, pinfo in type_def.get("params", {}).items():
        ptype = pinfo.get("类型", "string")
        if ptype == "struct":
            lines.append(f"{indent}  子对象 '{pname}'（struct）：")
            nested_type = {"params": pinfo.get("参数", {})}
            lines.extend(_format_type_section(pname, nested_type, depth + 1))
        elif ptype == "array":
            lines.append(f"{indent}  数组 '{pname}'（array）：")
            nested_type = {"params": pinfo.get("参数", {})}
            lines.extend(_format_type_section(pname, nested_type, depth + 1))
        else:
            lines.append(_format_param(pname, pinfo, depth))
    return lines


def build_extraction_prompt(user_input: str, type_schemas: dict) -> str:
    """构造提示词：按实体类型（types/*.json）组织，LLM 输出实体级 JSON"""
    lines = ["你是一个场景数据提取专家。根据用户的描述，提取以下实体类型的实例数据。\n"]

    for type_name, type_def in type_schemas.items():
        lines.append(f"实体类型 '{type_name}' 的字段：")
        lines.extend(_format_type_section(type_name, type_def, 0))
        lines.append("")

    lines.append("输出格式（严格的 JSON，不要多余文字）：")
    lines.append("```json")
    lines.append("{")
    lines.append('  "<实体类型名>": [')
    lines.append("    {")
    lines.append('      "字段1": 值,')
    lines.append('      "字段2": 值,')
    lines.append('      "嵌套子对象名": {')
    lines.append('        "子字段1": 值')
    lines.append("      },")
    lines.append('      "嵌套数组名": [')
    lines.append("        {")
    lines.append('          "数组字段1": 值')
    lines.append("        }")
    lines.append("      ]")
    lines.append("    }")
    lines.append("  ]")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("规则：")
    lines.append("- 每个实体类型是一个顶层 key，值是该类型的实例数组")
    lines.append("- 用户输入中明确提到的字段才填入，未提及的字段留空或省略")
    lines.append("- 字符串值用双引号，数值不加引号，bool 用 true/false")
    lines.append("- 嵌套 struct 用 {{ }}，嵌套 array 用 [ ]")
    lines.append("- 多条同类数据作为数组的多个元素")
    lines.append("")
    lines.append(f"用户描述：\n{user_input}")

    return "\n".join(lines)


def extract_entities(user_input: str, type_schemas: dict) -> dict:
    """阶段1：LLM 提取实体实例"""
    prompt = build_extraction_prompt(user_input, type_schemas)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "你只输出JSON，不输出任何解释文字。"},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "temperature": 0.1,
                "options": {"num_predict": 8192},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except Exception as e:
        return {"error": str(e)}

    match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    json_str = match.group(1) if match else content

    try:
        return json.loads(json_str, object_pairs_hook=_merge_duplicate_keys)
    except json.JSONDecodeError:
        repaired = _fix_json_brackets(json_str)
        try:
            return json.loads(repaired, object_pairs_hook=_merge_duplicate_keys)
        except json.JSONDecodeError as e:
            debug_path = BASE_DIR / "scene" / "llm_raw_response.txt"
            Path(BASE_DIR / "scene").mkdir(exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"=== 提取的JSON字符串 ===\n{json_str}\n\n=== 原始响应 ===\n{content}")
            return {"error": f"JSON 解析失败: {e}", "raw": json_str[:500]}


def _merge_duplicate_keys(pairs):
    """合并JSON中重复key为数组"""
    d = {}
    for k, v in pairs:
        if k in d:
            existing = d[k]
            if not isinstance(existing, list):
                d[k] = [existing]
            d[k].append(v)
        else:
            d[k] = v
    return d


def _fix_json_brackets(s: str) -> str:
    """修复JSON中缺失的 ]：LLM 常把数组结尾的 ] 写成 }"""
    opens = s.count('[')
    closes = s.count(']')
    if opens <= closes:
        return s

    in_string = False
    stack = []
    insert_positions = []

    for i, ch in enumerate(s):
        if ch == '"' and (i == 0 or s[i - 1] != '\\'):
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == '[':
            stack.append('[')
        elif ch == '{':
            stack.append('{')
        elif ch == ']':
            while stack and stack[-1] != '[':
                stack.pop()
            if stack:
                stack.pop()
        elif ch == '}':
            if stack and stack[-1] == '[':
                insert_positions.append(i)
                stack.pop()
            else:
                while stack and stack[-1] != '{':
                    stack.pop()
                if stack:
                    stack.pop()

    for pos in reversed(insert_positions):
        s = s[:pos] + ']' + s[pos + 1:]

    need = s.count('[') - s.count(']')
    if need > 0:
        s += ']' * need

    return s


# ── 阶段2：类型→Excel 显式映射 ──────────────────────

# 映射表：定义每种实体类型的字段如何分发到各 Excel 表
# 每个条目: {"sheet": 表名, "fields": [字段名], "id_field": 可选, "fk_field": 可选, "fk_from": 可选}
TYPE_EXCEL_MAP = {
    "carrier": [
        {"sheet": "carrier_info", "fields": [
            "载体ID", "载体名称", "载体模型ID", "载体平台ID", "项目ID",
            "经度", "纬度", "海拔高度", "所属阵营", "水平朝向",
            "速度", "运动标志", "编队的路径ID", "载体开始工作时间", "载体接收工作时间",
        ]},
    ],

    "equipment": [
        {"sheet": "equipment_base", "fields": [
            "装备ID", "装备名称", "装备模型ID", "所在载体ID",
            "平均功率", "峰值功率", "载波功率", "必要带宽",
            "在载体上的高度", "天线方位角", "天线俯仰角", "当前工作参数ID",
            "发射馈线损耗", "接收馈线损耗", "链接发射机的id", "装备开机状态", "优先级",
        ], "id_field": "装备ID"},
        {"sheet": "equipment_model", "fields": [
            "模型ID", "模型名称", "模型类型",
            "发射频率下限", "发射频率上限", "发射功率", "发射机数据率",
            "发射机必要带宽", "发射机频谱模板ID", "调制类型", "编码特性",
            "非谐波杂散发射抑制", "发射机二阶谐波衰减", "发射机三阶谐波衰减",
            "发射系统损耗", "接收天线id", "接收频率下限", "接收频率上限",
            "接收灵敏度", "噪声温度", "中频带宽", "中频带宽选择性",
            "接收调制类型", "接收编码特性", "接收扩频特性", "处理增益",
            "要求的信噪比", "邻信道抑制", "接收机阻塞电平", "动态范围",
            "互调响应抑制", "杂散响应抑制", "接收机频谱模板ID", "要求的信干比",
            "接收系统损耗", "默认工作参数", "扩频速率", "扩频增益",
            "跳频带宽", "跳频频率数", "跳频速率",
            "发射机基带特性", "发射机射频特性", "接收机射频特性",
            "发射天线ID", "用户ID", "天线名称", "收发机状态",
        ], "nested_key": "equipment_model", "id_field": "模型ID",
         "fk_field": "模型ID", "fk_from": "_parent_id"},
        {"sheet": "equipment_radar_extd", "fields": [
            "装备ID", "调制方式", "发射工作频率起", "发射工作频率止",
            "脉冲宽度", "脉冲重复频率", "占空比",
            "全向探测距离", "立体探测距离", "立体盲区距离", "立体的仰角",
            "雷达水平探测范围", "雷达水平角度", "定向探测距离", "仰角",
            "雷达追踪载体的ID", "雷达工作类型", "备注",
        ], "nested_key": "equipment_radar_extd",
         "fk_field": "装备ID", "fk_from": "_parent_id"},
        {"sheet": "equipment_radio_extd", "fields": [
            "装备ID", "调制方式", "码速率",
            "发射工作频率起", "发射工作频率止",
            "是否跳频", "跳速", "跳频序列",
            "是否扩频", "扩频带宽", "收发频率间隔",
            "发射馈线损耗", "接收馈线损耗", "全向探测距离", "备注",
        ], "nested_key": "equipment_radio_extd",
         "fk_field": "装备ID", "fk_from": "_parent_id"},
        {"sheet": "equipment_interference_extd", "fields": [
            "装备ID", "调制方式", "码速率",
            "发射工作频率起", "发射工作频率止",
            "是否跳频", "跳速", "跳频序列",
            "是否扩频", "扩频带宽", "收发频率间隔",
            "发射馈线损耗", "接收馈线损耗", "备注1",
            "任务类型", "干扰类型", "备注2",
        ], "nested_key": "equipment_interference_extd",
         "fk_field": "装备ID", "fk_from": "_parent_id"},
        {"sheet": "equipment_model_radar_extd", "fields": [
            "模型ID", "脉冲宽度", "脉冲重复周期", "占空比",
            "上升沿时间", "下降沿时间", "变频带宽", "信道带宽",
            "频点个数", "备注",
        ], "nested_key": "equipment_model_radar_extd",
         "fk_field": "模型ID", "fk_from": "_model_id"},
        {"sheet": "equipment_model_radio_extd", "fields": [
            "模型ID", "调制方式", "码速率",
            "是否跳频", "跳速", "跳频序列",
            "是否扩频", "扩频带宽", "收发频率间隔",
            "发射馈线损耗", "接收馈线损耗", "备注",
        ], "nested_key": "equipment_model_radio_extd",
         "fk_field": "模型ID", "fk_from": "_model_id"},
    ],

    "antenna": [
        {"sheet": "antenna_base", "fields": [
            "天线id", "天线名称", "天线类型", "天线制造商", "模型精度参数",
            "天线增益", "水平波束宽度", "垂直波束宽度", "前后比",
            "天线长度", "波束下倾角", "迎风面积", "测量频率",
            "频率下限", "频率上限", "极化方式", "极化隔离度",
            "特性阻抗", "其他损耗", "备注", "创建时间", "最后修改时间",
        ], "id_field": "天线id"},
        {"sheet": "antenna_pattern_h", "fields": [
            "天线ID", "水平角度", "损耗",
        ], "nested_key": "水平天线方向图",
         "fk_field": "天线ID", "fk_from": "_parent_id"},
        {"sheet": "antenna_pattern_v", "fields": [
            "天线ID", "垂直角度", "损耗",
        ], "nested_key": "垂直天线方向图",
         "fk_field": "天线ID", "fk_from": "_parent_id"},
    ],

    "project": [
        {"sheet": "project_base", "fields": [
            "项目ID", "项目名称", "仿真开始时间", "仿真结束时间",
            "项目创建时间", "仿真步进", "项目区域矢量ID",
            "用户ID", "备注", "项目模板标识",
        ], "id_field": "项目ID"},
        {"sheet": "project_vector", "fields": [
            "矢量ID", "矢量名称", "项目ID", "载体ID",
            "矢量类型", "用户ID", "备注",
        ], "nested_key": "project_vector",
         "fk_field": "项目ID", "fk_from": "_parent_id",
         "id_field": "矢量ID"},
        {"sheet": "project_vector_extd", "fields": [
            "主键ID", "项目矢量ID", "经度", "纬度", "高度",
            "距起始点距离", "从起始点运动到该点的时间间隔", "载体运动的总时间",
        ], "nested_key": "project_vector_extd",
         "fk_field": "项目矢量ID", "fk_from": "_vector_id",
         "parent_key": "project_vector"},
    ],

    "work": [
        {"sheet": "work_param_base", "fields": [
            "工作参数ID", "开始时间", "结束时间",
            "发射频率", "发射天线方位角", "发射天线俯仰角",
            "接收频率", "接收天线方位角", "接收天俯仰角",
            "发射天线方位角范围", "发射天线俯仰角范围",
            "平均功率", "峰值功率", "载波功率",
            "单位", "调试方式", "干扰保护门限",
        ]},
    ],
}


def _find_id_field(fields: dict) -> str | None:
    """从字段中找到第一个 ID 类字段名"""
    for name in fields:
        if "ID" in name or "Id" in name or "id" in name:
            return name
    return None


def _auto_generate_id(type_name: str, idx: int, fields: dict) -> str:
    """为实例自动生成 ID"""
    id_field = _find_id_field(fields)
    if id_field:
        val = fields[id_field]
        if val not in (None, ""):
            return str(val)
    ts = int(datetime.now().timestamp()) % 100000
    return f"{type_name}_{idx + 1}_{ts}"


def _match_excel_fields(excel_schema: dict, data: dict) -> dict:
    """按 excel schema 的字段名从 data 中取值，缺失字段填 None"""
    return {f["name"]: data.get(f["name"]) for f in excel_schema["fields"]}


def map_to_tables(entities: dict, type_schemas: dict, excel_schemas: dict) -> dict:
    """将 LLM 输出按显式映射分发到各 Excel 表。

    处理逻辑：
    1. 顶层字段 → 直接映射到对应表
    2. nested_key=xxx 的 struct → 从实例中取出，按其字段映射
    3. nested_key=xxx 的 array → 展平为多行，注入父级 ID 作为外键
    4. 二层嵌套数组（如 equipment_model 内的 radar_extd）→ 从 struct 内取出，注入模型 ID
    """
    result = {sheet_name: [] for sheet_name in excel_schemas}

    for type_name, instances in entities.items():
        if not isinstance(instances, list):
            instances = [instances]
        if type_name not in TYPE_EXCEL_MAP:
            continue

        mapping = TYPE_EXCEL_MAP[type_name]

        for inst_idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue

            # 生成该实例的主 ID
            base_id = _auto_generate_id(type_name, inst_idx, inst)

            # 处理该类型的所有映射规则
            # 先收集二层嵌套规则（parent_key 不为空的），稍后在父级数组处理时一并提取
            second_level_rules = {r["parent_key"]: r for r in mapping if r.get("parent_key")}

            for rule in mapping:
                sheet = rule["sheet"]
                if sheet not in excel_schemas:
                    continue

                fields = rule.get("fields", [])
                nested_key = rule.get("nested_key")

                if nested_key is None:
                    # ── 顶层字段直接映射 ──
                    row = {fn: inst.get(fn) for fn in fields}
                    if rule.get("id_field"):
                        id_val = row.get(rule["id_field"])
                        if id_val in (None, ""):
                            row[rule["id_field"]] = base_id
                            inst[rule["id_field"]] = base_id
                    result[sheet].append(row)

                elif isinstance(inst.get(nested_key), dict):
                    # ── struct 嵌套：取出子对象，映射到独立表 ──
                    sub = inst[nested_key]
                    row = {fn: sub.get(fn) for fn in fields}
                    # 注入外键（fk_from=_parent_id 时，仅当 fk_field 不在子对象已有值时才注入）
                    fk_field = rule.get("fk_field")
                    fk_from = rule.get("fk_from")
                    if fk_field and fk_from:
                        if fk_from == "_parent_id":
                            if not row.get(fk_field):
                                row[fk_field] = base_id
                        elif fk_from == "_model_id":
                            row[fk_field] = sub.get(fk_field) or sub.get("模型ID") or base_id
                    # 生成子对象的 ID（不覆盖子对象自身已有的值）
                    if rule.get("id_field"):
                        id_val = row.get(rule["id_field"])
                        if id_val in (None, ""):
                            model_id = base_id
                            row[rule["id_field"]] = model_id
                            sub[rule["id_field"]] = model_id
                    result[sheet].append(row)

                elif isinstance(inst.get(nested_key), list):
                    # ── array 嵌套：展平为多行 ──
                    items = inst[nested_key]

                    for ai, item in enumerate(items):
                        if not isinstance(item, dict):
                            continue
                        row = {fn: item.get(fn) for fn in fields}
                        # 注入外键
                        fk_field = rule.get("fk_field")
                        fk_from = rule.get("fk_from")
                        if fk_field and fk_from:
                            if fk_from == "_parent_id":
                                row[fk_field] = base_id
                            elif fk_from == "_model_id":
                                model_obj = inst.get("equipment_model")
                                if isinstance(model_obj, dict):
                                    row[fk_field] = model_obj.get("模型ID") or base_id
                                else:
                                    row[fk_field] = base_id
                        # 生成行 ID
                        if rule.get("id_field"):
                            id_val = row.get(rule["id_field"])
                            if id_val in (None, ""):
                                row[rule["id_field"]] = f"{base_id}_{ai + 1}"
                        result[sheet].append(row)

                        # ── 处理该数组项内的二层嵌套数组（如 project_vector 内的 project_vector_extd）──
                        if nested_key in second_level_rules:
                            sl_rule = second_level_rules[nested_key]
                            sl_key = sl_rule["nested_key"]
                            sl_fields = sl_rule.get("fields", [])
                            sub_items = item.get(sl_key, [])
                            if not isinstance(sub_items, list):
                                sub_items = [sub_items]
                            # 父级 ID：优先用 item 的 id_field 值
                            p_id = item.get(sl_rule.get("id_field", "")) or item.get("矢量ID") or item.get("矢量名称") or f"{base_id}_v{ai + 1}"
                            for si, sub_item in enumerate(sub_items):
                                if not isinstance(sub_item, dict):
                                    continue
                                sl_row = {fn: sub_item.get(fn) for fn in sl_fields}
                                # 注入外键
                                sl_fk = sl_rule.get("fk_field")
                                if sl_fk:
                                    sl_row[sl_fk] = p_id
                                if sl_rule.get("id_field"):
                                    sl_id_val = sl_row.get(sl_rule["id_field"])
                                    if sl_id_val in (None, ""):
                                        sl_row[sl_rule["id_field"]] = f"{p_id}_ext{si + 1}"
                                result[sl_rule["sheet"]].append(sl_row)

    return result


# ── Excel 写入 ────────────────────────────────────────────

def write_to_excel(data: dict, excel_schemas: dict, output_path: str):
    """将表数据写入 Excel。所有 schema 都创建 sheet"""
    wb = Workbook()
    wb.remove(wb.active)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

    for sname, schema in excel_schemas.items():
        ws = wb.create_sheet(title=sname[:31])
        field_names = [f["name"] for f in schema["fields"]]
        rows = data.get(sname, [])

        for col, name in enumerate(field_names, 1):
            cell = ws.cell(row=1, column=col, value=name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        for row_idx, item in enumerate(rows, 2):
            for col_idx, fn in enumerate(field_names, 1):
                val = item.get(fn, "")
                if val == "" or val is None:
                    val = None
                ws.cell(row=row_idx, column=col_idx, value=val)

        for col_idx, name in enumerate(field_names, 1):
            max_len = len(name)
            for row_idx in range(2, len(rows) + 2):
                cell_val = ws.cell(row=row_idx, column=col_idx).value
                if cell_val is not None:
                    max_len = max(max_len, len(str(cell_val)))
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 50)

    wb.save(output_path)


# ── 主流程 ────────────────────────────────────────────────

def build_scene(scene_description: str) -> str:
    """场景构建主入口：加载类型定义 → LLM 提取 → 显式映射分发 → 写入 Excel"""
    type_schemas = load_type_schemas(str(TYPES_DIR))
    if not type_schemas:
        return "错误：schema/scene/types 目录为空或无法加载"

    excel_schemas = load_excel_schemas(str(EXCEL_DIR))
    if not excel_schemas:
        return "错误：schema/scene/excel 目录为空或无法加载"

    entities = extract_entities(scene_description, type_schemas)
    if "error" in entities:
        return f"提取失败：{entities['error']}"

    if os.environ.get("DEBUG_SCENE"):
        dump_path = BASE_DIR / "scene"
        dump_path.mkdir(exist_ok=True)
        with open(str(dump_path / "llm_parsed.json"), "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)

    try:
        table_data = map_to_tables(entities, type_schemas, excel_schemas)
    except Exception as e:
        return f"映射失败：{e}"

    output_dir = BASE_DIR / "scene"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / "scene.xlsx")

    try:
        write_to_excel(table_data, excel_schemas, output_path)
    except Exception as e:
        return f"写入 Excel 失败：{e}"

    row_counts = {k: len(v) for k, v in table_data.items() if v}
    summary = "、".join(f"{k}({v}条)" for k, v in row_counts.items())
    return f"场景数据已保存到：{output_path}\n已填充表：{summary}" if summary else f"场景数据已保存到：{output_path}"


if __name__ == "__main__":
    import sys
    desc = sys.argv[1] if len(sys.argv) > 1 else input("请输入场景描述：")
    result = build_scene(desc)
    print(result)
