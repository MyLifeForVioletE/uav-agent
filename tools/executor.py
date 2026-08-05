"""
工具调用执行器：参数校验 → 去重执行 → 结果总结
"""
"""
工具调用执行器：参数校验 → 去重执行 → 结果总结
"""
import json

from core.config import BASE_DIR
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

# ── 参数校验辅助函数 ────────────────────────────────────

# 占位文本特征词：LLM 生成的参数值中包含这些词即视为未填写真实值
_PLACEHOLDER_PATTERNS = [
    "请提供", "请输入", "请确认", "请问", "请指定", "请告知",
    "请检查", "请确保", "请给出", "请填写", "请补充",
    "您的", "你的", "相应的", "正确的",
    "__ask_user__",
]


def _is_placeholder(val):
    """判断参数值是否为占位文本（LLM 未提供真实值，仅暂时填充）"""
    if not isinstance(val, str):
        return False
    if len(val) > 80:
        return True
    return any(kw in val for kw in _PLACEHOLDER_PATTERNS)


def _user_mentioned(value, messages):
    """检查参数值是否在用户的历史消息中出现过（防止 LLM 自行猜测）"""
    if not isinstance(value, str):
        return True
    v = value.lower().replace("\\\\", "\\")
    for msg in messages:
        if type(msg).__name__ == "HumanMessage" and isinstance(msg.content, str):
            if v in msg.content.lower().replace("\\\\", "\\"):
                return True
    return False


def _load_param_descriptions():
    """从 algorithms.json 加载参数的中文描述，用于生成缺失参数的提示"""
    path = BASE_DIR / "algorithms.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    algos = data.get("capabilities", [])
    desc = {}
    for algo in algos:
        for pname, pinfo in algo.get("input_schema", {}).items():
            d = pinfo.get("description", "")
            if "。" in d:
                d = d.split("。")[0] + "。"
            elif "，" in d:
                d = d.split("，")[0]
            desc[pname] = d
    return desc


# 模块级缓存：参数名 → 中文描述
PARAM_DESC = _load_param_descriptions()


def _load_tool_descriptions():
    """从 algorithms.json 加载工具（算法）的整体描述"""
    path = BASE_DIR / "algorithms.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {a["name"]: a.get("description", "") for a in data.get("capabilities", [])}


# 模块级缓存：工具名 → 描述
TOOL_DESC = _load_tool_descriptions()


async def execute_tool_calls(response, tool_map, messages, llm_with_tools,
                              last_calc_type, last_file_path,
                              plan_generated=False, skip_summary=False, quiet=False,
                              skip_param_check=False):
    """
    执行 LLM 发起的工具调用：参数校验 → 去重执行 → 结果输出 → 总结。
    返回 (should_break, last_calc_type, last_file_path, plan_generated, plan_steps)
    """
    plan_steps = []
    tcs = response.tool_calls
    _print = lambda *a, **kw: None if quiet else print(*a, **kw)

    # ── 步骤1：校验参数是否包含占位文本 ──────────────────
    bad_params = set()
    first_tool = None
    if not skip_param_check:
        for tc in tcs:
            if first_tool is None:
                first_tool = tc["name"]
            for k, v in tc["args"].items():
                if _is_placeholder(v):
                    bad_params.add(k)
    if bad_params:
        items = "、".join(PARAM_DESC.get(n, n) for n in bad_params)
        if first_tool and TOOL_DESC.get(first_tool):
            text = f"需要{TOOL_DESC[first_tool]}，请提供：{items}"
        else:
            text = f"请提供：{items}"
        messages.append(AIMessage(content=text))
        _print(f"\n{text}", flush=True)
        return True, last_calc_type, last_file_path, plan_generated, plan_steps

    # ── 步骤2：校验参数是否在用户消息中出现过 ────────────
    guessed = set()
    if not skip_param_check:
        for tc in tcs:
            for k, v in tc["args"].items():
                if not _user_mentioned(v, messages):
                    guessed.add(k)
    if guessed:
        items = "、".join(PARAM_DESC.get(n, n) for n in guessed)
        text = f"缺少以下信息：{items}"
        messages.append(AIMessage(content=text))
        _print(f"\n{text}", flush=True)
        return True, last_calc_type, last_file_path, plan_generated, plan_steps

    # ── 步骤3：检测计算类型切换后是否复用了旧路径 ────────
    reuse_old = any(
        tc["args"].get("calc_type") is not None
        and last_calc_type is not None
        and tc["args"]["calc_type"] != last_calc_type
        and tc["args"].get("param_xml_path") == last_file_path
        for tc in tcs
    )
    if reuse_old:
        text = "计算类型已改变，请提供新的参数XML文件路径"
        messages.append(AIMessage(content=text))
        _print(f"\n{text}", flush=True)
        return True, last_calc_type, last_file_path, plan_generated, plan_steps

    # ── 步骤4：输出 LLM 的文本内容 ──────────────────────
    if response.content and response.content.strip():
        _print(f"\n{response.content}", flush=True)

    # ── 步骤5：执行工具调用（去重） ──────────────────────
    messages.append(response)
    seen = set()
    for tc in tcs:
        key = (tc["name"], json.dumps(tc["args"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        try:
            result = await tool_map[tc["name"]].ainvoke(tc["args"])
        except Exception as e:
            result = f"工具执行异常: {e}"
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        calc = tc["args"].get("calc_type")
        fp = tc["args"].get("param_xml_path")
        if calc is not None:
            last_calc_type = calc
        if fp is not None:
            last_file_path = fp

    # ── 步骤6：总结本轮执行结果 ──────────────────────────
    if skip_summary:
        return False, last_calc_type, last_file_path, plan_generated, plan_steps

    hint = SystemMessage(content="用一句话告知用户执行结果，然后自然进入下一步（如果有），等待用户提供所需参数。")
    messages.append(hint)
    summary = await llm_with_tools.ainvoke(messages)
    messages.pop()
    if summary.tool_calls:
        summary = AIMessage(content="执行完成")
    messages.append(summary)
    _print(f"\n{summary.content}", flush=True)

    return False, last_calc_type, last_file_path, plan_generated, plan_steps
