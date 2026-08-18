"""运行耗时统计：各阶段墙钟耗时累计，任务完成时输出性能报告。

用法（全局单例，进程内共享）：
    from core.timing import timing
    with timing.track("阶段名"):
        await do_something()
    timing.add_user_wait(seconds)   # 用户确认/回答等待（不计入执行耗时）

说明：
- 多 agent 并行（asyncio.gather）时各阶段的累计耗时可能大于墙钟总耗时，
  因为并行段重叠。报告会同时给出"墙钟总耗时"与"各阶段累计耗时"。
- 单进程单事件循环内字典累加天然线程安全（无并发写竞争）。
"""
import time
from collections import defaultdict
from contextlib import contextmanager


class Timing:
    def __init__(self):
        self._durations = defaultdict(float)  # 阶段名 -> 累计秒（保序）
        self._counts = defaultdict(int)       # 阶段名 -> 进入次数
        self.init_time = 0.0                  # 启动初始化（MCP连接 + RAG索引）
        self.exec_wall = 0.0                  # 任务执行墙钟总耗时（不含用户确认）
        self.user_wait = 0.0                  # 用户确认/回答等待

    @contextmanager
    def track(self, name: str):
        """累加某阶段墙钟耗时（支持同名并发，各自 t0 独立）"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._durations[name] += time.perf_counter() - t0
            self._counts[name] += 1

    def add(self, name: str, seconds: float):
        self._durations[name] += seconds
        self._counts[name] += 1

    def add_user_wait(self, seconds: float):
        self.user_wait += seconds

    def reset(self):
        """任务完成后清空阶段耗时（保留启动初始化耗时，跨任务共享）"""
        self._durations.clear()
        self._counts.clear()
        self.exec_wall = 0.0
        self.user_wait = 0.0

    def format_report(self) -> str:
        lines = []
        lines.append("=" * 56)
        lines.append("运行耗时统计（不含用户确认/回答等待）")
        lines.append("=" * 56)
        lines.append(f"启动初始化(MCP+RAG索引): {self.init_time:.2f}s")
        lines.append(f"任务执行墙钟总耗时:       {self.exec_wall:.2f}s")
        lines.append(f"用户确认/回答等待(排除):  {self.user_wait:.2f}s")
        lines.append("-" * 56)
        lines.append("各阶段累计耗时（含并行重叠，可能大于墙钟总耗时）:")
        if not self._durations:
            lines.append("  （无）")
        for name, dur in self._durations.items():
            lines.append(f"  {name}: {dur:.2f}s (x{self._counts[name]})")
        lines.append("=" * 56)
        return "\n".join(lines)


timing = Timing()