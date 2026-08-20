"""进程内消息总线：指挥/UAV/分析三种 agent 之间的通信层（内存实现，替代 Kafka）

所有 agent 运行在同一个进程内，不再依赖外部 Kafka broker：
每个 agent 一个收件箱队列（asyncio.Queue），send 直接 put，poll 即时取走，
无网络/序列化开销，无需轮询等待。

消息结构（dict）：
{
  "msg_id": "msg_xxx",
  "from": "UAV_1",
  "to": "_coordinator_",
  "msg_type": "instruction | report | request | reply | result",
  "content": "消息正文（中文）",
  "payload": {...},
  "correlation_id": "用于 request/reply 配对",
  "ts": 时间戳
}
"""
import asyncio
import sys
import time
import uuid
from typing import Optional

# 消息类型
MSG_INSTRUCTION = "instruction"   # 指挥→子：指令/任务调整
MSG_REPORT = "report"             # 子→指挥：上报状态/结果
MSG_REQUEST = "request"           # 子→指挥：请示/请求数据（等待回复）
MSG_REPLY = "reply"               # 指挥→子：对请示的回复
MSG_RESULT = "result"             # 子→指挥：算法/任务结果回传
MSG_TASK = "task"                 # 指挥→子：派发子任务（payload 携带任务对象）

VALID_MSG_TYPES = {MSG_INSTRUCTION, MSG_REPORT, MSG_REQUEST, MSG_REPLY, MSG_RESULT, MSG_TASK}


def new_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


class MemoryBus:
    """进程内消息总线：send 投递到目标收件箱队列，poll 即时批量取走"""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._events: dict[str, asyncio.Event] = {}

    def _queue(self, agent_id: str) -> asyncio.Queue:
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
        return self._queues[agent_id]

    def _event(self, agent_id: str) -> asyncio.Event:
        if agent_id not in self._events:
            self._events[agent_id] = asyncio.Event()
        return self._events[agent_id]

    async def send(self, from_id: str, to_id: str, msg_type: str, content: str,
                   payload: dict = None, correlation_id: str = None) -> str:
        """向目标 agent 的收件箱投递一条消息，返回 msg_id"""
        if msg_type not in VALID_MSG_TYPES:
            raise ValueError(f"非法消息类型: {msg_type}")
        msg = {
            "msg_id": new_msg_id(),
            "from": from_id,
            "to": to_id,
            "msg_type": msg_type,
            "content": content,
            "payload": payload or {},
            "correlation_id": correlation_id,
            "ts": time.time(),
        }
        self._queue(to_id).put_nowait(msg)
        self._event(to_id).set()
        sys.stderr.write(f"[Bus] {from_id} --{msg_type}--> {to_id}: {content[:80]}\n")
        sys.stderr.flush()
        return msg["msg_id"]

    async def poll(self, agent_id: str, max_messages: int = 20, timeout: float = 0.1) -> list[dict]:
        """即时批量取走收件箱中的消息（内存队列，无网络等待）"""
        q = self._queue(agent_id)
        messages = []
        while len(messages) < max_messages:
            try:
                messages.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        if messages:
            self._event(agent_id).clear()
        return messages

    async def wait_message(self, agent_id: str, timeout: float = None) -> list[dict]:
        """阻塞等待直到该 agent 收件箱有新消息，然后一次性取走全部（供事件驱动场景）"""
        ev = self._event(agent_id)
        if ev.is_set():
            return await self.poll(agent_id)
        try:
            if timeout is None:
                await ev.wait()
            else:
                await asyncio.wait_for(ev.wait(), timeout)
        except asyncio.TimeoutError:
            return []
        return await self.poll(agent_id)

    async def flush_inbox(self, agent_id: str):
        """清空收件箱中全部残留消息（一次性清理，供启动时使用）"""
        q = self._queue(agent_id)
        while True:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._event(agent_id).clear()

    async def close(self):
        self._queues.clear()
        self._events.clear()


_bus: Optional[MemoryBus] = None


def get_message_bus() -> MemoryBus:
    """进程内单例：所有 agent 共享同一总线"""
    global _bus
    if _bus is None:
        _bus = MemoryBus()
    return _bus