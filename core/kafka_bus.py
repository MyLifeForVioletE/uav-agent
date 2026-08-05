"""Kafka 消息总线：指挥/UAV/分析三种 agent 之间的通信层

Topic 约定：每个 agent 一个收件箱 topic `agent.inbox.<agent_id>`，
每个 agent 一个 consumer group（group.id = 自己的 agent_id），
读取即取走（enable_auto_commit=False + 显式 commit）。

消息结构（JSON）：
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
import json
import sys
import time
import uuid
from typing import Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from core.config import KAFKA_BOOTSTRAP_SERVERS

# 消息类型
MSG_INSTRUCTION = "instruction"   # 指挥→子：指令/任务调整
MSG_REPORT = "report"             # 子→指挥：上报状态/结果
MSG_REQUEST = "request"           # 子→指挥：请示/请求数据（等待回复）
MSG_REPLY = "reply"               # 指挥→子：对请示的回复
MSG_RESULT = "result"             # 子→指挥：算法/任务结果回传

VALID_MSG_TYPES = {MSG_INSTRUCTION, MSG_REPORT, MSG_REQUEST, MSG_REPLY, MSG_RESULT}


def inbox_topic(agent_id: str) -> str:
    """agent_id → 收件箱 topic"""
    return f"agent.inbox.{agent_id}"


def new_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


class KafkaBus:
    """Kafka 消息总线：send 生产消息，poll 拉取并消费本 agent 收件箱"""

    def __init__(self, bootstrap: str = None):
        self._bootstrap = bootstrap or KAFKA_BOOTSTRAP_SERVERS
        self._run_id = uuid.uuid4().hex[:8]
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumers: dict[str, AIOKafkaConsumer] = {}
        self._lock = asyncio.Lock()

    async def _get_producer(self) -> AIOKafkaProducer:
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            )
            await self._producer.start()
        return self._producer

    async def _get_consumer(self, agent_id: str) -> AIOKafkaConsumer:
        if agent_id not in self._consumers:
            # 每个进程用独立消费组（agent_id + run_id），避免跨进程残留偏移/僵死成员阻塞
            consumer = AIOKafkaConsumer(
                inbox_topic(agent_id),
                bootstrap_servers=self._bootstrap,
                group_id=f"{agent_id}.{self._run_id}",
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            )
            await consumer.start()
            self._consumers[agent_id] = consumer
        return self._consumers[agent_id]

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
        producer = await self._get_producer()
        await producer.send(inbox_topic(to_id), value=msg)
        sys.stderr.write(f"[KafkaBus] {from_id} --{msg_type}--> {to_id}: {content[:80]}\n")
        sys.stderr.flush()
        return msg["msg_id"]

    async def poll(self, agent_id: str, max_messages: int = 20, timeout: float = 0.1) -> list[dict]:
        """拉取并确认消费收件箱中的消息（读取即取走）"""
        consumer = await self._get_consumer(agent_id)
        records = await consumer.getmany(
            max_records=max_messages, timeout_ms=int(timeout * 1000)
        )
        messages = []
        for tp, batch in records.items():
            for rec in batch:
                messages.append(rec.value)
            await consumer.commit({tp: batch[-1].offset + 1})
        return messages

    async def flush_inbox(self, agent_id: str):
        """消费并丢弃收件箱中全部残留消息（一次性清理，供启动时使用）"""
        while True:
            msgs = await self.poll(agent_id, max_messages=500, timeout=0.2)
            if not msgs:
                break

    async def close(self):
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
        for c in self._consumers.values():
            await c.stop()
        self._consumers.clear()


_bus: Optional[KafkaBus] = None


def get_kafka_bus() -> KafkaBus:
    """进程内单例：所有 agent 共享同一总线"""
    global _bus
    if _bus is None:
        _bus = KafkaBus()
    return _bus
