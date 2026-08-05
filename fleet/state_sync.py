"""
状态同步机制：
1. 上行同步（子agent → coordinator）: 状态上报
2. 下行同步（coordinator → 子agent）: 任务注入、参数调整
3. 横向同步（子agent ↔ 子agent）: 通过 coordinator 中转
"""
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import sys


@dataclass
class StateSyncMessage:
    """状态同步消息"""
    msg_type: str        # "status_report" | "task_complete" | "data_share" | "conflict_alert"
    source_uav: str      # 发送方 UAV ID
    target_uav: str | None  # 接收方 UAV ID（None 表示发给 coordinator）
    task_id: str | None
    payload: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StateSync:
    """状态同步管理器"""
    
    def __init__(self):
        self._message_queue: list[StateSyncMessage] = []
    
    # ── 上行同步：子agent → coordinator ──
    
    @staticmethod
    def report_status(sub_agent, task_id: str) -> StateSyncMessage:
        """子agent向coordinator上报当前状态"""
        return StateSyncMessage(
            msg_type="status_report",
            source_uav=sub_agent.uav_id,
            target_uav=None,
            task_id=task_id,
            payload=sub_agent.get_status_dict(),
        )
    
    @staticmethod
    def report_completion(sub_agent, task_id: str, result: dict) -> StateSyncMessage:
        """子agent完成任务后上报结果"""
        return StateSyncMessage(
            msg_type="task_complete",
            source_uav=sub_agent.uav_id,
            target_uav=None,
            task_id=task_id,
            payload={
                "result": result,
                "detail_actions": sub_agent.state.get("detail_actions", []),
                "messages_summary": _summarize_messages(sub_agent.state.get("messages", [])),
            },
        )
    
    @staticmethod
    def report_error(sub_agent, task_id: str, error: str) -> StateSyncMessage:
        """子agent上报错误"""
        return StateSyncMessage(
            msg_type="conflict_alert",
            source_uav=sub_agent.uav_id,
            target_uav=None,
            task_id=task_id,
            payload={"error": error},
        )
    
    # ── 下行同步：coordinator → 子agent ──
    
    @staticmethod
    def inject_shared_data(sub_agent, data_key: str, data: Any):
        """coordinator向子agent注入共享数据（前置任务结果等）"""
        context_msg = f"[协同数据] 来自其他无人机的参考信息 ({data_key}):\n{data}"
        sub_agent.state["messages"].append(sub_agent.state["messages"][-1].__class__(
            content=context_msg
        ))
        sys.stderr.write(f"[StateSync] 注入数据到 {sub_agent.uav_id}: {data_key}\n")
        sys.stderr.flush()
    
    @staticmethod
    def inject_adjustment(sub_agent, adjustment: dict):
        """coordinator向子agent下发调整指令（如修改航线避开冲突）"""
        adjust_msg = (
            f"[协调指令] 根据全局规划，需要调整：\n"
            f"{adjustment.get('description', '')}\n"
            f"请在下一轮规划中考虑此约束。"
        )
        from langchain_core.messages import SystemMessage
        sub_agent.state["messages"].append(SystemMessage(content=adjust_msg))
        sys.stderr.write(f"[StateSync] 注入调整指令到 {sub_agent.uav_id}\n")
        sys.stderr.flush()
    
    # ── 横向同步：子agent → 子agent（通过coordinator中转）──
    
    @staticmethod
    def share_result(from_agent, to_agent, data_key: str, data: Any):
        """通过coordinator中转，在子agent间共享结果"""
        StateSync.inject_shared_data(to_agent, data_key, data)
    
    def queue_message(self, msg: StateSyncMessage):
        """将消息加入队列"""
        self._message_queue.append(msg)
    
    def get_messages_for_uav(self, uav_id: str) -> list[StateSyncMessage]:
        """获取发给指定UAV的所有消息"""
        return [m for m in self._message_queue if m.target_uav == uav_id]
    
    def get_pending_coordinator_messages(self) -> list[StateSyncMessage]:
        """获取等待coordinator处理的消息"""
        return [m for m in self._message_queue if m.target_uav is None]
    
    def clear_processed(self, messages: list[StateSyncMessage]):
        """清除已处理的消息"""
        for msg in messages:
            if msg in self._message_queue:
                self._message_queue.remove(msg)


def _summarize_messages(messages: list) -> str:
    """摘要消息历史（避免全量传输）"""
    summaries = []
    for msg in messages[-10:]:  # 只取最近10条
        role = msg.__class__.__name__
        content = msg.content[:200] if hasattr(msg, 'content') else ""
        summaries.append(f"{role}: {content}")
    return "\n".join(summaries)
