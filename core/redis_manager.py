"""
Redis 管理模块：会话状态存储、消息历史、子agent状态同步
"""
import json
import sys
from typing import Any, Optional
from datetime import datetime

import redis
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from core.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD


class RedisManager:
    """Redis 管理器：负责会话状态、消息历史、子agent状态的持久化存储"""
    
    def __init__(self):
        """初始化 Redis 连接池"""
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # 测试连接
            self.client.ping()
            sys.stderr.write(f"[Redis] 连接成功: {REDIS_HOST}:{REDIS_PORT}\n")
            sys.stderr.flush()
        except redis.ConnectionError as e:
            sys.stderr.write(f"[Redis] 连接失败: {e}\n")
            sys.stderr.flush()
            self.client = None
        except Exception as e:
            sys.stderr.write(f"[Redis] 初始化异常: {e}\n")
            sys.stderr.flush()
            self.client = None
    
    def _is_available(self) -> bool:
        """检查 Redis 是否可用"""
        if self.client is None:
            return False
        try:
            self.client.ping()
            return True
        except:
            self.client = None
            return False
    
    # ═══════════════════════════════════════════
    #  会话存储
    # ═══════════════════════════════════════════
    
    def save_session(self, session_id: str, state: dict) -> bool:
        """保存用户会话状态"""
        if not self._is_available():
            return False
        
        try:
            # 序列化 state（处理特殊类型）
            serializable_state = self._serialize_state(state)
            
            # 保存会话状态
            key = f"session:{session_id}"
            self.client.set(key, json.dumps(serializable_state, ensure_ascii=False))
            
            # 更新会话索引
            self.client.sadd("sessions:index", session_id)
            
            # 设置过期时间（24小时）
            self.client.expire(key, 86400)
            
            sys.stderr.write(f"[Redis] 保存会话: {session_id}\n")
            sys.stderr.flush()
            return True
        except Exception as e:
            sys.stderr.write(f"[Redis] 保存会话失败: {e}\n")
            sys.stderr.flush()
            return False
    
    # ═══════════════════════════════════════════
    #  消息历史
    # ═══════════════════════════════════════════
    
    def append_message(self, session_id: str, message: dict) -> bool:
        """追加对话消息"""
        if not self._is_available():
            return False
        
        try:
            key = f"session:{session_id}:messages"
            
            # 序列化消息
            serialized_msg = self._serialize_message(message)
            
            # 使用 List 存储消息
            self.client.rpush(key, json.dumps(serialized_msg, ensure_ascii=False))
            
            # 限制消息数量（保留最近100条）
            self.client.ltrim(key, -100, -1)
            
            # 设置过期时间（24小时）
            self.client.expire(key, 86400)
            
            return True
        except Exception as e:
            sys.stderr.write(f"[Redis] 追加消息失败: {e}\n")
            sys.stderr.flush()
            return False
    
    # ═══════════════════════════════════════════
    #  通用工具方法
    # ═══════════════════════════════════════════
    
    def save_agent_state(self, agent_id: str, state: dict) -> bool:
        """保存子agent状态"""
        if not self._is_available():
            return False
        
        try:
            key = f"agent:{agent_id}:state"
            serializable_state = self._serialize_state(state)
            self.client.set(key, json.dumps(serializable_state, ensure_ascii=False))
            
            # 更新agent索引
            self.client.sadd("agents:index", agent_id)
            
            # 设置过期时间（1小时）
            self.client.expire(key, 3600)
            
            sys.stderr.write(f"[Redis] 保存agent状态: {agent_id}\n")
            sys.stderr.flush()
            return True
        except Exception as e:
            sys.stderr.write(f"[Redis] 保存agent状态失败: {e}\n")
            sys.stderr.flush()
            return False
    
    # ═══════════════════════════════════════════
    #  通用工具方法
    # ═══════════════════════════════════════════
    
    def set(self, key: str, value: Any, expire: int = None) -> bool:
        """通用键值存储"""
        if not self._is_available():
            return False
        
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            self.client.set(key, serialized, ex=expire)
            return True
        except Exception as e:
            sys.stderr.write(f"[Redis] set失败: {e}\n")
            sys.stderr.flush()
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """通用键值获取"""
        if not self._is_available():
            return None
        
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            sys.stderr.write(f"[Redis] get失败: {e}\n")
            sys.stderr.flush()
            return None
    
    # ═══════════════════════════════════════════
    #  参数缓存（结构化 JSON 对象）
    # ═══════════════════════════════════════════
    
    def _context_key(self, session_id: str) -> str:
        return f"context:{session_id}"
    
    def init_context(self, session_id: str) -> bool:
        """从 core/context_template.json 模板初始化上下文，不覆盖已有值"""
        if not self._is_available():
            return False
        from core.config import BASE_DIR
        template_path = BASE_DIR / "core" / "context_template.json"
        if not template_path.is_file():
            sys.stderr.write("[Redis] context_template.json 不存在\n")
            sys.stderr.flush()
            return False
        try:
            template = json.loads(template_path.read_text(encoding="utf-8"))
            existing = self.get_context(session_id)
            if existing:
                merged = self._deep_merge(template, existing)
                return self.save_context(session_id, merged)
            return self.save_context(session_id, template)
        except Exception as e:
            sys.stderr.write(f"[Redis] init_context 失败: {e}\n")
            sys.stderr.flush()
            return False
    
    def _deep_merge(self, base, override):
        """递归深合并，支持 dict+dict、list+dict(按索引或ID)、list+list"""
        if isinstance(base, dict) and isinstance(override, dict):
            result = dict(base)
            for k, v in override.items():
                if k in result:
                    result[k] = self._deep_merge(result[k], v)
                else:
                    result[k] = v
            return result
        
        if isinstance(base, list) and isinstance(override, dict):
            result = list(base)
            for key, val in override.items():
                try:
                    idx = int(key)
                except ValueError:
                    # 非数字 key → 按 id 字段匹配
                    idx = None
                    for i, item in enumerate(result):
                        if isinstance(item, dict) and item.get("id") == key:
                            idx = i
                            break
                if idx is not None and 0 <= idx < len(result):
                    result[idx] = self._deep_merge(result[idx], val)
                elif idx is not None:
                    while len(result) < idx:
                        result.append(None)
                    result.append(val)
            return result
        
        if isinstance(base, list) and isinstance(override, list):
            result = list(base)
            for i, val in enumerate(override):
                if i < len(result):
                    result[i] = self._deep_merge(result[i], val)
                else:
                    result.append(val)
            return result
        
        return override
        
        return override
    
    def save_context(self, session_id: str, context: dict) -> bool:
        """保存结构化上下文对象到 Redis"""
        return self.set(self._context_key(session_id), context, expire=86400)
    
    def get_context(self, session_id: str) -> Optional[dict]:
        """获取结构化上下文对象"""
        raw = self.get(self._context_key(session_id))
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        return {}
    
    def update_context(self, session_id: str, new_data: dict) -> bool:
        """合并更新结构化上下文（递归深合并，保留模板结构）"""
        current = self.get_context(session_id)
        if not current:
            current = {}
        merged = self._deep_merge(current, new_data)
        return self.save_context(session_id, merged)
    
    # ═══════════════════════════════════════════
    #  序列化/反序列化
    # ═══════════════════════════════════════════
    
    def _serialize_state(self, state: dict) -> dict:
        """序列化 AgentState（处理特殊类型）"""
        serialized = {}
        
        for key, value in state.items():
            # 跳过不可序列化的对象
            if key == "_fleet_manager" or key == "_last_response":
                serialized[key] = None
                continue
            
            # 处理 LangChain Message 对象
            if key == "messages" and isinstance(value, list):
                serialized[key] = [self._serialize_message(msg) for msg in value]
            # 处理 set 类型
            elif isinstance(value, set):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def _deserialize_state(self, state: dict) -> dict:
        """反序列化 AgentState"""
        deserialized = {}
        
        for key, value in state.items():
            # 处理 messages 字段
            if key == "messages" and isinstance(value, list):
                deserialized[key] = [self._deserialize_message(msg) for msg in value]
            # 处理 _skill_injected 字段（存储为 list，恢复为 set）
            elif key == "_skill_injected" and isinstance(value, list):
                deserialized[key] = set(value)
            else:
                deserialized[key] = value
        
        return deserialized
    
    def _serialize_message(self, message) -> dict:
        """序列化 LangChain Message 对象"""
        if isinstance(message, dict):
            # 已经是序列化格式
            return message
        
        if isinstance(message, HumanMessage):
            return {"type": "human", "content": message.content}
        elif isinstance(message, AIMessage):
            result = {"type": "ai", "content": message.content}
            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = message.tool_calls
            return result
        elif isinstance(message, ToolMessage):
            return {"type": "tool", "content": message.content, "tool_call_id": message.tool_call_id}
        elif isinstance(message, SystemMessage):
            return {"type": "system", "content": message.content}
        else:
            # 未知类型，尝试转换为字符串
            return {"type": "unknown", "content": str(message)}
    
    def _deserialize_message(self, msg_dict: dict):
        """反序列化消息为 LangChain Message 对象"""
        msg_type = msg_dict.get("type", "unknown")
        content = msg_dict.get("content", "")
        
        if msg_type == "human":
            return HumanMessage(content=content)
        elif msg_type == "ai":
            tool_calls = msg_dict.get("tool_calls", [])
            if tool_calls:
                return AIMessage(content=content, tool_calls=tool_calls)
            return AIMessage(content=content)
        elif msg_type == "tool":
            tool_call_id = msg_dict.get("tool_call_id", "")
            return ToolMessage(content=content, tool_call_id=tool_call_id)
        elif msg_type == "system":
            return SystemMessage(content=content)
        else:
            # 未知类型，返回 HumanMessage
            return HumanMessage(content=content)
    
    def close(self):
        """关闭 Redis 连接"""
        if self.client:
            try:
                self.client.close()
                sys.stderr.write("[Redis] 连接已关闭\n")
                sys.stderr.flush()
            except:
                pass


# 全局单例
_redis_manager: Optional[RedisManager] = None


def get_redis_manager() -> RedisManager:
    """获取 Redis 管理器单例"""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager
