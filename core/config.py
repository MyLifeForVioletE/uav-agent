"""项目全局配置常量"""
from pathlib import Path

# 项目根目录（所有子模块通过此常量定位资源文件）
BASE_DIR = Path(__file__).parent.parent

# Ollama 服务地址与模型名称
OLLAMA_BASE = "http://127.0.0.1:11434"
MODEL = "ExpedientFalcon/qwen3-4b-agent:latest"

# Redis 配置
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None

# Kafka 消息总线配置（三种 agent 间通信：指挥/UAV/分析）
KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"

# Agent ID 常量
COORDINATOR_ID = "_coordinator_"
INFO_PROCESSOR_ID = "_info_processor_"
