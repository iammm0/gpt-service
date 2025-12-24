"""
GPT Service - 推理模型服务入口

提供本地模型训练、Ollama代理和完整的推理API服务。
"""

import uvicorn
from src.app import create_app
from src.config import get_config

# 创建FastAPI应用实例
app = create_app()

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_config=None  # 使用自定义日志配置
    )

