# GPT Service - 推理模型服务

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个即插即拔的GPT推理服务，支持本地模型训练（LoRA微调）和Ollama高级代理功能。

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API文档](#api文档)
- [配置说明](#配置说明)
- [使用示例](#使用示例)

## 🎯 项目简介

GPT Service是一个完整的推理模型服务，主要功能包括：

- **本地模型训练**：使用LoRA/PEFT进行参数高效微调
- **Ollama代理**：高级代理功能，支持模型切换、负载均衡、缓存
- **完整API**：文本生成、流式生成、模型管理、训练、批量处理、评估等
- **生产就绪**：完善的日志系统、环境配置分离、Docker支持

## ✨ 功能特性

### 核心功能
- ✅ **本地模型支持**：支持HuggingFace模型加载和推理
- ✅ **LoRA微调**：使用PEFT进行参数高效微调
- ✅ **Ollama代理**：智能代理，支持负载均衡和缓存
- ✅ **流式生成**：支持Server-Sent Events (SSE)流式输出
- ✅ **模型管理**：动态加载、卸载、切换模型
- ✅ **训练管理**：异步训练任务，支持进度跟踪

### 技术特性
- ✅ **环境分离**：开发/生产环境配置分离
- ✅ **结构化日志**：JSON格式日志，支持文件轮转
- ✅ **Docker支持**：完整的容器化部署方案
- ✅ **健康检查**：完善的健康检查和监控
- ✅ **类型注解**：完整的Python类型注解支持

## 📁 项目结构

```
gpt-service/
├── main.py                    # FastAPI应用入口
├── requirements.txt           # Python依赖
├── Dockerfile                 # Docker构建文件
├── .dockerignore             # Docker忽略文件
├── .env.example              # 环境变量示例
├── README.md                 # 项目文档
│
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── config.py             # 配置管理
│   ├── logger.py              # 日志系统
│   ├── model_manager.py      # 模型管理器
│   ├── ollama_proxy.py       # Ollama代理
│   ├── trainer.py            # 训练模块
│   ├── evaluator.py          # 评估模块
│   └── utils.py              # 工具函数
│
├── config/                   # 配置文件目录
│   ├── dev.yaml              # 开发环境配置
│   ├── prod.yaml             # 生产环境配置
│   └── models.yaml           # 模型配置
│
├── models/                   # 模型存储目录
│   ├── base/                 # 基础模型
│   ├── lora/                 # LoRA适配器
│   └── checkpoints/          # 训练检查点
│
├── scripts/                  # 脚本目录
│   ├── train_lora.py         # 训练脚本
│   └── evaluate_model.py     # 评估脚本
│
└── docs/                     # 文档目录
    ├── API_DOCUMENTATION.md
    ├── TRAINING_GUIDE.md
    └── DEPLOYMENT.md
```

## 🚀 快速开始

### 环境要求

- Python 3.10 或更高版本
- pip 包管理器
- 至少 8GB 可用内存（用于加载模型）
- NVIDIA GPU（推荐，用于训练和推理加速）
- 可选：Docker（用于容器化部署）

### 安装步骤

#### 1. 克隆项目

```bash
cd gpt-service
```

#### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 安装GPU版本的PyTorch（推荐）

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 5. 配置环境

复制环境变量示例文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置环境变量。

### 运行服务

```bash
python main.py
```

或者使用uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后，访问：
- **API文档**：http://localhost:8000/docs
- **交互式API文档**：http://localhost:8000/redoc
- **健康检查**：http://localhost:8000/health

## 📚 API文档

### 基础信息

- **Base URL**: `http://localhost:8000`
- **API版本**: v1.0.0
- **Content-Type**: `application/json`

### 主要接口

#### 1. 文本生成

**POST** `/generate`

生成文本内容。

#### 2. 流式生成

**POST** `/generate/stream`

流式生成文本（SSE格式）。

#### 3. 对话接口

**POST** `/chat`

多轮对话接口。

#### 4. 模型列表

**GET** `/models`

获取所有可用模型列表。

#### 5. 训练任务

**POST** `/train`

启动LoRA微调训练任务。

更多API文档请访问 http://localhost:8000/docs

## ⚙️ 配置说明

### 环境变量

通过 `.env` 文件或环境变量配置：

- `ENVIRONMENT`: 环境类型（dev/prod）
- `LOG_LEVEL`: 日志级别（DEBUG/INFO/WARNING/ERROR）
- `API_HOST`: API服务地址
- `API_PORT`: API服务端口
- `OLLAMA_BASE_URL`: Ollama服务地址

### 配置文件

编辑 `config/dev.yaml` 或 `config/prod.yaml` 来配置服务参数。

## 💡 使用示例

### Python示例

```python
import requests

url = "http://localhost:8000/generate"

data = {
    "prompt": "你好，请介绍一下人工智能。",
    "model_name": "qwen2-7b",
    "max_tokens": 512
}

response = requests.post(url, json=data)
result = response.json()
print(result["text"])
```

### cURL示例

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好，请介绍一下人工智能。",
    "model_name": "qwen2-7b"
  }'
```

## 📝 更新日志

### v1.0.0 (2024-12)

- ✨ **初始版本**：支持本地模型和Ollama代理
- ✨ **LoRA微调**：支持参数高效微调
- ✨ **流式生成**：支持SSE流式输出
- ✅ 完善的日志系统和配置管理
- ✅ Docker支持

## 📄 许可证

本项目采用 MIT 许可证。

---

**注意**：本项目仅供学习和研究使用。在生产环境中使用前，请进行充分的测试和优化。

