# 多阶段构建 - 构建阶段
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 先安装PyTorch GPU版本（CUDA 12.4）
# 注意：PyTorch GPU版本需要从PyTorch官网安装
# 需要至少 2.6 版本以满足 transformers 的安全要求（CVE-2025-32434）
# 使用 --extra-index-url 而不是 --index-url，避免依赖包元数据冲突
RUN pip install --no-cache-dir --user "torch>=2.6.0" torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# 安装其他Python依赖
# pip会检测到torch已安装并满足版本要求，不会重新安装
RUN pip install --no-cache-dir --user -r requirements.txt

# 运行阶段
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 从构建阶段复制Python包
COPY --from=builder /root/.local /root/.local

# 复制项目代码
COPY . .

# 创建必要的目录
RUN mkdir -p logs models/cache models/base models/lora models/checkpoints

# 设置环境变量
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 创建非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 运行应用
CMD ["python", "main.py"]

