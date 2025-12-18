"""
GPT Service - 推理模型服务

提供本地模型训练、Ollama代理和完整的推理API服务。
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_config
from src.logger import setup_logger
from src.model_manager import ModelManager
from src.ollama_proxy import OllamaProxy
from src.trainer import LoRATrainer
from src.evaluator import ModelEvaluator

# 初始化日志系统
logger = setup_logger(__name__)

# 全局管理器
_model_manager: ModelManager = None
_ollama_proxy: OllamaProxy = None
_trainer: LoRATrainer = None
_evaluator: ModelEvaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _model_manager, _ollama_proxy, _trainer, _evaluator
    
    # 启动时初始化
    config = get_config()
    logger.info("=" * 60)
    logger.info("GPT Service 启动中...")
    logger.info("=" * 60)
    
    try:
        # 初始化模型管理器
        _model_manager = ModelManager(config)
        logger.info("模型管理器初始化完成")
        
        # 初始化Ollama代理
        _ollama_proxy = OllamaProxy(config)
        await _ollama_proxy.initialize()
        logger.info("Ollama代理初始化完成")
        
        # 初始化训练器
        _trainer = LoRATrainer(config)
        logger.info("训练器初始化完成")
        
        # 初始化评估器
        _evaluator = ModelEvaluator(_model_manager, config)
        logger.info("评估器初始化完成")
        
        logger.info("=" * 60)
        logger.info("GPT Service 启动完成")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        raise
    finally:
        # 关闭时清理
        if _ollama_proxy:
            await _ollama_proxy.close()
        logger.info("GPT Service 关闭")


# 创建FastAPI应用
app = FastAPI(
    title="GPT Service",
    description="推理模型服务 - 支持本地模型训练和Ollama代理",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """根路径，返回服务信息"""
    config = get_config()
    return {
        "service": "GPT Service",
        "version": "1.0.0",
        "description": "推理模型服务 - 支持本地模型训练和Ollama代理",
        "environment": config.environment,
        "endpoints": {
            "/": "GET - 服务信息",
            "/health": "GET - 健康检查",
            "/models": "GET - 模型列表",
            "/generate": "POST - 文本生成",
            "/generate/stream": "POST - 流式生成",
            "/chat": "POST - 对话接口",
            "/train": "POST - 启动训练",
            "/ollama/models": "GET - Ollama模型列表"
        }
    }


@app.get("/health")
def health_check():
    """健康检查接口"""
    try:
        status = {
            "status": "healthy",
            "service": "gpt-service",
            "model_manager": _model_manager is not None,
            "ollama_proxy": _ollama_proxy is not None if _ollama_proxy else False,
            "trainer": _trainer is not None,
            "evaluator": _evaluator is not None
        }
        return status
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


# ========== 请求/响应模型 ==========

class GenerateRequest(BaseModel):
    """文本生成请求"""
    prompt: str = Field(..., description="输入提示")
    model_name: Optional[str] = Field(None, description="模型名称（可选，默认使用当前模型）")
    max_tokens: int = Field(512, description="最大生成token数")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="top-p采样参数")
    top_k: Optional[int] = Field(None, description="top-k采样参数")
    repetition_penalty: float = Field(1.05, ge=1.0, description="重复惩罚")
    use_ollama: bool = Field(False, description="是否使用Ollama代理")


class GenerateResponse(BaseModel):
    """文本生成响应"""
    text: str
    model_name: str
    tokens_generated: Optional[int] = None


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: user, assistant, system")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """对话请求"""
    messages: List[ChatMessage] = Field(..., description="消息列表")
    model_name: Optional[str] = Field(None, description="模型名称")
    max_tokens: int = Field(512, description="最大生成token数")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="温度参数")
    use_ollama: bool = Field(False, description="是否使用Ollama代理")


class BatchGenerateRequest(BaseModel):
    """批量生成请求"""
    prompts: List[str] = Field(..., description="提示列表")
    model_name: Optional[str] = Field(None, description="模型名称")
    max_tokens: int = Field(512, description="最大生成token数")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="温度参数")


class LoadModelRequest(BaseModel):
    """加载模型请求"""
    model_name: str = Field(..., description="模型名称")
    model_path: Optional[str] = Field(None, description="模型路径")
    lora_adapter: Optional[str] = Field(None, description="LoRA适配器路径")


class TrainRequest(BaseModel):
    """训练请求"""
    base_model: str = Field(..., description="基础模型路径或名称")
    dataset_path: str = Field(..., description="数据集路径")
    output_dir: Optional[str] = Field(None, description="输出目录")
    num_epochs: Optional[int] = Field(None, description="训练轮数")
    batch_size: Optional[int] = Field(None, description="批次大小")
    learning_rate: Optional[float] = Field(None, description="学习率")
    dataset_format: str = Field("json", description="数据集格式: json, jsonl, txt")


class EvaluateRequest(BaseModel):
    """评估请求"""
    model_name: str = Field(..., description="模型名称")
    test_data: List[Dict[str, str]] = Field(..., description="测试数据")
    metrics: Optional[List[str]] = Field(None, description="评估指标")


# ========== 模型管理端点 ==========

@app.get("/models")
def list_models():
    """获取模型列表"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        models = _model_manager.list_models()
        return {
            "models": models,
            "current_model": _model_manager.current_model,
            "total_count": len(models)
        }
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    """获取模型详情"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        model_info = _model_manager.get_model(model_name)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        
        return {
            "name": model_info.name,
            "type": model_info.model_type.value,
            "device": model_info.device,
            "lora_adapter": model_info.lora_adapter,
            "is_current": model_name == _model_manager.current_model
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@app.post("/models/load")
def load_model(request: LoadModelRequest):
    """加载模型"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        model_info = _model_manager.load_model(
            model_name=request.model_name,
            model_path=request.model_path,
            lora_adapter=request.lora_adapter
        )
        
        return {
            "status": "success",
            "message": f"模型 {request.model_name} 加载成功",
            "model_name": model_info.name,
            "type": model_info.model_type.value
        }
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.post("/models/unload")
def unload_model(model_name: str):
    """卸载模型"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        success = _model_manager.unload_model(model_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 未找到")
        
        return {
            "status": "success",
            "message": f"模型 {model_name} 已卸载"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"卸载模型失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")


@app.post("/models/switch")
def switch_model(model_name: str):
    """切换模型"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        success = _model_manager.switch_model(model_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 未找到")
        
        return {
            "status": "success",
            "message": f"已切换到模型: {model_name}",
            "current_model": model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换模型失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"切换模型失败: {str(e)}")


# ========== 生成端点 ==========

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """文本生成"""
    try:
        if request.use_ollama:
            if _ollama_proxy is None:
                raise HTTPException(status_code=500, detail="Ollama代理未初始化")
            
            if request.model_name is None:
                raise HTTPException(status_code=400, detail="使用Ollama时必须指定模型名称")
            
            # 使用Ollama生成
            import asyncio
            text = asyncio.run(_ollama_proxy.generate(
                model=request.model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ))
            
            return GenerateResponse(
                text=text,
                model_name=request.model_name
            )
        else:
            if _model_manager is None:
                raise HTTPException(status_code=500, detail="模型管理器未初始化")
            
            # 使用本地模型生成
            text = _model_manager.generate(
                prompt=request.prompt,
                model_name=request.model_name,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty
            )
            
            return GenerateResponse(
                text=text,
                model_name=request.model_name or _model_manager.current_model or "unknown"
            )
    except Exception as e:
        logger.error(f"生成文本失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成文本失败: {str(e)}")


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest):
    """流式文本生成"""
    try:
        if request.use_ollama:
            if _ollama_proxy is None:
                raise HTTPException(status_code=500, detail="Ollama代理未初始化")
            
            if request.model_name is None:
                raise HTTPException(status_code=400, detail="使用Ollama时必须指定模型名称")
            
            # 使用Ollama流式生成
            async def ollama_stream():
                try:
                    result = await _ollama_proxy.generate(
                        model=request.model_name,
                        prompt=request.prompt,
                        stream=True,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p
                    )
                    async for token in result:
                        yield f"data: {token}\n\n"
                except Exception as e:
                    yield f"data: [ERROR: {str(e)}]\n\n"
            
            return StreamingResponse(ollama_stream(), media_type="text/event-stream")
        else:
            if _model_manager is None:
                raise HTTPException(status_code=500, detail="模型管理器未初始化")
            
            # 使用本地模型流式生成
            def local_stream():
                for token in _model_manager.generate_stream(
                    prompt=request.prompt,
                    model_name=request.model_name,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                ):
                    yield f"data: {token}\n\n"
            
            return StreamingResponse(local_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"流式生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"流式生成失败: {str(e)}")


@app.post("/chat")
def chat(request: ChatRequest):
    """对话接口"""
    try:
        # 构建对话提示
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        prompt += "Assistant: "
        
        if request.use_ollama:
            if _ollama_proxy is None:
                raise HTTPException(status_code=500, detail="Ollama代理未初始化")
            
            if request.model_name is None:
                raise HTTPException(status_code=400, detail="使用Ollama时必须指定模型名称")
            
            import asyncio
            response = asyncio.run(_ollama_proxy.generate(
                model=request.model_name,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ))
        else:
            if _model_manager is None:
                raise HTTPException(status_code=500, detail="模型管理器未初始化")
            
            response = _model_manager.generate(
                prompt=prompt,
                model_name=request.model_name,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )
        
        return {
            "response": response,
            "model_name": request.model_name or "unknown"
        }
    except Exception as e:
        logger.error(f"对话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")


@app.post("/batch")
def batch_generate(request: BatchGenerateRequest):
    """批量生成"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        results = []
        for prompt in request.prompts:
            try:
                text = _model_manager.generate(
                    prompt=prompt,
                    model_name=request.model_name,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                results.append({
                    "prompt": prompt,
                    "text": text,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "text": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "results": results,
            "total": len(results),
            "success": sum(1 for r in results if r["status"] == "success")
        }
    except Exception as e:
        logger.error(f"批量生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")


# ========== 训练端点 ==========

@app.post("/train")
def start_training(request: TrainRequest):
    """启动训练任务"""
    try:
        if _trainer is None:
            raise HTTPException(status_code=500, detail="训练器未初始化")
        
        task_id = _trainer.create_task(
            base_model=request.base_model,
            dataset_path=request.dataset_path,
            output_dir=request.output_dir,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            dataset_format=request.dataset_format
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "训练任务已创建"
        }
    except Exception as e:
        logger.error(f"创建训练任务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建训练任务失败: {str(e)}")


@app.get("/train/status/{task_id}")
def get_training_status(task_id: str):
    """获取训练状态"""
    try:
        if _trainer is None:
            raise HTTPException(status_code=500, detail="训练器未初始化")
        
        task = _trainer.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"训练任务 {task_id} 不存在")
        
        return {
            "task_id": task.task_id,
            "status": task.status,
            "progress": task.progress,
            "current_epoch": task.current_epoch,
            "total_epochs": task.total_epochs,
            "loss": task.loss,
            "error": task.error,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "output_dir": task.output_dir
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练状态失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")


@app.get("/train/history")
def get_training_history():
    """获取训练历史"""
    try:
        if _trainer is None:
            raise HTTPException(status_code=500, detail="训练器未初始化")
        
        tasks = _trainer.list_tasks()
        return {
            "tasks": tasks,
            "total": len(tasks)
        }
    except Exception as e:
        logger.error(f"获取训练历史失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取训练历史失败: {str(e)}")


# ========== 评估端点 ==========

@app.post("/evaluate")
def evaluate_model(request: EvaluateRequest):
    """评估模型"""
    try:
        if _evaluator is None:
            raise HTTPException(status_code=500, detail="评估器未初始化")
        
        eval_id = _evaluator.evaluate(
            model_name=request.model_name,
            test_data=request.test_data,
            metrics=request.metrics
        )
        
        return {
            "status": "success",
            "eval_id": eval_id,
            "message": "评估任务已创建"
        }
    except Exception as e:
        logger.error(f"评估模型失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"评估模型失败: {str(e)}")


@app.get("/evaluate/results/{eval_id}")
def get_evaluation_results(eval_id: str):
    """获取评估结果"""
    try:
        if _evaluator is None:
            raise HTTPException(status_code=500, detail="评估器未初始化")
        
        result = _evaluator.get_result(eval_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"评估结果 {eval_id} 不存在")
        
        return {
            "eval_id": result.eval_id,
            "model_name": result.model_name,
            "metrics": result.metrics,
            "samples": result.samples,
            "created_at": result.created_at,
            "completed_at": result.completed_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取评估结果失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取评估结果失败: {str(e)}")


# ========== Ollama管理端点 ==========

@app.get("/ollama/models")
async def list_ollama_models():
    """获取Ollama模型列表"""
    try:
        if _ollama_proxy is None:
            raise HTTPException(status_code=500, detail="Ollama代理未初始化")
        
        models = await _ollama_proxy.list_models()
        return {
            "models": models,
            "total": len(models)
        }
    except Exception as e:
        logger.error(f"获取Ollama模型列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取Ollama模型列表失败: {str(e)}")


@app.post("/ollama/pull")
async def pull_ollama_model(model_name: str):
    """拉取Ollama模型"""
    try:
        if _ollama_proxy is None:
            raise HTTPException(status_code=500, detail="Ollama代理未初始化")
        
        result = await _ollama_proxy.pull_model(model_name)
        return {
            "status": "success",
            "model_name": model_name,
            "result": result
        }
    except Exception as e:
        logger.error(f"拉取Ollama模型失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"拉取Ollama模型失败: {str(e)}")


@app.get("/ollama/stats")
def get_ollama_stats():
    """获取Ollama统计信息"""
    try:
        if _ollama_proxy is None:
            raise HTTPException(status_code=500, detail="Ollama代理未初始化")
        
        stats = _ollama_proxy.get_stats()
        return stats
    except Exception as e:
        logger.error(f"获取Ollama统计信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取Ollama统计信息失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_config=None  # 使用自定义日志配置
    )

