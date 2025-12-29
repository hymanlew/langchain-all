import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException
from fastapi.params import Query
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from workflow import graph
from auth import verify_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="企业级 LLM 网关服务", version="1.0")

# 请求/响应模型
class ChatRequest(BaseModel):
    user_input: Query(min_length=1, max_length=1000)
    model: str = "gpt-3.5-turbo"  # 默认模型，实际对应 One‑API 中配置的模型标识

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
async def health():
    return {"status": "healthy"}

"""
from fastapi import Security, FastAPI
from .db import User
from .security import get_current_active_user

@app.get("/users/me/items/")
async def read_own_items(
    # 像FastAPI、Pydantic等框架利用这些元数据来做验证、依赖注入等
    skip: Annotated[int, Query(ge=0, description="Number of items to skip")],
    limit: Annotated[int, Query(le=100, description="Max items to return")],
    # 从请求的令牌（例如 JWT）中解码用户信息，并检查用户状态和权限
    current_user: Annotated[User, Security(get_current_active_user, scopes=["items"])]
):
    return [{"item_id": "Foo", "owner": current_user.username}]
"""
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    # 依赖注入，声明依赖的函数（依赖项），并自动调用它取得返回值
    api_key: str = Depends(verify_api_key)  # 身份验证
):
    logger.info(f"收到请求，用户: {api_key[:8]}..., 模型: {request.model}")
    try:
        # 调用 LangGraph 工作流
        state = graph.invoke({
            "user_input": request.user_input,
            "model": request.model
        })
        return ChatResponse(answer=state["answer"])
    except Exception as e:
        logger.exception("工作流执行异常")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    async def event_generator() -> AsyncGenerator[str, None]:
        yield json.dumps({"event": "done", "final": full_response}) + "\n"
    return StreamingResponse(event_generator())