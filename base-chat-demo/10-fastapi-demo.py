import json
import traceback
import requests

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import base64

from config.config_loader import load_config
from core.api.vision_handler import VisionHandler
from core.utils.util import is_valid_image_file

app = FastAPI(
    title="小智 SERVER 交互API",
    description="设备固件与 server 服务交互接口",
    version="1.0.0",
    # docs_url=None,
    # redoc_url=None,
    # openapi_prefix="/api",
    openapi_tags=[
        {"name": "固件交互", "description": "建立连接/聊天接口"},
        {"name": "配置管理", "description": "固件连接配置管理"},
    ]
)

MAX_FILE_SIZE = 5 * 1024 * 1024

class OTAPostRequest(BaseModel):
    device_id: str
    firmware_version: str
    device_model: str

class OTAPostResponse(BaseModel):
    session_id: str
    expire_time: int

class VisionPostRequest(BaseModel):
    device_id: str
    image_data: str  # Base64 编码的图像数据
    format: str = "jpeg"

class WSRequest(BaseModel):
    device_id: str
    action: str
    target: str
    params: dict

class WSResponse(BaseModel):
    session_id: str
    expire_time: int


@app.get("/xiaozhi/ota/",
         tags=["固件交互"],
         summary="测试服务 OTA 是否正常")
async def check_server():
    """获取当前系统配置"""
    return "OTA接口运行正常，向设备发送的websocket地址是：ws://192.168.1.170:8000/xiaozhi/v1/"


@app.post("/xiaozhi/ota/",
          tags=["固件交互"],
          summary="获取服务器时区 + WS 地址")
async def get_config(request):
    """获取设备ID"""
    device_id = request.headers.get("device-id", "")
    if not device_id:
        raise Exception("OTA请求设备ID为空")

    data_json = json.loads(request.text)
    server_config = data_json["server"]
    data_json["application"].get("version", "1.0.0"),

    # 实际处理流程：ASR -> LLM -> TTS
    synthesized_audio = b""
    return {
        "audio_base64": base64.b64encode(synthesized_audio).decode("utf-8"),
        "audio_format": "mp3",
        "session_id": request.session_id
    }

@app.get("/mcp/vision/explain/",
         tags=["固件交互"],
         summary="测试服务视觉分析接口是否正常")
async def do_vision_get():
    """获取当前系统配置"""
    return "MCP Vision 接口运行正常，视觉解释接口地址是：http://192.168.1.170:8003/mcp/vision/explain"

@app.post("/mcp/vision/explain/",
          tags=["固件交互"],
          summary="接收图片并调用视觉大模型做响应")
async def do_vision(request):
    # 验证token
    config = load_config()
    handle = VisionHandler(config)
    is_valid, token_device_id = handle._verify_auth_token(request)
    if not is_valid:
        return {}

    # 获取请求头信息
    device_id = request.headers.get("Device-Id", "")
    client_id = request.headers.get("Client-Id", "")
    if device_id != token_device_id:
        return {}

    # 解析multipart/form-data请求
    reader = await request.multipart()

    # 读取question字段
    question_field = await reader.next()
    if question_field is None:
        raise ValueError("缺少问题字段")
    question = await question_field.text()

    # 读取图片文件
    image_field = await reader.next()
    if image_field is None:
        raise ValueError("缺少图片文件")

    # 读取图片数据
    image_data = await image_field.read()
    if not image_data:
        raise ValueError("图片数据为空")

    # 检查文件大小
    if len(image_data) > MAX_FILE_SIZE:
        raise ValueError(
            f"图片大小超过限制，最大允许{MAX_FILE_SIZE / 1024 / 1024}MB"
        )

    # 检查文件格式
    if not is_valid_image_file(image_data):
        raise ValueError(
            "不支持的文件格式，请上传有效的图片文件（支持JPEG、PNG、GIF、BMP、TIFF、WEBP格式）"
        )

    # 实际处理流程：ASR -> LLM -> TTS
    synthesized_audio = b""
    return {
        "audio_base64": base64.b64encode(synthesized_audio).decode("utf-8"),
        "audio_format": "mp3",
        "session_id": request.session_id
    }

@app.websocket("/xiaozhi/v1/",)
async def ws_endpoint(websocket: WebSocket):
    """ws 连接"""
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        msg = json.loads(data)
        print(msg)

        """循环接收消息"""
        while True:
            data = await websocket.receive()
            print(data)
            # msg = json.loads(data)
    except Exception as e:
        print(f'ws 请求异常 {e} - {traceback.format_exc()}')


