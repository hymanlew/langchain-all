import io
import os
import ffmpeg
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
import gradio.processing_utils as processing_utils
from gradio_client import utils as client_utils

import traceback
import requests
import json
from argparse import ArgumentParser
from pathlib import Path
from websocket import WebSocket
'''
qwen-vl-utils 工具包：
支持图像/视频的 base64 编码、URL 解析及多帧视频处理。
推荐使用 decord 或 torchcodec 后端解码视频（比 torchvision 快 3 倍）。
'''
from qwen_omni_utils import process_mm_info


# 模拟配置数据存储
config_data = {
    "device_id": "device-01",
    "token_url": "http://127.0.0.1:8003",
    "ota_status": "未连接",
}

# 获取当前目录路径
current_dir = Path(__file__).parent.absolute()
html_file = current_dir / "test_page.html"
js_file = current_dir / "libopus.js"  # 依赖文件目录
file_url = f"file={html_file.as_posix()}"
js_url = f"file={js_file.as_posix()}"
gr.set_static_paths(paths=[current_dir.parent])

# 验证文件存在
if not html_file.exists():
    raise FileNotFoundError(f"HTML 文件不存在: {html_file}")
if not js_file.exists():
    raise FileNotFoundError(f"JS 文件不存在: {js_file}")

conversation_log = []  # 存储会话记录
request_log = []       # 存储请求响应日志
choices_list = []
token = ""
wsSocket = None


# ------------ 后端接口函数 ------------
def convert_webm_to_mp4(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='aac', ar='16000', audio_bitrate='192k')
            .run(quiet=True, overwrite_output=True)
        )
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print("An error occurred during conversion.")
        print(e.stderr.decode('utf-8'))

def connect_server(deviceid, clientid, activa):
    """连接服务器"""
    global token
    try:
        header = {"Content-Type": "application/json", "Device-Id": deviceid, "Client-Id": clientid, "Authorization": token}
        wsSocket = WebSocket()
        wsSocket.connect(config_data["ws_url"], headers=header, timeout=3)
        if wsSocket.connected:
            gr.Info("ws 成功连接")
            wsSocket.close()

        res = requests.get(config_data["token_url"] + "/system/token", headers=header)
        if res.status_code == 200:
            gr.Info("token: " + json.dumps(res.json()))
            token = "Bearer " + res.json().get('result')

        req_json = {"avatarId": "123-001", "avatarUserId": "123", "times": 10}
        if activa:
            resp = requests.put(
                config_data["token_url"] + "/mcp/device/activate", data=req_json, headers=header
            )
        else:
            resp = requests.delete(
                config_data["token_url"] + "/mcp/device/activate/delete", params=req_json, headers=header
            )

        return f"OTA已连接, WS已连接, token = {token}" if token else "未获取到 TOKEN"
    except Exception as e:
        return f"🔴 OTA连接失败, WS连接失败: \n{traceback.format_exc()}"

def send_media(deviceid, clientid, type, image, audio, text):
    """发送多媒体消息"""
    if not text:
        return "请填写验证信息"
    if not any([image, audio]):
        return "请至少上传一种媒体文件"

    try:
        global data
        global file
        header = {"Device-Id": deviceid, "Client-Id": clientid}
        if image:
            # "表单字段名": ("文件名", 文件对象, "MIME类型"-可选-尽量不选-让请求自动处理)
            data = {
                "question": json.dumps({"opera": "insert" if type == "send" else "get", "type": "image", "data": text})}
            file = {"file": (image, open(image, "rb"))}
        if audio:
            data = {
                "question": json.dumps({"opera": "insert" if type == "send" else "get", "type": "audio", "data": text})}
            file = {"file": (audio, open(audio, "rb"))}

        resp = requests.post(
            config_data["token_url"] + "/mcp/vision/explain", data=data, files=file, headers=header
        )
        return f"录入成功 ({resp.json()})" if resp.status_code == 200 else f"录入失败 ({resp.json()})"
    except Exception as e:
        return f"🔴 录入失败: \n{traceback.format_exc()}"

def send_message(deviceid, clientid, message, audio_path):
    """发送文本消息"""
    if not wsSocket:
        return "", "对话未连接"

    try:
        if message:
            wsSocket.send_text(message)
            conversation_log.append(message)
        if audio_path:
            with open(audio_path, "rb") as f:
                wsSocket.send_bytes(f.read())
            conversation_log.append("字节")

        data_received = wsSocket.recv()
        if isinstance(data_received, bytes):
            # 字节流转临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(data_received)  # audio_bytes 为音频字节流
                tmp_file.close()
                data_received = tmp_file.name


        # 获取 MIME 类型
        mime_type = client_utils.get_mimetype(data_received)
        conversation_log.append(data_received)
    except Exception as e:
        request_log.append(f"错误: {str(e)}")
        return None
    finally:
        if wsSocket: wsSocket.close()
    return "\n".join(conversation_log), "\n".join()

def format_history(history: list, system_prompt: str):
    messages = []
    messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    for item in history:
        if isinstance(item["content"], str):
            messages.append({"role": item['role'], "content": item['content']})
        elif item["role"] == "user" and (isinstance(item["content"], list) or
                                         isinstance(item["content"], tuple)):
            file_path = item["content"][0]

            # URL or path-like object（本地文件路径的对象）
            # 文件路径/字节流	用户上传的音频输入（麦克风或文件）
            mime_type = client_utils.get_mimetype(file_path)
            if mime_type.startswith("image"):
                messages.append({
                    "role":
                        item['role'],
                    "content": [{
                        "type": "image",
                        "image": file_path
                    }]
                })
            elif mime_type.startswith("video"):
                messages.append({
                    "role":
                        item['role'],
                    "content": [{
                        "type": "video",
                        "video": file_path
                    }]
                })
            elif mime_type.startswith("audio"):
                messages.append({
                    "role":
                        item['role'],
                    "content": [{
                        "type": "audio",
                        "audio": file_path,
                    }]
                })
    return messages

def clear_chat_history():
    return None, None, None, None, None

def media_predict(audio, video, history, system_prompt, voice_choice):
    # 生成器函数 (yield)，实现流式输出（Streaming Outputs）
    # 此处作用为 界面状态初始化。
    # - 输入部件先直接返回 none 触发界面更新，None 等价于 yield gr.update(value=None) 的默认行为（即组件清空）
    # - 按钮组件通过 gr.update() 动态修改组件属性（如可见性）
    # - 不需要处理的组件或内容，需要显示写上，例如 history 保持历史记录不变
    # 后续代码为 处理媒体流并分步 yield 结果
    yield (
        None,  # microphone, 重置麦克风输入（清空）
        None,  # webcam, 重置摄像头输入（清空）
        history,  # media_chatbot, 保持历史记录不变
        gr.update(visible=False),  # submit_btn, 隐藏提交按钮（防止重复提交）
        gr.update(visible=True),  # stop_btn, 显示停止按钮（允许用户中断任务）
    )

    # 转换格式后，引入新格式的文件的路径
    if video is not None:
        convert_webm_to_mp4(video, video.replace('.webm', '.mp4'))
        video = video.replace(".webm", ".mp4")
    files = [audio, video]

    for f in files:
        if f:
            history.append({"role": "user", "content": (f,)})

    formatted_history = format_history(history=history, system_prompt=system_prompt, )
    history.append({"role": "assistant", "content": ""})

    for chunk in predict(formatted_history, voice_choice):
        # 在生成文本或音频流时，可逐步返回音频片段，实现 边生成边播放
        if chunk["type"] == "text":
            history[-1]["content"] = chunk["data"]
            yield (
                None,  # microphone
                None,  # webcam
                history,  # media_chatbot
                gr.update(visible=False),  # submit_btn
                gr.update(visible=True),  # stop_btn
            )
        if chunk["type"] == "audio":
            history.append({
                "role": "assistant",
                "content": gr.Audio(chunk["data"])
            })

    # Final yield
    yield (
        None,  # microphone
        None,  # webcam
        history,  # media_chatbot
        gr.update(visible=True),  # submit_btn
        gr.update(visible=False),  # stop_btn
    )

def chat_predict(text, audio, image, video, history, system_prompt, voice_choice):
    # Process text input
    if text:
        history.append({"role": "user", "content": text})

    # Process audio input
    if audio:
        history.append({"role": "user", "content": (audio,)})

    # Process image input
    if image:
        history.append({"role": "user", "content": (image,)})

    # Process video input
    if video:
        history.append({"role": "user", "content": (video,)})

    formatted_history = format_history(history=history, system_prompt=system_prompt)

    # 生成器函数 (yield)，实现流式输出（Streaming Outputs）
    # 清空输入组件（文本/音频/图像/视频），保留历史记录。为后续生成回复腾出界面空间。
    yield None, None, None, None, history

    history.append({"role": "assistant", "content": ""})
    for chunk in predict(formatted_history, voice_choice):
        if chunk["type"] == "text":
            history[-1]["content"] = chunk["data"]
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(
            ), history
        if chunk["type"] == "audio":
            history.append({
                "role": "assistant",
                "content": gr.Audio(chunk["data"])
            })

    # gr.skip()：跳过组件的更新，避免无效渲染（如输入框无需刷新）。
    # 仅更新 history 以准备追加新对话。
    yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history


# ------------ Gradio 界面构建 ------------
with gr.Blocks(title="小智服务器控制面板", theme=gr.themes.Soft(), js=f"/gradio_api/{js_url}") as demo:
    # 标签页布局
    with gr.Tab("配置"):
        gr.Markdown("## 设备配置")
        with gr.Row():
            device_id = gr.Textbox(label="设备ID", value=config_data["device_id"], interactive=True)
            client_id = gr.Textbox(label="用户ID", value=config_data["client_id"], interactive=True)

        gr.Markdown("## 测试连接")
        with gr.Row():
            connect_btn = gr.Button("连接服务器", min_width=150, variant="primary")
        with gr.Row():
            ota_status = gr.Textbox(label="服务器状态", value=config_data["ota_status"], interactive=False)

    with gr.Tab("身份验证"):
        gr.Markdown("## 上传信息")
        with gr.Row():
            # 这个支持全部方式，上传，录制，拍照等等
            image_upload = gr.Image(label="上传图片", type="filepath")
            audio_upload = gr.Audio(label="上传音频", type="filepath")
        with gr.Row():
            media_text = gr.Textbox(label="附加文本", placeholder="输入描述信息，在进行验证时会返回此信息")
            with gr.Column():
                send_pict_btn = gr.Button("上传图片信息", variant="primary")
            with gr.Column():
                send_audio_btn = gr.Button("上传音频信息", variant="primary")
        media_output = gr.Textbox(label="操作结果", interactive=False)

    with gr.Tab("外链 聊天机器人"):
        gr.Markdown("## 对话交互")
        with gr.Row():
            text_input = gr.Textbox(label="输入消息", placeholder="输入要发送的消息...")
            with gr.Column(scale=2):
                text_send_btn = gr.Button("发送文本", variant="primary")
            with gr.Column(scale=2):
                voice_send_btn = gr.Button("录制语音", variant="secondary")

            voice_recorder = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="语音消息",
                interactive=True,
                visible=False
            )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="会话记录", type="messages", height=650)
            with gr.Column(scale=2):
                log_display = gr.Textbox(label="请求日志", lines=8, max_lines=12, interactive=False)

        gr.HTML(value=f"""<a href="/gradio_api/{file_url}">打开对话测试</a>""",
                min_height=10,
                container=True)
        # gr.HTML(value="""
        #        <iframe src="/gradio_api/{file_url}"
        #            width="100%"
        #            height="600px"
        #            frameborder="0">
        #        </iframe>
        #        """,
        #         min_height=800,
        #         container=True)

    with gr.Tab("cosyvoice 聊天机器人"):
        mode_checkbox_group = gr.Radio(choices=choices_list, label='单选框', value=choices_list[0], scale=0.3)
        sft_dropdown = gr.Dropdown(choices=choices_list, label='下拉框', value=choices_list[0], scale=0.3)
        speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1, scale=0.3)

        chatbot_cosy = gr.Chatbot(label="会话记录", type="messages", height=650)
        # Media upload section in one row，都只限制为文件上传
        with gr.Row(equal_height=True):
            audio_input = gr.Audio(sources=["upload"],
                                   type="filepath",
                                   label="Upload Audio",
                                   elem_classes="media-upload",
                                   scale=1)
            image_input = gr.Image(sources=["upload"],
                                   type="filepath",
                                   label="Upload Image",
                                   elem_classes="media-upload",
                                   scale=1)
            video_input = gr.Video(sources=["upload"],
                                   label="Upload Video",
                                   elem_classes="media-upload",
                                   scale=1)

        text_voi = gr.Textbox(show_label=False, placeholder="Enter text here...")
        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)
        # Control buttons
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary", size="lg")
            stop_btn = gr.Button("Stop", visible=False, size="lg")
            clear_btn = gr.Button("Clear History", size="lg")

    with gr.Tab("内嵌 聊天机器人"):
        # Add some custom CSS to improve the layout
        # 仅静态内容, 若要加载 JS 需要在 Blocks 初始化时通过 js 参数加载外部 JS 文件
        gr.HTML(
            value="""
            <!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>小智服务器测试页面</title>
    <style>
        body {
            font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 10px 20px 10px 20px;
        }

        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }

        .section {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            background-color: #f9f9f9;
        }

        .section h2 {
            margin-top: 0;
            color: #444;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 0;
        }

        .section h2 .toggle-button {
            margin-left: auto;
            padding: 4px 12px;
            font-size: 12px;
            border-radius: 4px;
            cursor: pointer;
            height: 28px;
            line-height: 20px;
        }

        .device-info {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-left: 20px;
            padding: 0 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
            height: 28px;
            line-height: 28px;
        }

        .device-info span {
            color: #666;
            font-size: 13px;
        }

        .device-info strong {
            color: #333;
            font-weight: 500;
        }

        .config-panel {
            display: none;
            transition: all 0.3s ease;
            margin-top: 5px;
            padding: 5px 0;
        }

        .config-panel.expanded {
            display: block;
        }

        .control-panel {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }

        button {
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            background-color: #4285f4;
            color: white;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        button:hover {
            background-color: #3367d6;
        }

        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }

        #serverUrl,
        #otaUrl {
            flex-grow: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        .message-input {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        #messageInput {
            flex-grow: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        #nfcCardId {
            flex-grow: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        .conversation {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
            flex: 1;
            margin-right: 10px;
        }

        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            max-width: 80%;
        }

        .user {
            background-color: #e2f2ff;
            margin-left: auto;
            margin-right: 10px;
            text-align: right;
        }

        .server {
            background-color: #f0f0f0;
            margin-right: auto;
            margin-left: 10px;
        }

        .status {
            color: #666;
            font-style: italic;
            font-size: 14px;
            margin: 0;
            padding: 0;
        }

        .audio-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }

        .audio-visualizer {
            height: 60px;
            width: 100%;
            margin-top: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fafafa;
        }

        .record-button {
            background-color: #db4437;
        }

        .record-button:hover {
            background-color: #c53929;
        }

        .record-button.recording {
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% {
                background-color: #db4437;
            }

            50% {
                background-color: #ff6659;
            }

            100% {
                background-color: #db4437;
            }
        }

        #logContainer {
            margin-top: 0;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
            font-family: monospace;
            height: 300px;
            overflow-y: auto;
            flex: 1;
            margin-left: 10px;
        }

        .log-entry {
            margin: 5px 0;
            font-size: 12px;
        }

        .log-info {
            color: #333;
        }

        .log-error {
            color: #db4437;
        }

        .log-success {
            color: #0f9d58;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translate(-50%, -60%);
            }

            to {
                opacity: 1;
                transform: translate(-50%, -50%);
            }
        }

        .script-status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }

        .script-loaded {
            background-color: #0f9d58;
        }

        .script-loading {
            background-color: #f4b400;
        }

        .script-error {
            background-color: #db4437;
        }

        .script-list {
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
            font-family: monospace;
            font-size: 11px;
        }

        #scriptStatus.success {
            background-color: #e6f4ea;
            color: #0f9d58;
            border-left: 4px solid #0f9d58;
        }

        #scriptStatus.error {
            background-color: #fce8e6;
            color: #db4437;
            border-left: 4px solid #db4437;
        }

        #scriptStatus.warning {
            background-color: #fef7e0;
            color: #f4b400;
            border-left: 4px solid #f4b400;
        }

        /* 标签页样式 */
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }

        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            color: #666;
            position: relative;
            transition: all 0.3s ease;
        }

        .tab:hover {
            color: #4285f4;
        }

        .tab.active {
            color: #4285f4;
            font-weight: bold;
        }

        .tab.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 100%;
            height: 2px;
            background-color: #4285f4;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .flex-container {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }

        .config-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            width: 100%;
        }

        .config-item label {
            width: 100px;
            text-align: right;
            margin-right: 10px;
            color: #666;
        }

        .config-item input {
            flex-grow: 1;
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        .control-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
        }

        .connection-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            width: 100%;
        }

        .connection-controls input {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            min-width: 200px;
        }

        .connection-controls button {
            white-space: nowrap;
            padding: 8px 15px;
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-left: 20px;
            padding: 0 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
            height: 28px;
            line-height: 28px;
        }

        .connection-status span {
            color: #666;
            font-size: 13px;
        }

        .connection-status .status {
            color: #333;
            font-weight: 500;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>小智服务器测试页面</h1>

        <div id="scriptStatus" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);">
            正在加载Opus库...</div>

        <!-- 添加配置面板 -->
        <div class="section">
            <h2>
                设备配置
                <span class="device-info">
                    <span>MAC: <strong id="displayMac"></strong></span>
                    <span>客户端: <strong id="displayClient">web_test_client</strong></span>
                </span>
                <button class="toggle-button" id="toggleConfig">编辑</button>
            </h2>
            <div class="config-panel" id="configPanel">
                <div class="control-panel">
                    <div class="config-item">
                        <label for="deviceMac">设备MAC:</label>
                        <input type="text" id="deviceMac" placeholder="设备MAC地址">
                    </div>
                    <div class="config-item">
                        <label for="deviceName">设备名称:</label>
                        <input type="text" id="deviceName" value="Web测试设备" placeholder="设备名称">
                    </div>
                    <div class="config-item">
                        <label for="clientId">客户端ID:</label>
                        <input type="text" id="clientId" value="web_test_client" placeholder="客户端ID">
                    </div>
                    <div class="config-item">
                        <label for="token">认证Token:</label>
                        <input type="text" id="token" value="your-token1" placeholder="认证Token">
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>
                连接信息
                <span class="connection-status">
                    <span>OTA: <span id="otaStatus" class="status">ota未连接</span></span>
                    <span>WS: <span id="connectionStatus" class="status">ws未连接</span></span>
                </span>
            </h2>
            <div class="connection-controls">
                <input type="text" id="otaUrl" value="http://127.0.0.1:8002/xiaozhi/ota/"
                    placeholder="OTA服务器地址，如：http://127.0.0.1:8002/xiaozhi/ota/" />
                <input type="text" id="serverUrl" value="ws://127.0.0.1:8000/xiaozhi/v1/"
                    placeholder="WebSocket服务器地址，如：ws://127.0.0.1:8000/xiaozhi/v1/" />
                <button id="connectButton">连接</button>
                <button id="authTestButton">测试认证</button>
            </div>
        </div>

        <div class="section">
            <div class="tabs">
                <button class="tab active" data-tab="text">文本消息</button>
                <button class="tab" data-tab="voice">语音消息</button>
            </div>

            <div class="tab-content active" id="textTab">
                <div class="message-input">
                    <input type="text" id="messageInput" placeholder="输入消息..." disabled>
                    <button id="sendTextButton" disabled>发送</button>
                </div>
            </div>

            <div class="tab-content" id="voiceTab">
                <div class="audio-controls">
                    <button id="recordButton" class="record-button" disabled>开始录音</button>
                </div>
                <canvas id="audioVisualizer" class="audio-visualizer"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>会话记录</h2>
            <div class="flex-container">
                <div id="conversation" class="conversation"></div>
                <div id="logContainer">
                    <div class="log-entry log-info">准备就绪，请连接服务器开始测试...</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Opus解码库，设置静态目录（跳过缓存）后，可使用相对路径 -->
    <script src="./libopus.js"></script>

    <script>
        // 需要加载的脚本列表 - 移除Opus依赖
        const scriptFiles = [];

        // 脚本加载状态
        const scriptStatus = {
            loading: 0,
            loaded: 0,
            failed: 0,
            total: scriptFiles.length
        };

        // 检查Opus库是否已加载
        function checkOpusLoaded() {
            try {
                // 检查Module是否存在（本地库导出的全局变量）
                if (typeof Module === 'undefined') {
                    throw new Error('Opus库未加载，Module对象不存在');
                }

                // 尝试先使用Module.instance（libopus.js最后一行导出方式）
                if (typeof Module.instance !== 'undefined' && typeof Module.instance._opus_decoder_get_size === 'function') {
                    // 使用Module.instance对象替换全局Module对象
                    window.ModuleInstance = Module.instance;
                    log('Opus库加载成功（使用Module.instance）', 'success');
                    updateScriptStatus('Opus库加载成功', 'success');

                    // 3秒后隐藏状态
                    const statusElement = document.getElementById('scriptStatus');
                    if (statusElement) statusElement.style.display = 'none';
                    return;
                }

                // 如果没有Module.instance，检查全局Module函数
                if (typeof Module._opus_decoder_get_size === 'function') {
                    window.ModuleInstance = Module;
                    log('Opus库加载成功（使用全局Module）', 'success');
                    updateScriptStatus('Opus库加载成功', 'success');

                    // 3秒后隐藏状态
                    const statusElement = document.getElementById('scriptStatus');
                    if (statusElement) statusElement.style.display = 'none';
                    return;
                }

                throw new Error('Opus解码函数未找到，可能Module结构不正确');
            } catch (err) {
                log(`Opus库加载失败，请检查libopus.js文件是否存在且正确: ${err.message}`, 'error');
                updateScriptStatus('Opus库加载失败，请检查libopus.js文件是否存在且正确', 'error');
            }
        }

        // 更新脚本状态显示
        function updateScriptStatus(message, type) {
            const statusElement = document.getElementById('scriptStatus');
            if (statusElement) {
                statusElement.textContent = message;
                statusElement.className = `script-status ${type}`;
                statusElement.style.display = 'block';
                statusElement.style.width = 'auto';
            }
        }

        // 全局变量
        let websocket = null;
        let mediaRecorder = null;
        let audioContext = null;
        let analyser = null;
        let audioChunks = [];
        let isRecording = false;
        let visualizerCanvas = document.getElementById('audioVisualizer');
        let visualizerContext = visualizerCanvas.getContext('2d');
        let audioQueue = [];
        let isPlaying = false;
        let opusDecoder = null; // Opus解码器
        let visualizationRequest = null; // 动画帧请求ID

        // 音频流缓冲相关
        let audioBuffers = []; // 用于存储接收到的所有音频数据
        let totalAudioSize = 0; // 跟踪累积的音频大小

        let audioBufferQueue = [];     // 存储接收到的音频包
        let isAudioBuffering = false;  // 是否正在缓冲音频
        let isAudioPlaying = false;    // 是否正在播放音频
        const BUFFER_THRESHOLD = 3;    // 缓冲包数量阈值，至少累积3个包再开始播放
        const MIN_AUDIO_DURATION = 0.1; // 最小音频长度(秒)，小于这个长度的音频会被合并
        let streamingContext = null;   // 音频流上下文
        const SAMPLE_RATE = 16000;     // 采样率
        const CHANNELS = 1;            // 声道数
        const FRAME_SIZE = 960;        // 帧大小

        // DOM元素
        const connectButton = document.getElementById('connectButton');
        const serverUrlInput = document.getElementById('serverUrl');
        const connectionStatus = document.getElementById('connectionStatus');
        const messageInput = document.getElementById('messageInput');
        const sendTextButton = document.getElementById('sendTextButton');
        const recordButton = document.getElementById('recordButton');
        const stopButton = document.getElementById('stopButton');
        const conversationDiv = document.getElementById('conversation');
        const logContainer = document.getElementById('logContainer');

        // 日志函数
        function log(message, type = 'info') {
            // 将消息按换行符分割成多行
            const lines = message.split('\n');
            const now = new Date();
            // const timestamp = `[${now.toLocaleTimeString()}] `;
            const timestamp = `[${now.toLocaleTimeString()}.${now.getMilliseconds().toString().padStart(3, '0')}] `;
            // 为每一行创建日志条目
            lines.forEach((line, index) => {
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${type}`;
                // 如果是第一条日志，显示时间戳
                const prefix = index === 0 ? timestamp : ' '.repeat(timestamp.length);
                logEntry.textContent = `${prefix}${line}`;
                // logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                // logEntry.style 保留起始的空格
                logEntry.style.whiteSpace = 'pre';
                if (type === 'error') {
                    logEntry.style.color = 'red';
                } else if (type === 'debug') {
                    logEntry.style.color = 'gray';
                    return;
                } else if (type === 'warning') {
                    logEntry.style.color = 'orange';
                } else if (type === 'success') {
                    logEntry.style.color = 'green';
                } else {
                    logEntry.style.color = 'black';
                }
                logContainer.appendChild(logEntry);
            });

            logContainer.scrollTop = logContainer.scrollHeight;
        }

        // 初始化可视化器
        function initVisualizer() {
            visualizerCanvas.width = visualizerCanvas.clientWidth;
            visualizerCanvas.height = visualizerCanvas.clientHeight;
            visualizerContext.fillStyle = '#fafafa';
            visualizerContext.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
        }

        // 绘制音频可视化效果
        function drawVisualizer(dataArray) {
            visualizationRequest = requestAnimationFrame(() => drawVisualizer(dataArray));

            if (!isRecording) return;

            analyser.getByteFrequencyData(dataArray);

            visualizerContext.fillStyle = '#fafafa';
            visualizerContext.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);

            const barWidth = (visualizerCanvas.width / dataArray.length) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
                barHeight = dataArray[i] / 2;

                visualizerContext.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
                visualizerContext.fillRect(x, visualizerCanvas.height - barHeight, barWidth, barHeight);

                x += barWidth + 1;
            }
        }

        // 添加消息到会话记录
        function addMessage(text, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'server'}`;
            messageDiv.textContent = text;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }


        // 开始音频缓冲过程
        function startAudioBuffering() {
            if (isAudioBuffering || isAudioPlaying) return;

            isAudioBuffering = true;
            log("开始音频缓冲...", 'info');

            // 先尝试初始化解码器，以便在播放时已准备好
            initOpusDecoder().catch(error => {
                log(`预初始化Opus解码器失败: ${error.message}`, 'warning');
                // 继续缓冲，我们会在播放时再次尝试初始化
            });

            // 设置超时，如果在一定时间内没有收集到足够的音频包，就开始播放
            setTimeout(() => {
                if (isAudioBuffering && audioBufferQueue.length > 0) {
                    log(`缓冲超时，当前缓冲包数: ${audioBufferQueue.length}，开始播放`, 'info');
                    playBufferedAudio();
                }
            }, 300); // 300ms超时

            // 监控缓冲进度
            const bufferCheckInterval = setInterval(() => {
                if (!isAudioBuffering) {
                    clearInterval(bufferCheckInterval);
                    return;
                }

                // 当累积了足够的音频包，开始播放
                if (audioBufferQueue.length >= BUFFER_THRESHOLD) {
                    clearInterval(bufferCheckInterval);
                    log(`已缓冲 ${audioBufferQueue.length} 个音频包，开始播放`, 'info');
                    playBufferedAudio();
                }
            }, 50);
        }

        // 播放已缓冲的音频
        function playBufferedAudio() {
            if (isAudioPlaying || audioBufferQueue.length === 0) return;

            isAudioPlaying = true;
            isAudioBuffering = false;

            // 确保Opus解码器已初始化
            const initDecoderAndPlay = async () => {
                try {
                    // 确保音频上下文存在
                    if (!audioContext) {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)({
                            sampleRate: SAMPLE_RATE
                        });
                        log('创建音频上下文，采样率: ' + SAMPLE_RATE + 'Hz', 'debug');
                    }

                    // 确保解码器已初始化
                    if (!opusDecoder) {
                        log('初始化Opus解码器...', 'info');
                        try {
                            opusDecoder = await initOpusDecoder();
                            if (!opusDecoder) {
                                throw new Error('解码器初始化失败');
                            }
                            log('Opus解码器初始化成功', 'success');
                        } catch (error) {
                            log('Opus解码器初始化失败: ' + error.message, 'error');
                            isAudioPlaying = false;
                            return;
                        }
                    }

                    // 创建流式播放上下文
                    if (!streamingContext) {
                        streamingContext = {
                            queue: [],          // 已解码的PCM队列
                            playing: false,     // 是否正在播放
                            endOfStream: false, // 是否收到结束信号
                            source: null,       // 当前音频源
                            totalSamples: 0,    // 累积的总样本数
                            lastPlayTime: 0,    // 上次播放的时间戳

                            // 将Opus数据解码为PCM
                            decodeOpusFrames: async function (opusFrames) {
                                if (!opusDecoder) {
                                    log('Opus解码器未初始化，无法解码', 'error');
                                    return;
                                }

                                let decodedSamples = [];

                                for (const frame of opusFrames) {
                                    try {
                                        // 使用Opus解码器解码
                                        const frameData = opusDecoder.decode(frame);
                                        if (frameData && frameData.length > 0) {
                                            // 转换为Float32
                                            const floatData = convertInt16ToFloat32(frameData);
                                            // 使用循环替代展开运算符
                                            for (let i = 0; i < floatData.length; i++) {
                                                decodedSamples.push(floatData[i]);
                                            }
                                        }
                                    } catch (error) {
                                        log("Opus解码失败: " + error.message, 'error');
                                    }
                                }

                                if (decodedSamples.length > 0) {
                                    // 使用循环替代展开运算符
                                    for (let i = 0; i < decodedSamples.length; i++) {
                                        this.queue.push(decodedSamples[i]);
                                    }
                                    this.totalSamples += decodedSamples.length;

                                    // 如果累积了至少0.2秒的音频，开始播放
                                    const minSamples = SAMPLE_RATE * MIN_AUDIO_DURATION;
                                    if (!this.playing && this.queue.length >= minSamples) {
                                        this.startPlaying();
                                    }
                                } else {
                                    log('没有成功解码的样本', 'warning');
                                }
                            },

                            // 开始播放音频
                            startPlaying: function () {
                                if (this.playing || this.queue.length === 0) return;

                                this.playing = true;

                                // 创建新的音频缓冲区
                                const minPlaySamples = Math.min(this.queue.length, SAMPLE_RATE); // 最多播放1秒
                                const currentSamples = this.queue.splice(0, minPlaySamples);

                                const audioBuffer = audioContext.createBuffer(CHANNELS, currentSamples.length, SAMPLE_RATE);
                                audioBuffer.copyToChannel(new Float32Array(currentSamples), 0);

                                // 创建音频源
                                this.source = audioContext.createBufferSource();
                                this.source.buffer = audioBuffer;

                                // 创建增益节点用于平滑过渡
                                const gainNode = audioContext.createGain();

                                // 应用淡入淡出效果避免爆音
                                const fadeDuration = 0.02; // 20毫秒
                                gainNode.gain.setValueAtTime(0, audioContext.currentTime);
                                gainNode.gain.linearRampToValueAtTime(1, audioContext.currentTime + fadeDuration);

                                const duration = audioBuffer.duration;
                                if (duration > fadeDuration * 2) {
                                    gainNode.gain.setValueAtTime(1, audioContext.currentTime + duration - fadeDuration);
                                    gainNode.gain.linearRampToValueAtTime(0, audioContext.currentTime + duration);
                                }

                                // 连接节点并开始播放
                                this.source.connect(gainNode);
                                gainNode.connect(audioContext.destination);

                                this.lastPlayTime = audioContext.currentTime;
                                log(`开始播放 ${currentSamples.length} 个样本，约 ${(currentSamples.length / SAMPLE_RATE).toFixed(2)} 秒`, 'info');

                                // 播放结束后的处理
                                this.source.onended = () => {
                                    this.source = null;
                                    this.playing = false;

                                    // 使用setTimeout避免递归调用
                                    setTimeout(() => {
                                        // 如果队列中还有数据，继续播放
                                        if (this.queue.length > 0) {
                                            this.startPlaying();
                                        } else if (audioBufferQueue.length > 0) {
                                            // 缓冲区有新数据，进行解码
                                            const frames = [...audioBufferQueue];
                                            audioBufferQueue = [];
                                            this.decodeOpusFrames(frames);
                                        } else if (this.endOfStream) {
                                            // 流已结束且没有更多数据
                                            log("音频播放完成", 'info');
                                            isAudioPlaying = false;
                                            this.endOfStream = false;
                                            streamingContext = null;
                                        } else {
                                            // 等待更多数据
                                            setTimeout(() => {
                                                // 如果仍然没有新数据，但有更多的包到达
                                                if (this.queue.length === 0 && audioBufferQueue.length > 0) {
                                                    const frames = [...audioBufferQueue];
                                                    audioBufferQueue = [];
                                                    this.decodeOpusFrames(frames);
                                                } else if (this.queue.length === 0 && audioBufferQueue.length === 0) {
                                                    // 真的没有更多数据了
                                                    log("音频播放完成 (超时)", 'info');
                                                    isAudioPlaying = false;
                                                    streamingContext = null;
                                                }
                                            }, 500); // 500ms超时
                                        }
                                    }, 10); // 10ms延迟，避免立即递归
                                };

                                this.source.start();
                            }
                        };
                    }

                    // 开始处理缓冲的数据
                    const frames = [...audioBufferQueue];
                    audioBufferQueue = []; // 清空缓冲队列

                    // 解码并播放
                    await streamingContext.decodeOpusFrames(frames);

                } catch (error) {
                    log(`播放已缓冲的音频出错: ${error.message}`, 'error');
                    isAudioPlaying = false;
                    streamingContext = null;
                }
            };

            // 执行初始化和播放
            initDecoderAndPlay();
        }

        // 将Int16音频数据转换为Float32音频数据
        function convertInt16ToFloat32(int16Data) {
            const float32Data = new Float32Array(int16Data.length);
            for (let i = 0; i < int16Data.length; i++) {
                // 将[-32768,32767]范围转换为[-1,1]
                float32Data[i] = int16Data[i] / (int16Data[i] < 0 ? 0x8000 : 0x7FFF);
            }
            return float32Data;
        }

        // 初始化Opus解码器 - 确保完全初始化完成后才返回
        async function initOpusDecoder() {
            if (opusDecoder) return opusDecoder; // 已经初始化

            try {
                // 检查ModuleInstance是否存在
                if (typeof window.ModuleInstance === 'undefined') {
                    if (typeof Module !== 'undefined') {
                        // 使用全局Module作为ModuleInstance
                        window.ModuleInstance = Module;
                        log('使用全局Module作为ModuleInstance', 'info');
                    } else {
                        throw new Error('Opus库未加载，ModuleInstance和Module对象都不存在');
                    }
                }

                const mod = window.ModuleInstance;

                // 创建解码器对象
                opusDecoder = {
                    channels: CHANNELS,
                    rate: SAMPLE_RATE,
                    frameSize: FRAME_SIZE,
                    module: mod,
                    decoderPtr: null, // 初始为null

                    // 初始化解码器
                    init: function () {
                        if (this.decoderPtr) return true; // 已经初始化

                        // 获取解码器大小
                        const decoderSize = mod._opus_decoder_get_size(this.channels);
                        log(`Opus解码器大小: ${decoderSize}字节`, 'debug');

                        // 分配内存
                        this.decoderPtr = mod._malloc(decoderSize);
                        if (!this.decoderPtr) {
                            throw new Error("无法分配解码器内存");
                        }

                        // 初始化解码器
                        const err = mod._opus_decoder_init(
                            this.decoderPtr,
                            this.rate,
                            this.channels
                        );

                        if (err < 0) {
                            this.destroy(); // 清理资源
                            throw new Error(`Opus解码器初始化失败: ${err}`);
                        }

                        log("Opus解码器初始化成功", 'success');
                        return true;
                    },

                    // 解码方法
                    decode: function (opusData) {
                        if (!this.decoderPtr) {
                            if (!this.init()) {
                                throw new Error("解码器未初始化且无法初始化");
                            }
                        }

                        try {
                            const mod = this.module;

                            // 为Opus数据分配内存
                            const opusPtr = mod._malloc(opusData.length);
                            mod.HEAPU8.set(opusData, opusPtr);

                            // 为PCM输出分配内存
                            const pcmPtr = mod._malloc(this.frameSize * 2); // Int16 = 2字节

                            // 解码
                            const decodedSamples = mod._opus_decode(
                                this.decoderPtr,
                                opusPtr,
                                opusData.length,
                                pcmPtr,
                                this.frameSize,
                                0 // 不使用FEC
                            );

                            if (decodedSamples < 0) {
                                mod._free(opusPtr);
                                mod._free(pcmPtr);
                                throw new Error(`Opus解码失败: ${decodedSamples}`);
                            }

                            // 复制解码后的数据
                            const decodedData = new Int16Array(decodedSamples);
                            for (let i = 0; i < decodedSamples; i++) {
                                decodedData[i] = mod.HEAP16[(pcmPtr >> 1) + i];
                            }

                            // 释放内存
                            mod._free(opusPtr);
                            mod._free(pcmPtr);

                            return decodedData;
                        } catch (error) {
                            log(`Opus解码错误: ${error.message}`, 'error');
                            return new Int16Array(0);
                        }
                    },

                    // 销毁方法
                    destroy: function () {
                        if (this.decoderPtr) {
                            this.module._free(this.decoderPtr);
                            this.decoderPtr = null;
                        }
                    }
                };

                // 初始化解码器
                if (!opusDecoder.init()) {
                    throw new Error("Opus解码器初始化失败");
                }

                return opusDecoder;

            } catch (error) {
                log(`Opus解码器初始化失败: ${error.message}`, 'error');
                opusDecoder = null; // 重置为null，以便下次重试
                throw error;
            }
        }

        // 初始化音频录制和处理
        async function initAudio() {
            try {
                // 请求麦克风权限
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000,  // 确保16kHz采样率
                        channelCount: 1     // 确保单声道
                    }
                });
                log('已获取麦克风访问权限', 'success');

                // 创建音频上下文
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000,  // 确保采样率与服务器期望的一致
                    latencyHint: 'interactive'
                });
                const source = audioContext.createMediaStreamSource(stream);

                // 获取实际音频轨道设置
                const audioTracks = stream.getAudioTracks();
                if (audioTracks.length > 0) {
                    const track = audioTracks[0];
                    const settings = track.getSettings();
                    log(`实际麦克风设置 - 采样率: ${settings.sampleRate || '未知'}Hz, 声道数: ${settings.channelCount || '未知'}`, 'info');
                }

                // 创建分析器用于可视化
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 2048;
                source.connect(analyser);

                // 尝试初始化MediaRecorder，按优先级尝试不同编码选项
                try {
                    // 优先尝试使用Opus编码
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=opus',
                        audioBitsPerSecond: 16000
                    });
                    log('已初始化MediaRecorder (使用Opus编码)', 'success');
                    log(`选择的编码格式: ${mediaRecorder.mimeType}`, 'info');
                } catch (e1) {
                    try {
                        // 如果Opus不支持，尝试MP3
                        mediaRecorder = new MediaRecorder(stream, {
                            mimeType: 'audio/webm',
                            audioBitsPerSecond: 16000
                        });
                        log('已初始化MediaRecorder (使用WebM标准编码，Opus不支持)', 'warning');
                        log(`选择的编码格式: ${mediaRecorder.mimeType}`, 'info');
                    } catch (e2) {
                        try {
                            // 尝试其他备选格式
                            mediaRecorder = new MediaRecorder(stream, {
                                mimeType: 'audio/ogg;codecs=opus',
                                audioBitsPerSecond: 16000
                            });
                            log('已初始化MediaRecorder (使用OGG+Opus编码)', 'warning');
                            log(`选择的编码格式: ${mediaRecorder.mimeType}`, 'info');
                        } catch (e3) {
                            // 最后使用默认编码
                            mediaRecorder = new MediaRecorder(stream);
                            log(`已初始化MediaRecorder (使用默认编码: ${mediaRecorder.mimeType})`, 'warning');
                        }
                    }
                }

                // 处理录制的数据
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                // 录制结束后处理数据
                mediaRecorder.onstop = async () => {
                    // 停止可视化
                    if (visualizationRequest) {
                        cancelAnimationFrame(visualizationRequest);
                        visualizationRequest = null;
                    }

                    log(`录音结束，已收集的音频块数量: ${audioChunks.length}`, 'info');
                    if (audioChunks.length === 0) {
                        log('警告：没有收集到任何音频数据，请检查麦克风是否工作正常', 'error');
                        return;
                    }

                    // 创建完整的录音blob
                    const blob = new Blob(audioChunks, { type: audioChunks[0].type });
                    log(`已创建音频Blob，MIME类型: ${audioChunks[0].type}，大小: ${(blob.size / 1024).toFixed(2)} KB`, 'info');

                    // 保存原始块，以防清空后需要调试
                    const chunks = [...audioChunks];
                    audioChunks = [];

                    try {
                        // 将blob转换为ArrayBuffer
                        const arrayBuffer = await blob.arrayBuffer();
                        const uint8Array = new Uint8Array(arrayBuffer);

                        log(`已转换为Uint8Array，准备发送，大小: ${(arrayBuffer.byteLength / 1024).toFixed(2)} KB`, 'info');

                        // 检查WebSocket状态
                        if (!websocket) {
                            log('错误：WebSocket连接不存在', 'error');
                            return;
                        }

                        if (websocket.readyState !== WebSocket.OPEN) {
                            log(`错误：WebSocket连接未打开，当前状态: ${websocket.readyState}`, 'error');
                            return;
                        }

                        // 直接发送二进制音频数据 - 这是最简单有效的方式
                        try {
                            // 注意：开始和结束消息已在录音开始和结束时发送
                            // 这里只需要发送音频数据
                            await new Promise(resolve => setTimeout(resolve, 50));

                            // 处理WebM容器格式，提取纯Opus数据
                            // 服务器使用opuslib_next.Decoder，需要纯Opus帧
                            log('正在处理音频数据，提取纯Opus帧...', 'info');
                            const opusData = extractOpusFrames(uint8Array);

                            // 记录Opus数据大小
                            log(`已提取Opus数据，大小: ${(opusData.byteLength / 1024).toFixed(2)} KB`, 'info');

                            // 发送音频消息第二步：二进制音频数据
                            websocket.send(opusData);
                            log(`已发送Opus音频数据: ${(opusData.byteLength / 1024).toFixed(2)} KB`, 'success');
                        } catch (error) {
                            log(`音频数据发送失败: ${error.message}`, 'error');

                            // 尝试使用base64编码作为备选方案
                            try {
                                log('尝试使用base64编码方式发送...', 'info');
                                const base64Data = arrayBufferToBase64(arrayBuffer);
                                const audioDataMessage = {
                                    type: 'audio',
                                    action: 'data',
                                    format: 'opus',
                                    sample_rate: 16000,
                                    channels: 1,
                                    mime_type: chunks[0].type,
                                    encoding: 'base64',
                                    data: base64Data
                                };
                                websocket.send(JSON.stringify(audioDataMessage));
                                log(`已使用base64编码发送音频数据: ${(arrayBuffer.byteLength / 1024).toFixed(2)} KB`, 'warning');
                            } catch (base64Error) {
                                log(`所有数据发送方式均失败: ${base64Error.message}`, 'error');
                            }
                        }
                    } catch (error) {
                        log(`处理录音数据错误: ${error.message}`, 'error');
                    }
                };

                // 尝试初始化Opus解码器
                try {
                    // 检查ModuleInstance是否存在（本地库导出的全局变量）
                    if (typeof window.ModuleInstance === 'undefined') {
                        throw new Error('Opus库未加载，ModuleInstance对象不存在');
                    }

                    // 简单测试ModuleInstance是否可用
                    if (typeof window.ModuleInstance._opus_decoder_get_size === 'function') {
                        const testSize = window.ModuleInstance._opus_decoder_get_size(1);
                        log(`Opus解码器测试成功，解码器大小: ${testSize} 字节`, 'success');
                    } else {
                        throw new Error('Opus解码函数未找到');
                    }
                } catch (err) {
                    log(`Opus解码器初始化警告: ${err.message}，将在需要时重试`, 'warning');
                }

                log('音频系统初始化完成', 'success');
                return true;
            } catch (error) {
                log(`音频初始化错误: ${error.message}`, 'error');
                return false;
            }
        }

        // 开始录音
        function startRecording() {
            if (isRecording) return;

            try {
                // 最小录音时长提示
                log('请至少录制1-2秒钟的音频，确保采集到足够数据', 'info');

                // 获取服务器类型 - 从URL判断
                const serverUrl = serverUrlInput.value.trim();
                let isXiaozhiNative = false;

                // 检查是否是小智原生服务器 (根据URL特征判断)
                if (serverUrl.includes('xiaozhi') || serverUrl.includes('localhost') || serverUrl.includes('127.0.0.1')) {
                    isXiaozhiNative = true;
                    log('检测到小智原生服务器，使用标准listen协议', 'info');
                }

                // 使用直接PCM录音和libopus编码的方式
                startDirectRecording();
            } catch (error) {
                log(`录音启动错误: ${error.message}`, 'error');
            }
        }

        // 停止录音
        function stopRecording() {
            if (!isRecording) return;

            try {
                // 使用直接PCM录音停止
                stopDirectRecording();
            } catch (error) {
                log(`停止录音错误: ${error.message}`, 'error');
            }
        }

        // 连接WebSocket服务器
        async function connectToServer() {
            const url = serverUrlInput.value.trim();
            if (url === '') return;

            try {
                // 获取并验证配置
                const config = getConfig();
                if (!validateConfig(config)) {
                    return;
                }

                // 检查URL格式
                if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
                    log('URL格式错误，必须以ws://或wss://开头', 'error');
                    return;
                }

                // 先检查OTA状态
                log('正在检查OTA状态...', 'info');
                const otaUrl = document.getElementById('otaUrl').value.trim();
                localStorage.setItem('otaUrl', otaUrl);
                localStorage.setItem('wsUrl', url);
                try {
                    const otaResponse = await fetch(otaUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Device-Id': config.deviceId,
                            'Client-Id': config.clientId
                        },
                        body: JSON.stringify({
                            "version": 0,
                            "uuid": "",
                            "application": {
                                "name": "xiaozhi-web-test",
                                "version": "1.0.0",
                                "compile_time": "2025-04-16 10:00:00",
                                "idf_version": "4.4.3",
                                "elf_sha256": "1234567890abcdef1234567890abcdef1234567890abcdef"
                            },
                            "ota": {
                                "label": "xiaozhi-web-test",
                            },
                            "board": {
                                "type": "xiaozhi-web-test",
                                "ssid": "xiaozhi-web-test",
                                "rssi": 0,
                                "channel": 0,
                                "ip": "192.168.1.1",
                                "mac": config.deviceMac
                            },
                            "flash_size": 0,
                            "minimum_free_heap_size": 0,
                            "mac_address": config.deviceMac,
                            "chip_model_name": "",
                            "chip_info": {
                                "model": 0,
                                "cores": 0,
                                "revision": 0,
                                "features": 0
                            },
                            "partition_table": [
                                {
                                    "label": "",
                                    "type": 0,
                                    "subtype": 0,
                                    "address": 0,
                                    "size": 0
                                }
                            ]
                        })
                    });

                    if (!otaResponse.ok) {
                        throw new Error(`OTA检查失败: ${otaResponse.status} ${otaResponse.statusText}`);
                    }

                    const otaResult = await otaResponse.json();
                    log(`OTA检查结果: ${JSON.stringify(otaResult)}`, 'info');

                    log('OTA检查通过，开始连接WebSocket...', 'success');
                    document.getElementById('otaStatus').textContent = 'ota已连接';
                    document.getElementById('otaStatus').style.color = 'green';
                } catch (error) {
                    log(`OTA检查错误: ${error.message}`, 'error');
                    document.getElementById('otaStatus').textContent = 'ota未连接';
                    document.getElementById('otaStatus').style.color = 'red';
                }

                // 使用自定义WebSocket实现以添加认证头信息
                let connUrl = new URL(url);

                // 添加认证参数
                connUrl.searchParams.append('device-id', config.deviceId);
                connUrl.searchParams.append('client-id', config.clientId);

                log(`正在连接: ${connUrl.toString()}`, 'info');
                websocket = new WebSocket(connUrl.toString());

                // 设置接收二进制数据的类型为ArrayBuffer
                websocket.binaryType = 'arraybuffer';

                websocket.onopen = async () => {
                    log(`已连接到服务器: ${url}`, 'success');
                    connectionStatus.textContent = 'ws已连接';
                    connectionStatus.style.color = 'green';

                    // 连接成功后发送hello消息
                    await sendHelloMessage();

                    connectButton.textContent = '断开';
                    connectButton.removeEventListener('click', connectToServer);
                    connectButton.addEventListener('click', disconnectFromServer);
                    // connectButton.onclick = disconnectFromServer;
                    messageInput.disabled = false;
                    sendTextButton.disabled = false;

                    const audioInitialized = await initAudio();
                    if (audioInitialized) {
                        recordButton.disabled = false;
                    }
                };

                websocket.onclose = () => {
                    log('已断开连接', 'info');
                    connectionStatus.textContent = 'ws已断开';
                    connectionStatus.style.color = 'red';

                    connectButton.textContent = '连接';
                    connectButton.removeEventListener('click', disconnectFromServer);
                    connectButton.addEventListener('click', connectToServer);
                    // connectButton.onclick = connectToServer;
                    messageInput.disabled = true;
                    sendTextButton.disabled = true;
                    recordButton.disabled = true;
                    stopButton.disabled = true;
                };

                websocket.onerror = (error) => {
                    log(`WebSocket错误: ${error.message || '未知错误'}`, 'error');
                    connectionStatus.textContent = 'ws未连接';
                    connectionStatus.style.color = 'red';
                };

                websocket.onmessage = function (event) {
                    try {
                        // 检查是否为文本消息
                        if (typeof event.data === 'string') {
                            const message = JSON.parse(event.data);

                            if (message.type === 'hello') {
                                log(`服务器回应：${JSON.stringify(message, null, 2)}`, 'success');
                            } else if (message.type === 'tts') {
                                // TTS状态消息
                                if (message.state === 'start') {
                                    log('服务器开始发送语音', 'info');
                                } else if (message.state === 'sentence_start') {
                                    log(`服务器发送语音段: ${message.text}`, 'info');
                                    // 添加文本到会话记录
                                    if (message.text) {
                                        addMessage(message.text);
                                    }
                                } else if (message.state === 'sentence_end') {
                                    log(`语音段结束: ${message.text}`, 'info');
                                } else if (message.state === 'stop') {
                                    log('服务器语音传输结束', 'info');
                                    // 结束后更新UI状态
                                    if (recordButton.disabled) {
                                        recordButton.disabled = false;
                                        recordButton.textContent = '开始录音';
                                        recordButton.classList.remove('recording');
                                    }
                                }
                            } else if (message.type === 'audio') {
                                // 音频控制消息
                                log(`收到音频控制消息: ${JSON.stringify(message)}`, 'info');
                            } else if (message.type === 'stt') {
                                // 语音识别结果
                                log(`识别结果: ${message.text}`, 'info');
                                // 添加识别结果到会话记录
                                addMessage(`[语音识别] ${message.text}`, true);
                            } else if (message.type === 'llm') {
                                // 大模型回复
                                log(`大模型回复: ${message.text}`, 'info');
                                // 添加大模型回复到会话记录
                                if (message.text && message.text !== '😊') {
                                    addMessage(message.text);
                                }
                            }else if (message.type === 'mcp') {
                                const payload = message.payload || {};
                                log(`服务器下发: ${JSON.stringify(message)}`, 'info');
                                if (payload) {
                                    // 模拟小智客户端行为
                                    if(payload.method === 'tools/list'){
                                        const replay_message = JSON.stringify({"session_id":"","type":"mcp","payload":{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"self.get_device_status","description":"Provides the real-time information of the device, including the current status of the audio speaker, screen, battery, network, etc.\nUse this tool for: \n1. Answering questions about current condition (e.g. what is the current volume of the audio speaker?)\n2. As the first step to control the device (e.g. turn up / down the volume of the audio speaker, etc.)","inputSchema":{"type":"object","properties":{}}},{"name":"self.audio_speaker.set_volume","description":"Set the volume of the audio speaker. If the current volume is unknown, you must call `self.get_device_status` tool first and then call this tool.","inputSchema":{"type":"object","properties":{"volume":{"type":"integer","minimum":0,"maximum":100}},"required":["volume"]}},{"name":"self.screen.set_brightness","description":"Set the brightness of the screen.","inputSchema":{"type":"object","properties":{"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["brightness"]}},{"name":"self.screen.set_theme","description":"Set the theme of the screen. The theme can be 'light' or 'dark'.","inputSchema":{"type":"object","properties":{"theme":{"type":"string"}},"required":["theme"]}}]}}})
                                        websocket.send(replay_message);
                                        log(`回复MCP消息: ${replay_message}`, 'info');
                                    } else if(payload.method === 'tools/call'){
                                        // 模拟回复
                                        const replay_message = JSON.stringify({"session_id":"9f261599","type":"mcp","payload":{"jsonrpc":"2.0","id": payload.id,"result":{"content":[{"type":"text","text":"true"}],"isError":false}}})
                                        websocket.send(replay_message);
                                        log(`回复MCP消息: ${replay_message}`, 'info');
                                    }
                                }

                            } else {
                                // 未知消息类型
                                log(`未知消息类型: ${message.type}`, 'info');
                                addMessage(JSON.stringify(message, null, 2));
                            }
                        } else {
                            // 处理二进制数据 - 兼容多种二进制格式
                            handleBinaryMessage(event.data);
                        }
                    } catch (error) {
                        log(`WebSocket消息处理错误: ${error.message}`, 'error');
                        // 非JSON格式文本消息直接显示
                        if (typeof event.data === 'string') {
                            addMessage(event.data);
                        }
                    }
                };

                connectionStatus.textContent = 'ws未连接';
                connectionStatus.style.color = 'orange';
            } catch (error) {
                log(`连接错误: ${error.message}`, 'error');
                connectionStatus.textContent = 'ws未连接';
            }
        }

        // 发送hello握手消息
        async function sendHelloMessage() {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

            try {
                const config = getConfig();

                // 设置设备信息
                const helloMessage = {
                    type: 'hello',
                    device_id: config.deviceId,
                    device_name: config.deviceName,
                    device_mac: config.deviceMac,
                    token: config.token,
                    features: {
                        mcp: true
                    }
                };

                log('发送hello握手消息', 'info');
                websocket.send(JSON.stringify(helloMessage));

                // 等待服务器响应
                return new Promise(resolve => {
                    // 5秒超时
                    const timeout = setTimeout(() => {
                        log('等待hello响应超时', 'error');
                        log('提示: 请尝试点击"测试认证"按钮进行连接排查', 'info');
                        resolve(false);
                    }, 5000);

                    // 临时监听一次消息，接收hello响应
                    const onMessageHandler = (event) => {
                        try {
                            const response = JSON.parse(event.data);
                            if (response.type === 'hello' && response.session_id) {
                                log(`服务器握手成功，会话ID: ${response.session_id}`, 'success');
                                clearTimeout(timeout);
                                websocket.removeEventListener('message', onMessageHandler);
                                resolve(true);
                            }
                        } catch (e) {
                            // 忽略非JSON消息
                        }
                    };

                    websocket.addEventListener('message', onMessageHandler);
                });
            } catch (error) {
                log(`发送hello消息错误: ${error.message}`, 'error');
                return false;
            }
        }

        // 断开WebSocket连接
        function disconnectFromServer() {
            if (!websocket) return;

            websocket.close();
            stopRecording();
        }

        // 发送文本消息
        function sendTextMessage() {
            const message = messageInput.value.trim();
            if (message === '' || !websocket || websocket.readyState !== WebSocket.OPEN) return;

            audioBufferQueue = [];
            isAudioBuffering = false;
            isAudioPlaying = false;

            try {
                // 直接发送listen消息，不需要重复发送hello
                const listenMessage = {
                    type: 'listen',
                    mode: 'manual',
                    state: 'detect',
                    text: message
                };

                websocket.send(JSON.stringify(listenMessage));
                addMessage(message, true);
                log(`发送文本消息: ${message}`, 'info');

                messageInput.value = '';
            } catch (error) {
                log(`发送消息错误: ${error.message}`, 'error');
            }
        }

        // 生成随机MAC地址
        function generateRandomMac() {
            const hexDigits = '0123456789ABCDEF';
            let mac = '';
            for (let i = 0; i < 6; i++) {
                if (i > 0) mac += ':';
                for (let j = 0; j < 2; j++) {
                    mac += hexDigits.charAt(Math.floor(Math.random() * 16));
                }
            }
            return mac;
        }

        // 初始化事件监听器
        function initEventListeners() {
            connectButton.addEventListener('click', connectToServer);
            document.getElementById('authTestButton').addEventListener('click', testAuthentication);

            // 设备配置面板折叠/展开
            const toggleButton = document.getElementById('toggleConfig');
            const configPanel = document.getElementById('configPanel');
            const deviceMacInput = document.getElementById('deviceMac');
            const clientIdInput = document.getElementById('clientId');
            const displayMac = document.getElementById('displayMac');
            const displayClient = document.getElementById('displayClient');

            // 从localStorage加载MAC地址，如果没有则生成新的
            let savedMac = localStorage.getItem('deviceMac');
            if (!savedMac) {
                savedMac = generateRandomMac();
                localStorage.setItem('deviceMac', savedMac);
            }
            deviceMacInput.value = savedMac;
            displayMac.textContent = savedMac;

            // 更新显示的值
            function updateDisplayValues() {
                const newMac = deviceMacInput.value;
                displayMac.textContent = newMac;
                displayClient.textContent = clientIdInput.value;
                // 保存MAC地址到localStorage
                localStorage.setItem('deviceMac', newMac);
            }

            // 监听输入变化
            deviceMacInput.addEventListener('input', updateDisplayValues);
            clientIdInput.addEventListener('input', updateDisplayValues);

            // 初始更新显示值
            updateDisplayValues();

            const savedOtaUrl = localStorage.getItem('otaUrl');
            if (savedOtaUrl) {
                document.getElementById('otaUrl').value = savedOtaUrl;
            }

            const savedWsUrl = localStorage.getItem('wsUrl');
            if (savedWsUrl) {
                document.getElementById('serverUrl').value = savedWsUrl;
            }

            // 切换面板显示
            toggleButton.addEventListener('click', () => {
                const isExpanded = configPanel.classList.contains('expanded');
                configPanel.classList.toggle('expanded');
                toggleButton.textContent = isExpanded ? '编辑' : '收起';
            });

            // 标签页切换
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // 移除所有标签页的active类
                    tabs.forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                    // 添加当前标签页的active类
                    tab.classList.add('active');
                    document.getElementById(`${tab.dataset.tab}Tab`).classList.add('active');
                });
            });

            sendTextButton.addEventListener('click', sendTextMessage);
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendTextMessage();
            });

            recordButton.addEventListener('click', () => {
                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            });

            window.addEventListener('resize', initVisualizer);
        }

        // 测试认证
        async function testAuthentication() {
            log('开始测试认证...', 'info');

            const config = getConfig();

            // 显示服务器配置
            log('-------- 服务器认证配置检查 --------', 'info');
            log('请确认config.yaml中的auth配置：', 'info');
            log('1. server.auth.enabled 为 false 或服务器已正确配置认证', 'info');
            log('2. 如果启用了认证，请确认使用了正确的token', 'info');
            log(`3. 或者在allowed_devices中添加了测试设备MAC：${config.deviceMac}`, 'info');

            const serverUrl = serverUrlInput.value.trim();
            if (!serverUrl) {
                log('请输入服务器地址', 'error');
                return;
            }

            // 测试连接
            log('尝试不同认证参数的连接：', 'info');

            // 测试1: 无参数连接
            try {
                log('测试1: 尝试无参数连接...', 'info');
                const ws1 = new WebSocket(serverUrl);

                ws1.onopen = () => {
                    log('测试1成功: 无参数可连接，服务器可能没有启用认证', 'success');
                    ws1.close();
                };

                ws1.onerror = (error) => {
                    log('测试1失败: 无参数连接被拒绝，服务器可能启用了认证', 'error');
                };

                // 5秒后关闭测试连接
                setTimeout(() => {
                    if (ws1.readyState === WebSocket.CONNECTING || ws1.readyState === WebSocket.OPEN) {
                        ws1.close();
                    }
                }, 5000);
            } catch (error) {
                log(`测试1出错: ${error.message}`, 'error');
            }

            // 测试2: 带参数连接
            setTimeout(async () => {
                try {
                    log('测试2: 尝试带token参数连接...', 'info');

                    let url = new URL(serverUrl);
                    url.searchParams.append('token', config.token);
                    url.searchParams.append('device_id', config.deviceId);
                    url.searchParams.append('device_mac', config.deviceMac);

                    const ws2 = new WebSocket(url.toString());

                    ws2.onopen = () => {
                        log('测试2成功: 带token参数可连接', 'success');

                        // 尝试发送hello消息
                        const helloMsg = {
                            type: 'hello',
                            device_id: config.deviceId,
                            device_mac: config.deviceMac,
                            token: config.token
                        };

                        ws2.send(JSON.stringify(helloMsg));
                        log('已发送hello测试消息', 'info');

                        // 监听响应
                        ws2.onmessage = (event) => {
                            try {
                                const response = JSON.parse(event.data);
                                if (response.type === 'hello' && response.session_id) {
                                    log(`测试完全成功! 收到hello响应，会话ID: ${response.session_id}`, 'success');
                                    ws2.close();
                                }
                            } catch (e) {
                                log(`收到非JSON响应: ${event.data}`, 'info');
                            }
                        };

                        // 5秒后关闭
                        setTimeout(() => ws2.close(), 5000);
                    };

                    ws2.onerror = (error) => {
                        log('测试2失败: 带token参数连接被拒绝', 'error');
                        log('请检查token是否正确，或服务器是否接受URL参数认证', 'error');
                    };
                } catch (error) {
                    log(`测试2出错: ${error.message}`, 'error');
                }
            }, 6000);

            log('认证测试已启动，请查看测试结果...', 'info');
        }

        // 帮助函数：ArrayBuffer转Base64
        function arrayBufferToBase64(buffer) {
            let binary = '';
            const bytes = new Uint8Array(buffer);
            const len = bytes.byteLength;
            for (let i = 0; i < len; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            return window.btoa(binary);
        }

        // 使用libopus创建一个Opus编码器
        let opusEncoder = null;
        function initOpusEncoder() {
            try {
                if (opusEncoder) {
                    return true; // 已经初始化过
                }

                if (!window.ModuleInstance) {
                    log('无法创建Opus编码器：ModuleInstance不可用', 'error');
                    return false;
                }

                // 初始化一个Opus编码器
                const mod = window.ModuleInstance;
                const sampleRate = 16000; // 16kHz采样率
                const channels = 1;       // 单声道
                const application = 2048; // OPUS_APPLICATION_VOIP = 2048

                // 创建编码器
                opusEncoder = {
                    channels: channels,
                    sampleRate: sampleRate,
                    frameSize: 960, // 60ms @ 16kHz = 60 * 16 = 960 samples
                    maxPacketSize: 4000, // 最大包大小
                    module: mod,

                    // 初始化编码器
                    init: function () {
                        try {
                            // 获取编码器大小
                            const encoderSize = mod._opus_encoder_get_size(this.channels);
                            log(`Opus编码器大小: ${encoderSize}字节`, 'info');

                            // 分配内存
                            this.encoderPtr = mod._malloc(encoderSize);
                            if (!this.encoderPtr) {
                                throw new Error("无法分配编码器内存");
                            }

                            // 初始化编码器
                            const err = mod._opus_encoder_init(
                                this.encoderPtr,
                                this.sampleRate,
                                this.channels,
                                application
                            );

                            if (err < 0) {
                                throw new Error(`Opus编码器初始化失败: ${err}`);
                            }

                            // 设置位率 (16kbps)
                            mod._opus_encoder_ctl(this.encoderPtr, 4002, 16000); // OPUS_SET_BITRATE

                            // 设置复杂度 (0-10, 越高质量越好但CPU使用越多)
                            mod._opus_encoder_ctl(this.encoderPtr, 4010, 5);     // OPUS_SET_COMPLEXITY

                            // 设置使用DTX (不传输静音帧)
                            mod._opus_encoder_ctl(this.encoderPtr, 4016, 1);     // OPUS_SET_DTX

                            log("Opus编码器初始化成功", 'success');
                            return true;
                        } catch (error) {
                            if (this.encoderPtr) {
                                mod._free(this.encoderPtr);
                                this.encoderPtr = null;
                            }
                            log(`Opus编码器初始化失败: ${error.message}`, 'error');
                            return false;
                        }
                    },

                    // 编码PCM数据为Opus
                    encode: function (pcmData) {
                        if (!this.encoderPtr) {
                            if (!this.init()) {
                                return null;
                            }
                        }

                        try {
                            const mod = this.module;

                            // 为PCM数据分配内存
                            const pcmPtr = mod._malloc(pcmData.length * 2); // 2字节/int16

                            // 将PCM数据复制到HEAP
                            for (let i = 0; i < pcmData.length; i++) {
                                mod.HEAP16[(pcmPtr >> 1) + i] = pcmData[i];
                            }

                            // 为输出分配内存
                            const outPtr = mod._malloc(this.maxPacketSize);

                            // 进行编码
                            const encodedLen = mod._opus_encode(
                                this.encoderPtr,
                                pcmPtr,
                                this.frameSize,
                                outPtr,
                                this.maxPacketSize
                            );

                            if (encodedLen < 0) {
                                throw new Error(`Opus编码失败: ${encodedLen}`);
                            }

                            // 复制编码后的数据
                            const opusData = new Uint8Array(encodedLen);
                            for (let i = 0; i < encodedLen; i++) {
                                opusData[i] = mod.HEAPU8[outPtr + i];
                            }

                            // 释放内存
                            mod._free(pcmPtr);
                            mod._free(outPtr);

                            return opusData;
                        } catch (error) {
                            log(`Opus编码出错: ${error.message}`, 'error');
                            return null;
                        }
                    },

                    // 销毁编码器
                    destroy: function () {
                        if (this.encoderPtr) {
                            this.module._free(this.encoderPtr);
                            this.encoderPtr = null;
                        }
                    }
                };

                const result = opusEncoder.init();
                return result;
            } catch (error) {
                log(`创建Opus编码器失败: ${error.message}`, 'error');
                return false;
            }
        }

        // 初始化应用
        function initApp() {
            initVisualizer();
            initEventListeners();

            // 检查libopus.js是否正确加载
            checkOpusLoaded();

            // 初始化Opus编码器
            initOpusEncoder();

            // 预加载Opus解码器
            log('预加载Opus解码器...', 'info');
            initOpusDecoder().then(() => {
                log('Opus解码器预加载成功', 'success');
            }).catch(error => {
                log(`Opus解码器预加载失败: ${error.message}，将在需要时重试`, 'warning');
            });
        }

        // PCM录音处理器代码 - 会被注入到AudioWorklet中
        const audioProcessorCode = `
            class AudioRecorderProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.buffers = [];
                    this.frameSize = 960; // 60ms @ 16kHz = 960 samples
                    this.buffer = new Int16Array(this.frameSize);
                    this.bufferIndex = 0;
                    this.isRecording = false;

                    // 监听来自主线程的消息
                    this.port.onmessage = (event) => {
                        if (event.data.command === 'start') {
                            this.isRecording = true;
                            this.port.postMessage({ type: 'status', status: 'started' });
                        } else if (event.data.command === 'stop') {
                            this.isRecording = false;

                            // 发送剩余的缓冲区
                            if (this.bufferIndex > 0) {
                                const finalBuffer = this.buffer.slice(0, this.bufferIndex);
                                this.port.postMessage({
                                    type: 'buffer',
                                    buffer: finalBuffer
                                });
                                this.bufferIndex = 0;
                            }

                            this.port.postMessage({ type: 'status', status: 'stopped' });
                        }
                    };
                }

                process(inputs, outputs, parameters) {
                    if (!this.isRecording) return true;

                    const input = inputs[0][0]; // 获取第一个输入通道
                    if (!input) return true;

                    // 将浮点采样转换为16位整数并存储
                    for (let i = 0; i < input.length; i++) {
                        if (this.bufferIndex >= this.frameSize) {
                            // 缓冲区已满，发送给主线程并重置
                            this.port.postMessage({
                                type: 'buffer',
                                buffer: this.buffer.slice(0)
                            });
                            this.bufferIndex = 0;
                        }

                        // 转换为16位整数 (-32768到32767)
                        this.buffer[this.bufferIndex++] = Math.max(-32768, Math.min(32767, Math.floor(input[i] * 32767)));
                    }

                    return true;
                }
            }

            registerProcessor('audio-recorder-processor', AudioRecorderProcessor);
        `;

        // 创建音频处理器
        async function createAudioProcessor() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000,
                    latencyHint: 'interactive'
                });
            }

            try {
                // 检查是否支持AudioWorklet
                if (audioContext.audioWorklet) {
                    // 注册音频处理器
                    const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
                    const url = URL.createObjectURL(blob);
                    await audioContext.audioWorklet.addModule(url);
                    URL.revokeObjectURL(url);

                    // 创建音频处理节点
                    const audioProcessor = new AudioWorkletNode(audioContext, 'audio-recorder-processor');

                    // 设置音频处理消息处理
                    audioProcessor.port.onmessage = (event) => {
                        if (event.data.type === 'buffer') {
                            // 收到PCM缓冲区数据
                            processPCMBuffer(event.data.buffer);
                        }
                    };

                    log('使用AudioWorklet处理音频', 'success');
                    return { node: audioProcessor, type: 'worklet' };
                } else {
                    // 使用旧版ScriptProcessorNode作为回退方案
                    log('AudioWorklet不可用，使用ScriptProcessorNode作为回退方案', 'warning');

                    const frameSize = 4096; // ScriptProcessorNode缓冲区大小
                    const scriptProcessor = audioContext.createScriptProcessor(frameSize, 1, 1);

                    // 将audioProcess事件设置为处理音频数据
                    scriptProcessor.onaudioprocess = (event) => {
                        if (!isRecording) return;

                        const input = event.inputBuffer.getChannelData(0);
                        const buffer = new Int16Array(input.length);

                        // 将浮点数据转换为16位整数
                        for (let i = 0; i < input.length; i++) {
                            buffer[i] = Math.max(-32768, Math.min(32767, Math.floor(input[i] * 32767)));
                        }

                        // 处理PCM数据
                        processPCMBuffer(buffer);
                    };

                    // 需要连接输出，否则不会触发处理
                    // 我们创建一个静音通道
                    const silent = audioContext.createGain();
                    silent.gain.value = 0;
                    scriptProcessor.connect(silent);
                    silent.connect(audioContext.destination);

                    return { node: scriptProcessor, type: 'processor' };
                }
            } catch (error) {
                log(`创建音频处理器失败: ${error.message}，尝试回退方案`, 'error');

                // 最后回退方案：使用ScriptProcessorNode
                try {
                    const frameSize = 4096; // ScriptProcessorNode缓冲区大小
                    const scriptProcessor = audioContext.createScriptProcessor(frameSize, 1, 1);

                    scriptProcessor.onaudioprocess = (event) => {
                        if (!isRecording) return;

                        const input = event.inputBuffer.getChannelData(0);
                        const buffer = new Int16Array(input.length);

                        for (let i = 0; i < input.length; i++) {
                            buffer[i] = Math.max(-32768, Math.min(32767, Math.floor(input[i] * 32767)));
                        }

                        processPCMBuffer(buffer);
                    };

                    const silent = audioContext.createGain();
                    silent.gain.value = 0;
                    scriptProcessor.connect(silent);
                    silent.connect(audioContext.destination);

                    log('使用ScriptProcessorNode作为回退方案成功', 'warning');
                    return { node: scriptProcessor, type: 'processor' };
                } catch (fallbackError) {
                    log(`回退方案也失败: ${fallbackError.message}`, 'error');
                    return null;
                }
            }
        }

        // 初始化直接从PCM数据录音的系统
        let audioProcessor = null;
        let audioProcessorType = null;
        let audioSource = null;

        // 处理PCM缓冲数据
        let pcmDataBuffer = new Int16Array();
        function processPCMBuffer(buffer) {
            if (!isRecording) return;

            // 将新的PCM数据追加到缓冲区
            const newBuffer = new Int16Array(pcmDataBuffer.length + buffer.length);
            newBuffer.set(pcmDataBuffer);
            newBuffer.set(buffer, pcmDataBuffer.length);
            pcmDataBuffer = newBuffer;

            // 检查是否有足够的数据进行Opus编码（16000Hz, 60ms = 960个采样点）
            const samplesPerFrame = 960; // 60ms @ 16kHz

            while (pcmDataBuffer.length >= samplesPerFrame) {
                // 从缓冲区取出一帧数据
                const frameData = pcmDataBuffer.slice(0, samplesPerFrame);
                pcmDataBuffer = pcmDataBuffer.slice(samplesPerFrame);

                // 编码为Opus
                encodeAndSendOpus(frameData);
            }
        }

        // 编码并发送Opus数据
        function encodeAndSendOpus(pcmData = null) {
            if (!opusEncoder) {
                log('Opus编码器未初始化', 'error');
                return;
            }

            try {
                // 如果提供了PCM数据，则编码该数据
                if (pcmData) {
                    // 使用已初始化的Opus编码器编码
                    const opusData = opusEncoder.encode(pcmData);

                    if (opusData && opusData.length > 0) {
                        // 存储音频帧
                        audioBuffers.push(opusData.buffer);
                        totalAudioSize += opusData.length;

                        // 如果WebSocket已连接，则发送数据
                        if (websocket && websocket.readyState === WebSocket.OPEN) {
                            try {
                                // 服务端期望接收原始Opus数据，不需要任何额外包装
                                websocket.send(opusData.buffer);
                                log(`发送Opus帧，大小：${opusData.length}字节`, 'debug');
                            } catch (error) {
                                log(`WebSocket发送错误: ${error.message}`, 'error');
                            }
                        }
                    } else {
                        log('Opus编码失败，无有效数据返回', 'error');
                    }
                } else {
                    // 处理剩余的PCM数据
                    if (pcmDataBuffer.length > 0) {
                        // 如果剩余的采样点不足一帧，用静音填充
                        const samplesPerFrame = 960;
                        if (pcmDataBuffer.length < samplesPerFrame) {
                            const paddedBuffer = new Int16Array(samplesPerFrame);
                            paddedBuffer.set(pcmDataBuffer);
                            // 剩余部分为0（静音）
                            encodeAndSendOpus(paddedBuffer);
                        } else {
                            encodeAndSendOpus(pcmDataBuffer.slice(0, samplesPerFrame));
                        }
                        pcmDataBuffer = new Int16Array(0);
                    }
                }
            } catch (error) {
                log(`Opus编码错误: ${error.message}`, 'error');
            }
        }

        // 开始直接从PCM数据录音
        async function startDirectRecording() {
            if (isRecording) return;

            try {
                // 初始化Opus编码器
                if (!initOpusEncoder()) {
                    log('无法启动录音: Opus编码器初始化失败', 'error');
                    return;
                }

                // 请求麦克风权限
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000,
                        channelCount: 1
                    }
                });

                // 创建音频上下文和分析器
                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 16000,
                        latencyHint: 'interactive'
                    });
                }

                // 创建音频处理器
                const processorResult = await createAudioProcessor();
                if (!processorResult) {
                    log('无法创建音频处理器', 'error');
                    return;
                }

                audioProcessor = processorResult.node;
                audioProcessorType = processorResult.type;

                // 连接音频处理链
                audioSource = audioContext.createMediaStreamSource(stream);
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 2048;

                audioSource.connect(analyser);
                audioSource.connect(audioProcessor);

                // 启动录音
                pcmDataBuffer = new Int16Array();
                audioBuffers = [];
                totalAudioSize = 0;
                isRecording = true;

                // 启动音频处理器的录音 - 只有AudioWorklet才需要发送消息
                if (audioProcessorType === 'worklet' && audioProcessor.port) {
                    audioProcessor.port.postMessage({ command: 'start' });
                }

                // 发送监听开始消息
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    // 使用与服务端期望的listen消息格式
                    const listenMessage = {
                        type: 'listen',
                        mode: 'manual',  // 使用手动模式，由我们控制开始/停止
                        state: 'start'   // 表示开始录音
                    };

                    log(`发送录音开始消息: ${JSON.stringify(listenMessage)}`, 'info');
                    websocket.send(JSON.stringify(listenMessage));
                } else {
                    log('WebSocket未连接，无法发送开始消息', 'error');
                    return false;
                }

                // 开始音频可视化
                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                drawVisualizer(dataArray);

                // 在UI上显示录音计时器
                let recordingSeconds = 0;
                const recordingTimer = setInterval(() => {
                    recordingSeconds += 0.1;
                    recordButton.textContent = `停止录音 ${recordingSeconds.toFixed(1)}秒`;
                }, 100);

                // 保存计时器，以便在停止时清除
                window.recordingTimer = recordingTimer;

                recordButton.classList.add('recording');
                recordButton.disabled = false;

                log('开始PCM直接录音', 'success');
                return true;
            } catch (error) {
                log(`直接录音启动错误: ${error.message}`, 'error');
                isRecording = false;
                return false;
            }
        }

        // 停止直接从PCM数据录音
        function stopDirectRecording() {
            if (!isRecording) return;

            try {
                // 停止录音
                isRecording = false;

                // 停止音频处理器的录音
                if (audioProcessor) {
                    // 只有AudioWorklet才需要发送停止消息
                    if (audioProcessorType === 'worklet' && audioProcessor.port) {
                        audioProcessor.port.postMessage({ command: 'stop' });
                    }

                    audioProcessor.disconnect();
                    audioProcessor = null;
                }

                // 断开音频连接
                if (audioSource) {
                    audioSource.disconnect();
                    audioSource = null;
                }

                // 停止可视化
                if (visualizationRequest) {
                    cancelAnimationFrame(visualizationRequest);
                    visualizationRequest = null;
                }

                // 清除录音计时器
                if (window.recordingTimer) {
                    clearInterval(window.recordingTimer);
                    window.recordingTimer = null;
                }

                // 编码并发送剩余的数据
                encodeAndSendOpus();

                // 发送一个空的消息作为结束标志（模拟接收到空音频数据的情况）
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    // 使用空的Uint8Array发送最后一个空帧
                    const emptyOpusFrame = new Uint8Array(0);
                    websocket.send(emptyOpusFrame);

                    // 发送监听结束消息
                    const stopMessage = {
                        type: 'listen',
                        mode: 'manual',
                        state: 'stop'
                    };

                    websocket.send(JSON.stringify(stopMessage));
                    log('已发送录音停止信号', 'info');
                }

                // 重置UI
                recordButton.textContent = '开始录音';
                recordButton.classList.remove('recording');
                recordButton.disabled = false;

                log('停止PCM直接录音', 'success');
                return true;
            } catch (error) {
                log(`直接录音停止错误: ${error.message}`, 'error');
                return false;
            }
        }

        async function handleBinaryMessage(data) {
            try {
                let arrayBuffer;

                // 根据数据类型进行处理
                if (data instanceof ArrayBuffer) {
                    arrayBuffer = data;
                    log(`收到ArrayBuffer音频数据，大小: ${data.byteLength}字节`, 'debug');
                } else if (data instanceof Blob) {
                    // 如果是Blob类型，转换为ArrayBuffer
                    arrayBuffer = await data.arrayBuffer();
                    log(`收到Blob音频数据，大小: ${arrayBuffer.byteLength}字节`, 'debug');
                } else {
                    log(`收到未知类型的二进制数据: ${typeof data}`, 'warning');
                    return;
                }

                // 创建Uint8Array用于处理
                const opusData = new Uint8Array(arrayBuffer);

                if (opusData.length > 0) {
                    // 将数据添加到缓冲队列
                    audioBufferQueue.push(opusData);

                    // 如果收到的是第一个音频包，开始缓冲过程
                    if (audioBufferQueue.length === 1 && !isAudioBuffering && !isAudioPlaying) {
                        startAudioBuffering();
                    }
                } else {
                    log('收到空音频数据帧，可能是结束标志', 'warning');

                    // 如果缓冲队列中有数据且没有在播放，立即开始播放
                    if (audioBufferQueue.length > 0 && !isAudioPlaying) {
                        playBufferedAudio();
                    }

                    // 如果正在播放，发送结束信号
                    if (isAudioPlaying && streamingContext) {
                        streamingContext.endOfStream = true;
                    }
                }
            } catch (error) {
                log(`处理二进制消息出错: ${error.message}`, 'error');
            }
        }

        // 获取配置值
        function getConfig() {
            const deviceMac = document.getElementById('deviceMac').value.trim();
            return {
                deviceId: deviceMac,  // 使用MAC地址作为deviceId
                deviceName: document.getElementById('deviceName').value.trim(),
                deviceMac: deviceMac,
                clientId: document.getElementById('clientId').value.trim(),
                token: document.getElementById('token').value.trim()
            };
        }

        // 验证配置
        function validateConfig(config) {
            if (!config.deviceMac) {
                log('设备MAC地址不能为空', 'error');
                return false;
            }
            if (!config.clientId) {
                log('客户端ID不能为空', 'error');
                return false;
            }
            return true;
        }

        initApp();
    </script>
</body>

</html>
            """,  # 完整 HTML 内容
            min_height=800,  # 设置最小高度确保显示完整
            container=True  # 启用容器样式
        )

    # ------------ 事件处理 ------------
    connect_btn.click(
        fn=connect_server,
        inputs=[device_id, client_id],
        outputs=[ota_status]
    )
    # 验证页面事件
    send_pict_btn.click(
        fn=lambda did, cid, iu, te: send_media(did, cid, "send", iu, None, te),
        inputs=[device_id, client_id, image_upload, media_text],
        outputs=[media_output]
    )
    send_audio_btn.click(
        fn=lambda did, cid, au, te: send_media(did, cid, "send", None, au, te),
        inputs=[device_id, client_id, audio_upload, media_text],
        outputs=[media_output]
    )
    # 聊天页面事件
    text_send_btn.click(
        fn=send_message,
        inputs=[device_id, client_id, text_input, None],
        outputs=[chatbot, log_display]
    )
    voice_send_btn.click(
        fn=lambda: gr.Audio(visible=True),
        inputs=None,
        outputs=[voice_recorder]
    )
    voice_recorder.stop_recording(
        fn=send_message,
        inputs=[device_id, client_id, None, voice_recorder],
        outputs=[voice_recorder, chatbot, log_display]
    )

    # ------------ cosyvoice 事件处理 ------------
    # 定义按钮触发事件
    # 统一的事件监听入口，可替代组件的 .click()、.submit() 等方法，语法更简洁。
    # gr.on(触发器, 目标函数, 输入, 输出)。
    submit_event = gr.on(
        # text_voi.submit，监听用户在文本框（gr.Textbox）中按下 Enter 键 的提交行为，触发绑定的函数
        # 与按钮点击事件（.click()）不同，此事件由键盘交互触发
        triggers=[submit_btn.click, text_voi.submit],
        fn=chat_predict,
        inputs=[
            text_voi, audio_input, image_input, video_input, chatbot,
            system_prompt_textbox, voice_choice
        ],
        outputs=[
            text_voi, audio_input, image_input, video_input, chatbot
        ]
    )
    # 定义按钮更新动作（lambda -> outputs）, 并指定触发事件机制（cancels -> 取消正在进行的任务 -> 针对于流式任务才有效（如AI生成文本），提供用户中途停止）
    stop_btn.click(fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
       inputs=None,
       outputs=[submit_btn, stop_btn],
       cancels=[submit_event],  # 终止指定的异步任务
       queue=False  # 表示立即执行，不等待排队
    )
    clear_btn.click(fn=clear_chat_history,
        inputs=None,
        outputs=[
            chatbot, text_voi, audio_input, image_input,
            video_input
        ]
    )


def _get_args():
    '''
    prog	    sys.argv	程序名称（帮助信息中显示）
    description	None	帮助信息顶部的功能描述文本
    epilog	    None	帮助信息底部的补充说明文本
    add_help	True	是否自动添加 -h/--help 帮助选项
    formatter_class 	定制帮助信息的排版格式（如 argparse.RawTextHelpFormatter 保留换行格式）
    prefix_chars	-	定义可选参数前缀（如 + 可改为 +f）
    '''
    parser = ArgumentParser()
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    '''
    action	参数触发行为      store 保存值（默认），store_true 存在则置为 True，append 多次出现保存为列表
    nargs	参数接收数量      ? = 0或1个，* = 0或多个，+ = 1或多个
    choices	参数值范围限制     choices=['start', 'stop', 'status']
    metavar	帮助信息中显示的参数名 metavar="<file>" → 显示为 --file <file>
    dest	解析结果中的属性名 dest="output" → 通过 args.output 访问
    '''
    parser.add_argument('--browser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    '''
    name/flags	位置参数（如 file）或可选参数（如 -f, --file）     parser.add_argument('input_file')
    type	    参数类型转换（如 int, float, str）               type=int → 输入转为整数
    help	    参数的帮助说明文本                               help="Path to input file"
    default	    未提供参数时的默认值（仅适用于可选参数）             default="output.txt"
    required	是否必须提供（仅可选参数可用）                     required=True
    
    位置参数（如 input_file）必须按顺序提供，不能省略。可选参数（如 --compress）可通过前缀标识顺序无关。
    '''
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Demo server name.')
    parser.add_argument('--ui-language', type=str, choices=['en', 'zh'], default='en',
                        help='Display language for the UI.')
    return parser.parse_args()


# 启动应用
# python gradio-pro.py --browser --share
if __name__ == "__main__":
    args = _get_args()
    demo.queue(default_concurrency_limit=10, max_size=100).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        favicon_path=None,
        max_threads=100,
        ssr_mode=False,
        show_error=True,
        show_api=True,
        inbrowser=args.browser,
        allowed_paths=[current_dir]  # 必须声明
    )