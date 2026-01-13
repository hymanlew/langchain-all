"""
在 Ubuntu 上安装：sudo apt install ffmpeg

在 Windows 上安装：
- 找到 Windows 版本的 ffmpeg 压缩包下载：https://www.gyan.dev/ffmpeg/builds/
- 解压并配置 ffmpeg：解压到文件夹，并添加 ffmpeg bin 目录到系统系统变量 path 路径

安装 opencv-python 和 moviepy
pip install opencv-python
pip install moviepy

使用 --quiet 标志来抑制安装过程中的详细输出。可以通过以下 Python 脚本来验证安装是否成功：
import cv2
import moviepy.editor as mp
print("OpenCV version:", cv2.version)
print("MoviePy version:", mp.version)
"""
import os
import cv2  # 视频处理
import base64  # 编码帧
import moviepy.editor as mp # 处理音频
from openai import OpenAI

VIDEO_FILE = "Good_Driver.mp4"

# 提取视频帧和音频
def extract_frames_and_audio(video_file, interval=2):
    """
    :param video_file: 视频文件路径
    :param interval: 每提取一帧所间隔的秒数

    使用OpenCV遍历视频并提取指定间隔的帧。将每个帧编码为Base64格式并存储在列表中。
    使用MoviePy从视频中提取音频并保存为MP3文件。

    :return:
    """
    encoded_frames = []
    file_name, _ = os.path.splitext(video_file)

    video_capture = cv2.VideoCapture(video_file)
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frames_interval = int(frame_rate * interval)
    current_frame = 0

    # 循环遍历视频并以指定的采样率提取帧
    while current_frame < total_frame_count - 1:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = video_capture.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        encoded_frames.append(base64.b64encode(buffer).decode("utf-8"))
        current_frame += frames_interval
    video_capture.release()

    # 从视频中提取音频
    audio_output = f"{file_name}.mp3"
    video_clip = mp.VideoFileClip(video_file)
    video_clip.audio.write_audiofile(audio_output, bitrate="32k")
    video_clip.audio.close()
    video_clip.close()

    print(f"提取了 {len(encoded_frames)} 帧")
    print(f"音频提取到 {audio_output}")
    return encoded_frames, audio_output

# 每2秒提取1帧（采样率）
encoded_frames, audio_output = extract_frames_and_audio(VIDEO_FILE, interval=2)

# 创建OpenAI客户端
client = OpenAI()
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {"role": "system", "content": "请用Markdown格式生成视频的介绍."},
        {"role": "user", "content": [
            "下面是视频的图像帧",
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, encoded_frames)
        ]},
    ],
    temperature=0,
)
# 打印生成的Markdown格式介绍
print(response.choices[0].message.content)


# 使用GPT-4o模型根据视频内容回答问题
QUESTION = "图中的人做了什么?"

qa_response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content": "请用Markdown格式根据视频内容回答问题."},
    {"role": "user", "content": [
        "下面是视频的图像帧.",
        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, encoded_frames),
        QUESTION
        ],
    }
    ],
    temperature=0,
)
# 打印生成的Markdown格式问题回答
print(QUESTION + "\n" + qa_response.choices[0].message.content)


import time
from IPython.display import Image, display, Audio

# 创建显示句柄，以动态更新的显示内容
display_handle = display(None, display_id=True)

# 显示提取的帧，每帧之间暂停0.025秒
for frame in encoded_frames:
    display_handle.update(Image(data=base64.b64decode(frame.encode("utf-8")), width=600))
    time.sleep(0.025)

# 显示提取的音频
Audio(audio_output)
