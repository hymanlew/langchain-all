'''
PyAudio 是跨平台的 python 音频库，可以借助 PyAudio 进行录制和播放音频。
wave 模块可以处理 wav 格式的音频。并且它是标准库，不需要安装。
pip3 install pyaudio

WAV 是 PyAudio 最常搭配使用的格式，因为 wave 模块是 Python 标准库。但也可以结合其他库，处理更多格式：
MP3: 需要 pydub 或 librosa 等库进行编解码
FLAC: 需要 soundfile 等库
OGG: 需要额外编解码器
AIFF: 可以使用 aifc 模块

处理 mp3：
from pydub import AudioSegment
# 将WAV转MP3
sound = AudioSegment.from_wav("output.wav")
sound.export("output.mp3", format="mp3")
# 播放MP3 (仍需要PyAudio作为后端)
sound = AudioSegment.from_mp3("output.mp3")
play(sound)
'''
import pyaudio
import wave
import time


class Recorder:
    """录音"""
    def __init__(self):
        self.format = pyaudio.paInt16  # 采样位数 16bit
        self.channels = 1              # 通道数
        self.width = 2                 # 采样位数 2字节
        self.rate = 22058              # 采样频率
        self.recording = False         # 录音状态
        self.paudio = pyaudio.PyAudio()     # Pyaudio对象

    def record(self, filename, t=5):
		'''
		in_data: 包含音频数据的字节串
		frame_count: 帧数（样本数）
		time_info: 时间信息字典
		status: 流状态标志

		返回一个元组 (in_data, pyaudio.paContinue)
		in_data: 这里返回原始数据（对于输入流通常不需要修改）
		pyaudio.paContinue: 告诉PortAudio继续流处理
		'''
        def callback(in_data, frame_count, time_info, status):
            frames.append(in_data)
            return in_data, pyaudio.paContinue

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "开始录音：", filename)
		
        self.recording = True
        frames = []     # 存放音频数据
        stream = self.paudio.open(format=self.format,
                         channels=self.channels,
                         rate=self.rate,
                         input=True,	# True表示输入流-录音，False表示输出流-播放（旧的写法）
                         stream_callback=callback)
        for i in range(t*1000):  # 设置等待时间
            time.sleep(0.001)
            if not self.recording:
                break
        stream.close()  # 关闭录音端口
		
        # 保存录音文件
        wf = wave.open(filename, 'wb')  # 打开一个文件，用于写入音频, "write bytes" 以二进制模式写入（且是必需的）
        wf.setnchannels(self.channels)  # 保存声道数
        wf.setsampwidth(self.width)  # 保存采样位数
        wf.setframerate(self.rate)  # 保存采样频率
        wf.writeframes(b''.join(frames))  # 将音频数据字节写入到文件中
        wf.close()  # 关闭文件，进行保存
		
        self.recording = False
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "录音结束：", filename)
        return filename

    def stop(self):
        self.recording = False

    def is_recording(self):
        return self.recording


class Player:
    def __init__(self):
        """播放"""
        self.playing = False
        self.paudio = pyaudio.PyAudio()     # Pyaudio对象

    def play(self, filename):
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            return data, pyaudio.paContinue

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "开始播放音频：", filename)
		
        self.playing = True
        wf = wave.open(filename, "rb")
        stream = self.paudio.open(format=self.paudio.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True,
                             stream_callback=callback)
        while stream.is_active() and self.playing:
            time.sleep(0.1)
        stream.close()
        wf.close()
        self.playing = False
		
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "播放音频结束：", filename)

    def stop(self):
        self.playing = False

    def is_playing(self):
        return self.playing


"""
百度语音识别，包含语音和视觉开发等工具。
- 百度AI开放平台：http://ai.baidu.com/
- 语音识别技术文档：https://ai.baidu.com/ai-doc/SPEECH/6k38lxjid
- 安装：pip3 install baidu-aip

获取 app id：进入百度AI控制台 https://console.bce.baidu.com(需要登陆)：
- 选择人脸识别 - 创建应用 - 选择使用的服务 - 创建应用
- 这个 AI 应用有使用次数限制

语音识别仅支持以下格式：
- pcm 格式（不压缩)、wav（不压缩，pcm编码)、amr（有损压缩格式)，8k/16k 采样率，16bit 位深的单声道。
- 频率：采用率二选 8000 或者 16000。(我们使用16000)
- 声道:单声道
"""
from aip import AipSpeech

class BaiduASR():
    def __init__(self):
        appid = ''
        api_key = ''
        secret_key = ''
        self.client = AipSpeech(appid, api_key, secret_key)
		
		'''
		1537 中文
		1737 英文
		1637 粤语
		'''
        #self.dev_pid = 1537   # 支持的语言编号
		self.dev_pid = 1536   # 普通话支持简单英文
        self.per = 4      # 度小宇=1，度小美=0，度逍遥=3，度丫丫=4

	# 语音识别
    def transcribe(self, filename):
        # 读取本地文件
        wav = open(filename, "rb").read()
		'''
		调用百度AI进行语音识别
		wav 是音频的内容，
		pcm 是文件格式，
		16000 是采样率，固定值
		{} 字典中的内容是一系列的可选参数，但只有 dev_pid 比较有用，表示支持的语言。
        '''
		res = self.client.asr(wav, 'pcm', 16000, {'dev_pid': self.dev_pid,})
        # print(res)
        if res['err_no'] == 0:
            result = ''.join(res['result'])
            ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(ctime, "百度语音识别成功：", result)
            return result
        else:
            return ''

	# 文字转语音、语音合成
    def get_speech(self, tex):
		'''
		tex：字符串，表示要生成语音的文本，使用 UTF-8 编码，请注意文本长度必须小于 1024字节
		zh：固定值，表示语言是中英混合模式
		1：客户端类型，web端为 1，不需要改变
		per：发音人选择，度小宇=1，度小美=0，度逍遥=3，度丫丫=4
		aue：音频格式选择为 wav==6，固定值
		'''
        result = self.client.synthesis(tex, "zh", 1, {'per': self.per, 'aue': 6})

        # 识别正确，则返回语音二进制，错误则返回 dict 参照下面错误码
        if not isinstance(result, dict):
            filepath = self.write_wav_file(result)
            ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(ctime, "百度语音合成成功：", filepath)
            return filepath
        else:
            print('语音合成失败！')

    def write_wav_file(self, data):
        """
        写入wav文件
        :param data: 数据
        :returns: 文件保存后的路径
        """
        filepath = "temp_tts.wav"
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(1)  # 写入声道 1
        wf.setsampwidth(2)  # 写入采样位数 2字节 (16bit)
        wf.setframerate(16000)  # 写入声音频率 16000
        wf.writeframes(data)  # 写入声音的二进制内容
        return filepath




if __name__ == '__main__':
	filename = "test.wav"
	
	# 录音
    r = Recorder(filename)
    r.record()
	
	# 多线程调用执行，但注意有 GIL 全局锁，可以使用协程或是进程
	t=threading.Thread(target=r.record,args=(name, during))
	t.daemon = True
	t.start()

	# 语音识别
    asr = BaiduASR()
    asr.transcribe(fp)
	
	# 播放
	p = Player()
    p.play(filename)
	
	# 文字转语音，播放
	filename = asr.get_speech("今天天气真好")
    p.play(filename)
	
	