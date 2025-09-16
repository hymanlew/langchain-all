# 自定义 client 进行访问 fastAPI 部署的 API 服务（see 01-prompt.py）
from langserve import RemoteRunnable

if __name__ == '__main__':
    # 由于是调的 invoke 函数，它会自动在路径后加 /invoke 路径
    client = RemoteRunnable('http://127.0.0.1:8000/chainDemo/')
    print(client.invoke({'language': 'italian', 'text': '你好！'}))