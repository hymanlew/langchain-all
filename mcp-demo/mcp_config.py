# config.py - 集中管理配置
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
	# 定义远程 MCP 服务配置（建议从环境变量读取）
    # 天气查询，需先启动 weather_server.py
	MCP_SERVERS = {
        "weather": {
            "url": os.getenv("MCP_WEATHER_URL", "http://localhost:8000/streamable"),
            "transport": "streamable_http",
            "timeout": int(os.getenv("MCP_TIMEOUT", "30")),	# 增加超时设置
			"retry_policy": {
				"max_attempts": 3,
				"delay": 1
			}
        }
    }
	
    MCP_TEXT2SQL_CONFIG = {
		"sql": {
			"url": os.getenv("MCP_TEXT2SQL_URL", "http://localhost:8000/streamable"),
            "transport": "streamable_http",
            "timeout": int(os.getenv("MCP_TIMEOUT", "30")),	# 增加超时设置
			"endpoint": "tcp://text2sql-service:50051",
			"retry_policy": {
				"max_attempts": 3,
				"delay": 1
			}
		},
		"nlp": {
			"transport": "http",
			"endpoint": "https://nlp-service/api"
		}
	}
	
    GRADIO_AUTH = (
        os.getenv("GRADIO_USERNAME", "admin"),
        os.getenv("GRADIO_PASSWORD", "secure_password")
    )
    
	# 安全配置
	GRADIO_CONFIG = {
		"debug": False,  # 生产环境关闭debug模式
		"share": False,  # 不生成公开分享链接
		"auth": ("admin", "secure_password"),  # 基本认证
		"server_name": "0.0.0.0",
		"server_port": 7860,
		"allowed_paths": [],  # 限制文件访问
	}

	# 企业级UI配置
	CSS = '''
	.gradio-container {font-family: "Arial", sans-serif}
	.chatbot {min-height: 500px}
	.warning {color: #ff6b6b}
	'''

    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
    