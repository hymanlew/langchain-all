"""
用户请求
    │
    ▼
FastAPI (业务逻辑、身份验证、输入验证)
    │
    ▼
LangGraph (有状态工作流编排)
    │
    ▼
One‑API (统一 API 网关、密钥管理、负载均衡、故障转移)
    │
    ├── 渠道1 (模型 A)
    ├── 渠道2 (模型 B)
    └── 渠道3 (模型 C)
"""
# .env
# One-API Web 管理界面，添加的渠道（即模型后端）和令牌，设置额度、模型权限
ONE_API_BASE_URL=http://localhost:3000/v1
ONE_API_MASTER_KEY=sk-your-master-key-here
ALLOWED_TOKENS=sk-user1-token,sk-user2-token  # 允许的客户端令牌


import os
from dotenv import load_dotenv

load_dotenv()

ONE_API_BASE_URL = os.getenv("ONE_API_BASE_URL")
ONE_API_MASTER_KEY = os.getenv("ONE_API_MASTER_KEY")
ALLOWED_TOKENS = os.getenv("ALLOWED_TOKENS", "").split(",")
