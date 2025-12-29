from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config import ALLOWED_TOKENS

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# 声明安全依赖项（例如认证和授权），获取请求头中数据 APIKeyHeader
async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 缺失"
        )
    if api_key not in ALLOWED_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无效的 API Key"
        )
    return api_key