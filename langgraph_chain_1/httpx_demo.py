import asyncio
import httpx
import logging
from typing import AsyncGenerator, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局共享的、配置良好的异步客户端
# 重要：在长时间运行的应用中，此客户端应在应用启动时创建，并在关闭时显式清理。
_client: Optional[httpx.AsyncClient] = None


def get_async_client() -> httpx.AsyncClient:
    """获取全局共享的异步客户端（单例模式）。"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            # 配置连接池限制，防止耗尽资源
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=100),
            # 设置默认超时（连接、读取、写入）
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=1.0),
            # 自动跟随重定向（按需调整）
            follow_redirects=True,
        )
        logger.info("全局 HTTP 异步客户端已初始化。")
    return _client


async def close_async_client():
    """关闭全局客户端（应在应用退出时调用）。"""
    global _client
    if _client:
        await _client.aclose()
        _client = None
        logger.info("全局 HTTP 异步客户端已关闭。")


@retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避等待
    retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),  # 仅对网络和超时错误重试
    reraise=True  # 重试次数用尽后，重新抛出原始异常
)
async def stream_response_production(
        url: str,
        *,
        max_line_length: int = 65536,
        encoding_fallback: str = 'utf-8',
        chunk_size: int = 4096
) -> AsyncGenerator[str, None]:
    """
    生产级别安全的流式读取器。

    参数:
        url: 请求地址。
        max_line_length: 单行最大字节数，防止内存耗尽攻击。
        encoding_fallback: 当响应头未指定编码时使用的默认编码。
        chunk_size: 每次从网络读取的字节块大小。

    返回:
        一个异步生成器，每次 yield 一行文本。

    抛出:
        httpx.HTTPStatusError: 当响应状态码为4xx/5xx时。
        httpx.RequestError: 当请求本身失败时。
    """
    client = get_async_client()
    buffer = bytearray()

    try:
        # 1. 发起请求，明确设置流式模式
        async with client.stream('GET', url, timeout=30.0) as response:
            # 2. 检查HTTP状态码，非2xx状态码会引发 httpx.HTTPStatusError
            response.raise_for_status()

            # 3. 确定编码
            content_type = response.headers.get('content-type', '')
            # 这里可以更复杂地解析 content-type 中的 charset
            encoding = response.encoding or encoding_fallback
            logger.info(f"开始流式读取 {url}, 编码: {encoding}")

            # 4. 按字节块迭代
            async for byte_chunk in response.aiter_bytes(chunk_size):
                if not byte_chunk:
                    continue

                # 高效地将字节添加到缓冲区
                buffer.extend(byte_chunk)

                # 5. 按行分割（防止行过长）
                while True:
                    # 查找换行符
                    newline_pos = buffer.find(b'\n')
                    if newline_pos == -1:
                        # 没找到换行符，检查缓冲区是否过大
                        if len(buffer) > max_line_length:
                            logger.warning(f"单行数据超过最大长度限制({max_line_length} bytes)，强制截断。")
                            # 处理过长的行：将超出部分作为一行解码，并清空缓冲区
                            line = buffer[:max_line_length]
                            del buffer[:max_line_length]
                            try:
                                yield line.decode(encoding, errors='replace')
                            except Exception as e:
                                yield f"[行解码失败: {e}]"
                        break  # 等待更多数据

                    # 找到换行符，提取一行（包含换行符本身）
                    line_bytes = bytes(buffer[:newline_pos + 1])
                    del buffer[:newline_pos + 1]  # 从缓冲区移除已处理部分

                    # 6. 解码并 yield
                    try:
                        line_text = line_bytes.decode(encoding, errors='strict')
                    except UnicodeDecodeError:
                        # 解码失败，使用替换字符并记录警告
                        logger.warning(f"解码失败，使用替换字符。字节长度: {len(line_bytes)}")
                        line_text = line_bytes.decode(encoding, errors='replace')
                    yield line_text

    except httpx.TimeoutException as e:
        logger.error(f"请求超时: {url}, 错误: {e}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 错误状态码: {e.response.status_code} for {url}")
        # 可以在这里根据状态码决定是否重试
        raise
    except httpx.RequestError as e:
        logger.error(f"请求失败: {url}, 错误: {e}")
        raise
    except Exception as e:
        logger.critical(f"处理流时发生未预期的异常: {url}, 错误类型: {type(e).__name__}, 详情: {e}")
        raise
    finally:
        # 7. 处理缓冲区中剩余的任何数据（最后一行可能没有换行符）
        if buffer:
            try:
                remaining_line = buffer.decode(encoding_fallback, errors='replace')
                yield remaining_line
            except Exception as e:
                logger.warning(f"处理最终缓冲区数据时失败: {e}")
                yield f"[最终数据块解码失败]"
        logger.info(f"流式读取结束: {url}")


# 使用示例
async def main_production():
    url = "https://api.example.com/stream"
    try:
        async for line in stream_response_production(url, chunk_size=8192):
            # 在这里处理每一行业务逻辑
            print(line, end='')
    except httpx.RequestError as e:
        print(f"请求失败，无法继续: {e}")
        # 这里可以进行业务级的错误处理，如通知用户、写入死信队列等
    except Exception as e:
        print(f"发生未处理的异常: {e}")


if __name__ == "__main__":
    asyncio.run(main_production())