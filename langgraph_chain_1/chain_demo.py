import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from langchain_classic.chains.sequential import SimpleSequentialChain
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableConfig


class MonitoringCallback(BaseCallbackHandler):
    """企业级监控回调中间件"""
    def __init__(self, run_id: str = None):
        super().__init__()
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "duration_ms": None
        }

    def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            *,
            run_id: str,
            parent_run_id: Optional[str] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> None:
        """链开始时触发"""
        self.metrics["start_time"] = datetime.now()
        print(f"[{self.run_id}] 链开始执行 {tags}, 输入: {json.dumps(inputs, ensure_ascii=False, default=str)}")

        # 生产环境可记录到监控系统
        # self.log_to_monitoring("chain_start", inputs)

    def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: str,
            parent_run_id: Optional[str] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        """链结束时触发"""
        self.metrics["end_time"] = datetime.now()
        duration = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds() * 1000
        self.metrics["duration_ms"] = duration

        print(f"[{self.run_id}] 链执行完成  {tags}")
        print(f"   输出: {json.dumps(outputs, ensure_ascii=False, default=str)}")
        print(f"   耗时: {duration:.2f} ms")

        # 性能监控
        if duration > 1000:  # 超过1秒记录警告
            print(f"   性能警告: 执行时间过长")

    def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: str,
            parent_run_id: Optional[str] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        """链错误时触发"""
        print(f"[{self.run_id}] - {tags} 链执行错误: {error}")

        # 生产环境应记录到错误追踪系统
        # self.log_to_error_tracking(error, run_id)

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[监控] 调用LLM，Prompt: {prompts[0][:50]}...")
        print(f"[监控] 调用LLM，{tags}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print("工具调用开始")


def validate_user_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """验证用户输入"""
    if "user_id" not in input_data:
        raise ValueError("缺少必要参数: user_id")
    if "amount" in input_data and input_data["amount"] <= 0:
        raise ValueError("金额必须大于0")

    # 添加验证标记
    input_data["_validated"] = True
    input_data["_timestamp"] = datetime.now().isoformat()
    return input_data

def process_payment(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """处理支付逻辑"""
    # 模拟业务处理
    user_id = input_data["user_id"]
    amount = input_data.get("amount", 0)

    # 业务逻辑
    result = {
        "transaction_id": f"tx_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "user_id": user_id,
        "amount": amount,
        "status": "success",
        "processed_at": datetime.now().isoformat(),
        "message": f"用户 {user_id} 支付 {amount} 元成功"
    }

    # 模拟处理耗时
    import time
    time.sleep(0.5)  # 模拟处理时间

    return result


# 标签列表，用于分类和过滤
tags = RunnableConfig(tags=['print'])

# 构建链: 数据处理 -> 提示词 -> LLM -> 输出解析
# 使用 .with_config() 方法预先配置链，之后使用 callback 进行处理
# RunnablePassthrough.assign()
chain = (
    RunnableLambda(validate_user_input).with_config(tags)
    | RunnableLambda(process_payment).with_config(tags)
    # | prompt
    # | llm
    # | output_parser
)

# 异步调用时，通过字典形式传入 RunnableConfig
async def main():
    input_text = {
        "user_id": 1,
        "amount": 10,
    }

    config = RunnableConfig(
        max_concurrency=2,
        # tags=["my_chain"], # 这里的标签列表，是用于整个 chain 的流程中
        run_name="payment_chain",  # 链运行名称，用于追踪（当系统中有多个 chain 时可区分）
        callbacks=[
            MonitoringCallback(run_id="payment_001"),
            # ConsoleCallbackHandler(),  # LangChain 内置的控制台回调，打印入参，出参
        ],
        metadata={ # 元数据，会传递给回调
            "environment": "production",
            "service": "payment-service",
            "version": "1.0.0",
            "team": "fintech-team"
        },
        configurable={"request_id": str(uuid.uuid4())}
    )
    result = await chain.ainvoke(input_text, config)
    print(result)


# 运行
if __name__ == "__main__":
    asyncio.run(main())

