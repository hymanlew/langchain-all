import asyncio
import uuid

from langchain_classic.chains.sequential import SimpleSequentialChain
from langchain_core.callbacks import (BaseCallbackHandler,
                                      CallbackManager)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableConfig


class MonitoringCallback(BaseCallbackHandler):
    """企业级监控回调中间件"""
    def on_chain_start(self, serialized, inputs, **kwargs):
        request_id = kwargs.get("run_id")
        print(f"[监控] 链开始执行，请求ID: {request_id}, 输入: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"[监控] 链执行结束，输出: {outputs}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[监控] 调用LLM，Prompt: {prompts[0][:50]}...")


# 构建链: 数据处理 -> 提示词 -> LLM -> 输出解析
# RunnablePassthrough.assign()
chain = (
    RunnableLambda(sync_method)
    | prompt
    | llm
    | output_parser
)

# 异步调用时，通过字典形式传入 RunnableConfig
async def main():
    input_text = "hello world"

    config = RunnableConfig(
        max_concurrency=2, tags=["my_chain"],
        callbacks=CallbackManager([MonitoringCallback()]),
        configurable={"request_id": str(uuid.uuid4())}
    )
    result2 = await chain.ainvoke(input_text, config)
    print(result2)

    # 使用 .with_config() 方法预先配置链
    configured_chain = chain.with_config({"max_concurrency": 2})
    result3 = await configured_chain.ainvoke(input_text)
    print(result3)

# 运行
if __name__ == "__main__":
    asyncio.run(main())

