# langgraph 用于协调多个组件（如 Chain、Agent、Tool）的协作
# 是用于编排 Agents、工具、其他函数之间的协作
# pip install langgraph -i https://pypi.org/simple
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGSMITH_API_KEY"] = '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'
os.environ["TAVILY_API_KEY"] = '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'

model = ChatOpenAI(model='gpt-4-turbo')

# 没有使用任何代理或工具的情况下, 无法获得正确的结果
# result = model.invoke([HumanMessage(content='北京天气怎么样？')])
# print(result)

# LangChain 内置了一个工具，使用国外的 Tavily 搜索引擎作为工具。
# 需安装 langchain-community 和 tavily-python 包，并通过环境变量或代码配置 API Key
# max_results: 只返回两个结果，否则会搜索很多与主题相关的网站及结果
search = TavilySearchResults(max_results=2)
# print(search.invoke('北京的天气怎么样？'))

# 让模型绑定工具
tools = [search]
model_with_tools = model.bind_tools(tools)

# 此时模型会自动推理：如果自己可以返回结果，则就不去调用工具。否则就调用工具去完成用户的答案
# 并且这里只是选择去调用工具，但工具是不执行的。工具执行操作是由 agent 代理调用工具的
resp = model_with_tools.invoke([HumanMessage(content='中国的首都是哪个城市？')])
print(f'Model_Result_Content: {resp.content}')
print(f'Tools_Result_Content: {resp.tool_calls}')

resp2 = model_with_tools.invoke([HumanMessage(content='北京天气怎么样？')])
print(f'Model_Result_Content: {resp2.content}')
print(f'Tools_Result_Content: {resp2.tool_calls}')

#  创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
resp = agent_executor.invoke({'messages': [HumanMessage(content='中国的首都是哪个城市？')]})
print(resp['messages'])

resp2 = agent_executor.invoke({'messages': [HumanMessage(content='北京天气怎么样？')]})
print(resp2['messages'])
print(resp2['messages'][2].content)


#----------------------------------------------

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, Tool
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import ChatOpenAI

"""
搜索引擎集成方案，国内企业通常采用以下两种方式：
方案一：使用支持中文搜索的国内 API 平台（如百度搜索 API、幂简集成 API HUB），通过 HTTP 请求封装为 LangChain 工具。
方案二：自建爬虫服务，针对企业内部文档或特定网站，构建定制化爬虫工具，结合向量数据库实现本地化检索。
"""
class SearchInput(BaseModel):
    query: str = Field(description="搜索文本")


@tool("search-tool", args_schema=SearchInput, return_direct=False)
def cn_search(query: str) -> str:
    return query


@tool
def cn_search(query: str) -> str:
    """使用国内搜索引擎 API 进行实时搜索"""
    description = '使用国内搜索引擎 API 进行实时搜索'
    url = "https://api.example.cn/search/v1"  # 替换为实际 API 地址
    params = {
        "key": "your_search_api_key",
        "q": query,
        "num": 5  # 返回结果数量
    }
    response = requests.get(url, params=params)
    return response.json().get("results", [])

model = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key="your_zhipu_api_key",
    model="glm-4"
)

"""
模型的输出质量与提供给它的输入信息的质量和结构密切相关。正是这个提示指导大模型遵循 ReAct 框架。
ReAct 的 prompt 是动态变化的（模仿 few-shot learning）。
即形成有效的观察一思考一行动一再观察的循环，直到最终正确的结果。

Thought（思考）:...
Action（行动）:...
Observation（观察）:...
"""
#导人 LangChain Hub, 从Hub 中获取 ReAct 的提示
from langchain import hub
prompt = hub.pull("hwchasel7/react")
# prompt = ChatPromptTemplate.from_messages([('user', "系统提示词...")])
print(prompt)

"""
以下是翻译后的 prompt 内容：
尽你所能用中文回答以下问题。如果能力不够，你可以访问以下工具:
{tools}
请使用以下格式回答
问题:输入的问题你必须回答
思考:你每次都应该思考接下来怎么做
行动:要采取的行动，应该是[{tool_names}]中的一个
行动输入:行动的输入
观察:行动的结果
.......(这个 思考/行动/行动输入/观察 的过程可以重复N次)
思考:我现在知道最终答案了
最终答案:原始输入问题的最终答案
开始!
问题:{input}
思考:{agent_scratchpad}
"""

# 工具列表
# tools = [cn_search]
# 加载 langchain 封装好的工具
# tools = load_tools(["bingSearch"], llm=model)
# Description 非常重要，它就是用于判断是否应该调用这个工具的依据。
tools =[
    Tool(
        name="Search",
        func=cn_search,
        description="当大模型没有相关知识时，用于搜索知识"
    ),
    cn_search,
]

# Agents 适合单代理任务，动态调用工具完成简单决策
# langgraph agent 适合多代理协作或复杂流程编排，支持循环、分支、并行
# 创建 Agent（以智谱 GLM-4 为例）
memorySaver = InMemorySaver()
agent = create_react_agent(
    model,
    tools,
    prompt,
    checkpointer=memorySaver
)

"""
AgentExecutor 是 Agent 的运行环境，它首先调用大模型，接收并观察结果，然后执行大模型所选择的操作，同时也负责处理多种复杂情况，包括Agent选择了不存在的工具的情况、
工具出错的情况、Agent 产生无法解析成 Function Caling 格式的情况，以及在 Agent 决策和工具调用期间进行日志记录。

agent_executor 实际就是一个 Chain(链)类。AgentExecutor 类就继承自 Chain 类。
"""
# 执行搜索任务
agent_executor = AgentExecutor(
    agent=agent, tools=tools, memory=memorySaver, verbose=True,
    handle_parsing_errors='自定义的异常信息'
)

def get_session_history(user_id: str, session_id: str) -> BaseChatMessageHistory:
    if (user_id, session_id) not in store:
        store[(user_id, session_id)] = ChatMessageHistory()
    return store[(user_id, session_id)]


do_message = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key='input',  # 每次聊天时要发送的 msg 的 key
    history_message_key='history',
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户的唯一标识符",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="对话的唯一标识符",
            default="",
            is_shared=True,
        ),
    ],
)

config = {'configurable': {'user_id':'111', 'session_id': 'user1'}}
response = do_message.invoke(
    {"input": "2025 年新能源汽车补贴政策有哪些变化？"},
    config=config,
)
print(response["output"])


# -------------------------------------------------------

import openai
from typing import List, Dict

# 1. 定义工具函数（实际业务逻辑）
def get_weather(location: str, unit: str = "celsius") -> Dict:
    """获取指定城市的天气信息"""
    # 这里替换为真实API调用
    return {
        "location": location,
        "temperature": "25",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }

# 2. 描述函数结构（大模型需要知道如何调用）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取某地的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 3. 与大模型交互
def run_conversation(user_query: str):
    # 首次调用（模型决定是否调用函数）
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_query}],
        tools=tools,
        tool_choice="auto"
    )
    
    # 4. 处理函数调用
    tool_calls = response.choices[0].message.get("tool_calls", [])
    if tool_calls:
        available_functions = {"get_weather": get_weather}  # 注册可用函数
        messages = [{"role": "user", "content": user_query}]
        
        # 执行函数调用
        for call in tool_calls:
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            function_response = available_functions[function_name](**function_args)
            
            # 将结果返回给大模型
            messages.append({
                "role": "tool",
                "content": str(function_response),
                "tool_call_id": call.id
            })
        
        # 获取最终回答
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return second_response.choices[0].message.content
    
    return response.choices[0].message.content

# 测试
print(run_conversation("上海今天天气怎么样？"))

