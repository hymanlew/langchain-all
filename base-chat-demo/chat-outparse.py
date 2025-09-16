"""
安装主要依赖库
pip install langchain -i https://pypi.org/simple
pip install langchain-openai -i https://pypi.org/simple
pip install fastapi uvicorn httpx -i https://pypi.org/simple
pip install "langserve[all]" -i https://pypi.org/simple

DeepSeek 完全遵循 OpenAI 的 API 接口规范，且在 SDK 调用方式和参数设计三个层面
都兼容。端点路径、请求参数完全一致。因此可以直接使用 openai 库调用 deepseek，只
需要修改大模型地址即可。

登录，并且获取 LangSmish 监控工具的 API key
https://smith.langchain.com/settings
"""
import os
from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

# 调用大语言模型, 创建模型
# model = ChatOpenAI(model='gpt-4-turbo')
model = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your_aliyun_api_key",
    model="qwen-turbo",
    temperature=0.6
)

# 2、准备prompt
# msg = [
#     SystemMessage(content='请将以下的内容翻译成日语'),
#     HumanMessage(content='你好，请问你要去哪里？')
# ]
# 需符合智谱AI的格式
msg = [
    {"role": "system", "content": "请将以下的内容翻译成日语"},
    {"role": "user", "content": "你好，请问你要去哪里？"}
]

# 此时的返回是包含全部信息（同 http 响应的结果），所以要解析出真正想要的结果
result = model.invoke(msg)
print(result)

# 3、创建返回的数据解析器
parser = StrOutputParser()
print(parser.invoke(result))

# 定义提示模板，其中列表中的每项都表示不同的角色
"""
通常包含不同角色的消息，比如系统消息、用户消息、助手消息等。
系统消息（system）用于设定对话的上下文或指示模型（指令）的行为，是后台指令，不会在前端展示给用户。
用户消息（user）则是具体要发送的内容。
用户只会看到自己输入的内容（user 消息）和模型的回复（assistant 消息）。

ChatPromptTemplate.from_messages([
    ('system', '你正在帮用户预订机票，请逐步询问出发地、目的地和日期'),
    ('assistant', '请问您要从哪个城市出发？'),  # 表示模型之前的回复，用于多轮对话的上下文传递。
    ('user', '{user_input}')                   # 用户当前输入
])
"""
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译成{language}'),
    ('user', "{text}")
])

# 4、得到链
chain = prompt_template | model | parser

# 5、 直接使用chain来调用
# print(chain.invoke(msg))
print(chain.invoke({'language': 'English', 'text': '我下午还有一节课，不能去打球了。'}))


# -------------------------------------------------------------

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")
response_schemas = [gift_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""
messages = (ChatPromptTemplate.from_template(template=review_template)
            .format_messages(text=customer_review,
                            format_instructions=format_instructions))
response = model(messages)
output_dict = output_parser.parse(response.content)
print(output_dict.get('delivery_days'))


# 把程序部署成 API 服务，用于自测，测试
app = FastAPI(title='DeepSeek 翻译服务', version='V1.0', description='基于 LangChain 0.2.x 的 DeepSeek 模型翻译服务')

add_routes(
    app,
    chain,
    path="/chainDemo",
)

"""
调用接口为 localhost:8000/chainDemo/invoke, POST
{
    "input":{'language': 'English', 'text': '我下午还有一节课，不能去打球了。'}
}

也可以自定义 client 进行访问（see fastapi-client.py）
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)