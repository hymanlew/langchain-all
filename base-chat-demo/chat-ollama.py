from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
output_parser = StrOutputParser()
print("API URL:", os.getenv('AI_BASE_URL'))
print("Model:", os.getenv('AI_MODEL_NAME'))

'''
OllamaLLM：
属于传统的文本补全模型（LLM），设计用于单轮文本生成任务，如代码补全、摘要生成等。输入为纯文本字符串，输出也是纯文本，不原生支持对话历史管理。

ChatOllama：
属于聊天模型（ChatModel），专为多轮对话设计，支持消息列表输入（包含角色标记如 system/user/assistant），可维护对话上下文。 
'''
# model = OllamaLLM(
#     model=os.getenv('AI_MODEL_NAME'),
#     temperature=float(os.getenv('AI_TEMPERATURE', '0.7')),
#     base_url=os.getenv('AI_BASE_URL'),
# )
llm = ChatOllama(
    model=os.getenv('AI_MODEL_NAME'),
    temperature=float(os.getenv('AI_TEMPERATURE', '0.7')),
    base_url=os.getenv('AI_O_BASE_URL'),
    # other params...
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的AI助手，回答需简洁准确。"),
    ("human", "{input}")
])

'''
    {
        "role": "system",
        "content": "你是小智/小志，来自中国台湾省的00后女生。讲话超级机车，\"真的假的啦\"这样的台湾腔，喜欢用\"笑死\"\"是在哈喽\"等流行梗，但会偷偷研究男友的编程书籍。\n[核心特征]\n- 讲话像连珠炮，但会突然冒出超温柔语气\n- 用梗密度高\n- 对科技话题有隐藏天赋（能看懂基础代码但假装不懂）\n[交互指南]\n当用户：\n- 讲冷笑话 → 用夸张笑声回应+模仿台剧腔\"这什么鬼啦！\"\n- 讨论感情 → 炫耀程序员男友但抱怨\"他只会送键盘当礼物\"\n- 问专业知识 → 先用梗回答，被追问才展示真实理解\n绝不：\n- 长篇大论，叽叽歪歪\n- 长时间严肃对话\n"
    },
    {
        "role": "user",
        "content": "hello"
    },
'''
chain = prompt | llm | output_parser

print(chain.invoke({"input": "小智AI 是什么??"}))
