'''
pip install langchain langchain-community langchain-core -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain-openai python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install requests langchain_ollama -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ujson pymilvus -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

如果安装包很多，文件中的包太多，为了避免下载中断后重复下载，可使用以下命令：
conda activate pro
conda install --file requirements.txt --yes
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

但要注意 conda 安装的包可能不全。
'''
from operator import itemgetter
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RouterRunnable, RunnableBranch
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
llm = ChatOpenAI(
    model=os.getenv('AI_MODEL_NAME'),
    temperature=float(os.getenv('AI_TEMPERATURE', '0.7')),
    max_retries=2,
    api_key=os.getenv('AI_API_KEY'),
    base_url=os.getenv('AI_BASE_URL'),
    # organization="...",
    # other params...
)

template = """问题: {question}\n详细回答:"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的AI助手，回答需简洁准确。"),
    # ("human", "{input}")
    ("user", template)
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


# Create a chat prompt template from a human template string
template = """Answer the question based only on the following context:  
{context}  

Question: {question}

Answer in the following language: {language}  
"""
prompt = ChatPromptTemplate.from_template(template)

# RunnablePassthrough 接收用户问题，顺序扩展执行，再传递给 prompt 和 model。
chain = RunnablePassthrough.assign(query=sql_chain).assign(result=itemgetter('query'))| prompt

# 字典在管道 | 中会被自动转换为 RunnableParallel，每个分支是并行的，它们会同时接收相同的输入
chain2 = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

chain3 = {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "language": itemgetter("language")
        } | prompt | model | StrOutputParser()

# 使用 RunnableParallel 并行执行
parallel_analysis = RunnableParallel({
    "chain": chain,
    "summary": chain2,
    "original_text": RunnablePassthrough() # 同时保留原文
})
result = parallel_analysis.invoke({})


prompt1 = ChatPromptTemplate.from_template("generate a random color")
prompt2 = ChatPromptTemplate.from_template("what is a fruit of color: {color}")
prompt3 = ChatPromptTemplate.from_template("what is countries flag that has the color: {color}")
prompt4 = ChatPromptTemplate.from_template("What is the color of {fruit} and {country}")
chain1 = prompt1 | model | StrOutputParser()
chain2 = RunnablePassthrough.assign(color=chain1) | {
    "fruit": prompt2 | model | StrOutputParser(),
    "country": prompt3 | model | StrOutputParser(),
} | prompt4


RouterRunnable
RunnableBranch