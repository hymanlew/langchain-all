'''
python == 3.9.13
pip install langchain langchain-community langchain-core -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain-openai python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install requests langchain_ollama -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ujson pymilvus -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

如果安装包很多，文件中的包太多，为了避免下载中断后重复下载，可使用以下命令：
conda activate xiaozhi-pro
conda install --file requirements.txt --yes
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

但要注意 conda 安装的包可能不全。
'''
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 加载.env文件中的环境变量
load_dotenv()

print()
print("API URL:", os.getenv('AI_BASE_URL'))
print("Model:", os.getenv('AI_MODEL_NAME'))

os.environ["OPENAI_API_KEY"] = "xxxx"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGSMITH_API_KEY"] = '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'

llm = ChatOpenAI(
    model=os.getenv('AI_MODEL_NAME'),
    temperature=float(os.getenv('AI_TEMPERATURE', '0.7')),
    max_retries=2,
    api_key=os.getenv('AI_API_KEY'),
    base_url=os.getenv('AI_BASE_URL'),
    # organization="...",
    # other params...
)

template = """问题: {question}
详细回答:"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的AI助手，回答需简洁准确。"),
    # ("user", "{input}")
    ("user", template)
])

# 创建链（模板 → 模型 → 解析器）
chain = prompt | llm | StrOutputParser()

# 调用示例
# response = chain.invoke([{"input": "解释Transformer的注意力机制"}])
response = chain.invoke([{"question": "解释Transformer的注意力机制"}])
print(response)


# --------------------------------------------------------------------------

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
customer_style = """American English \
in a calm and respectful tone
"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
prompt_template = ChatPromptTemplate.from_template(template_string)

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
response = chain.invoke(customer_messages)
print(response)


