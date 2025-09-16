import os

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import BaichuanTextEmbeddings, HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['BAICHUAN_API_KEY'] = 'sk-732b2b80be7bd800cb3a1dbc330722b4'
loader = TextLoader('file/state_of_the_union.txt', encoding='utf8')
documents = loader.load()

"""
所有 LLM 都有上下文窗口限制（如 GPT-4 的128K tokens），精确计算token数量可避免因输入过长导致的请求失败。
langChain 默认使用 tiktoken（OpenAI系模型的分词器），但若对接其他模型（如LLaMA、Claude）时，它们的中文分词与 tiktoken 差异极大（如"你好"可能被拆成['你', '好']而非单个token）。
所以当切换不同后端模型时，需要手动指定分词器：
"""
# 加载自定义tokenizer（以LLaMA为例）
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 定义token计数函数（关键修改）
def token_counter(text: str) -> int:
    return len(llama_tokenizer.encode(text, add_special_tokens=False))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # 基于token数量而非字符数
    chunk_overlap=20,  # 建议设置重叠（如20%的chunk_size）
    #length_function=len,
    length_function=token_counter, # 原代码使用 len 函数（按字符计数），现改为通过tokenizer精确计算token数量
    is_separator_regex=True, # 启用正则分隔符
    separators=[
        "\n\n",
        "\n",
		"(?<!\\.\\d)\\.(?!\\d)",  # 排除小数点
        ".",
        "?",
        "!",
        "。",
        "！",
        "？",
        ",",
        "，",
        " "
    ]
)
docs = text_splitter.split_documents(documents)
print('=======', len(docs))

# 创建向量数据库, 并存储到本地（会自动执行）
model_name = 'glm-4-0520'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = BaichuanTextEmbeddings()
embedding = HuggingFaceBgeEmbeddings(
	model_name=model_name,
	model_kwargs=model_kwargs,
	encode_kwargs=encode_kwargs,
	query_instruction="为文本生成向量表示用于文本检索"
)

vectorStore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
print("向量数据已保存至 ./chroma_db")

# 从本地加载已有的向量存储
vectorStore = Chroma(
    persist_directory="./chroma_db",      # 指定之前保存的路径
    embedding_function=embeddings        # 需使用相同的嵌入模型
)

# 测试一下向量数据库
query = '今年长三角铁路春游运输共经历多少天？'
# docs_and_score = vectorStore.similarity_search_with_score(query)
# for doc, score in docs_and_score:
#     print('-------------------------')
#     print('Score: ', score)
#     print("Content:  ", doc.page_content)

# 和大语言模型整合
# 创建模型
model = ChatOpenAI(
    model='glm-4-0520',
    api_key='0884a4262379e6b9e98d08be606f2192.TOaCwXTLNYo1GlRM',
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)

template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

#  把检索器和用户输入的问题，结合得到检索结果
retriever = vectorStore.as_retriever()
start_retriever = RunnableParallel({'context': retriever, 'question': RunnablePassthrough()})

# 创建长链
output_parser = StrOutputParser()
chain = start_retriever | prompt | model | output_parser

res = chain.invoke(query)
print(res)

