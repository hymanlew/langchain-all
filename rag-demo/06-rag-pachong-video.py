import datetime
import os
from typing import Optional, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic.v1 import BaseModel, Field

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

model = ChatOpenAI(model='gpt-4-turbo')
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# 一些YouTube的视频连接
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]

# document的数组
docs = []
for url in urls:
    # 一个Youtube的视频字幕数据，对应一个document
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())

print(len(docs))
print(docs[0])

# 给doc添加额外的元数据： 视频发布的年份
for doc in docs:
    doc.metadata['publish_year'] = int(
        datetime.datetime.strptime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y')
    )

# 第一个视频的字幕内容
print(docs[0].metadata)
print(docs[0].page_content[:500])

# 根据多个doc构建向量数据库
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
split_doc = text_splitter.split_documents(docs)

# 存放向量数据库的目录，即是在当前所属目录下会新建一个文件夹存放
persist_dir = 'chroma_data_dir'

# 把向量数据库持久化到磁盘
vectorstore = Chroma.from_documents(split_doc, embeddings, persist_directory=persist_dir)
# 加载磁盘中的向量数据库
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 测试向量数据库的相似检索
result = vectorstore.similarity_search_with_score('how do I build a RAG agent')
print(result[0])
print(result[0][0].metadata['publish_year'])


system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


# 使用 pydantic 进行数据管理（定义，序列化等），或验证（检索）的支持
# 定义一个数据模型，按照真实的字幕数据的数据结构来构建，以便于精细化检索。类似于数据库表对应的实体类结构
# 比如检索第三层的数据，检索一个数据它存在于三层和五层的结构里
class SearchEntity(BaseModel):
    # 根据内容的相似性和发布年份，进行检索
    # 使用 pydantic 的 Field 字段进行定义包装
    content: str = Field(None, description='Similarity search query applied to video transcripts.')
    # Optional 表示可有可无的，即这里表示可以不传值，不按这个字段检索
    publish_year: Optional[int] = Field(None, description='Year video was published')


# 模型在这里是查询转换器（prompt 中指定了），而非知识库。它不直接回答问题，而是理解用户问题后生成检索的查询条件
# 因为是按自定义的结构检索，和结构输出，所以要使用 with_structured_output 进行结构化输出
search_chain = {'question': RunnablePassthrough()} | prompt | model.with_structured_output(SearchEntity)
resp1 = search_chain.invoke('how do I build a RAG agent?')
print(resp1)
resp2 = search_chain.invoke('videos on RAG published in 2023')
print(resp2)


def retrieval(search: SearchEntity) -> List[Document]:
    year = None
    if search.publish_year:
        # 定义 publish_year 匹配检索，即要匹配输出结构中的，指定字段的值
        # $eq 是 Chroma 向量数据库的固定语法
        year = {'publish_year': {"$eq": search.publish_year}}

    return vectorstore.similarity_search(search.content, filter=year)


new_chain = search_chain | retrieval

result = new_chain.invoke('videos on RAG published in 2023')
result = new_chain.invoke('RAG tutorial')
print([(doc.metadata['title'], doc.metadata['publish_year']) for doc in result])
