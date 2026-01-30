from datetime import datetime, timedelta
import faiss
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# 定义嵌入模型
embeddings_model = OpenAIEmbeddings()
# 初始化向量存储
index = faiss.IndexFlatL2(1536)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})
# 创建高衰减率的TimeWeightedVectorStoreRetriever
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.5, k=1
)
# 设置文档的上次访问时间为昨天
yesterday = datetime.now() - timedelta(days=1)
# 添加文档
retriever.add_documents(
    [Document(page_content="hello world", metadata={"last_accessed_at": yesterday})]
)
# 添加没有指定访问时间的文档，默认当前时间作为其最后访问时间
retriever.add_documents([Document(page_content="hello foo")])
# 在检索"hello world"时，由于设置了较高的衰减率，"hello foo"（因其较新的"访问"时间）可能会首先返回
results = retriever.get_relevant_documents("hello world")
# 输出检索结果
for doc in results:
    print(f"Document Content: {doc.page_content}")

from langchain_core.utils import mock_now
import datetime
# 模拟未来的时间（使用较短的时间间隔）
with mock_now(datetime.datetime(2028, 8, 8, 12, 0)):  # 模拟几小时后的时间
    print(retriever.get_relevant_documents("hello world"))

