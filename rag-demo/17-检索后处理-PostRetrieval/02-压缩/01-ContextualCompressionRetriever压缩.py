# 导入所需的库
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()
# 准备示例文档
documents = [
    Document(
        page_content="五台山是中国四大佛教名山之一，以文殊菩萨道场闻名。",
        metadata={"source": "山西旅游指南"}
    ),
    Document(
        page_content="云冈石窟是中国三大石窟之一，以精美的佛教雕塑著称。",
        metadata={"source": "山西旅游指南"}
    ),
    Document(
        page_content="平遥古城是中国保存最完整的古代县城之一，被列为世界文化遗产。",
        metadata={"source": "山西旅游指南"}
    )
]

"""
压缩什么？
压缩检索到的文档内容，而不是压缩数据存储
将多个相关文档合并、去重、提取关键信息
减少输入给LLM的token数量，同时保留核心信息
"""
# 创建BM25检索器
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # 设置返回前3个结果
# 设置Cohere重排序器
compressor = CohereRerank(model="rerank-multilingual-v2.0")
# 创建ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
# 执行查询、重排和压缩
query = "山西有哪些著名的旅游景点？"
compressed_docs = compression_retriever.invoke(query)
# 输出压缩结果
print(f"查询：{query}\n")
print("重排并压缩后的结果：")
for i, doc in enumerate(compressed_docs, 1):
    print(f"{i}. {doc.page_content}")

