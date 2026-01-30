# 导入所需的库
from langchain_cohere import CohereRerank
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

# 创建BM25检索器
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # 设置返回前3个结果

# 设置Cohere重排序器
reranker = CohereRerank(model="rerank-multilingual-v2.0")

# 执行查询和重排
query = "山西有哪些著名的旅游景点？"
# 先获取初始检索结果
initial_docs = retriever.invoke(query)
# 使用重排序器对结果进行重排
reranked_docs = reranker.compress_documents(documents=initial_docs, query=query)

# 打印重排结果
print(f"查询：{query}\n")
print("重排序后的结果：")
for i, doc in enumerate(reranked_docs, 1):
    print(f"{i}. {doc.page_content}")
