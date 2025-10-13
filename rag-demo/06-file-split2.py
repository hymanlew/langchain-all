from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import torch
import uuid

embeddings = HuggingFaceEmbeddings(
    model_name= "models/text/bge-small-zh-v1.5",
    model_kwargs={"device": 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={
        "normalize_embeddings": True,  # 归一化
        "batch_size": 32  # 根据内存调整
    }
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",   # 使用百分位阈值 或 "standard_deviation"（标准差）
    breakpoint_threshold_amount=0.8,          # 取相似度最低的20%（即低于第80百分位的相似度）作为分割点
    add_start_index=False                      # 可选，是否记录每个分块在原文中的起始位置
)

async def get_text_vectors(query: str):
    return embeddings.embed_query(query)

async def get_train_vectors(documents, metas):
    documents = text_splitter.create_documents(documents, metadatas=metas)
    text_contents = [
        f"{{'question': {doc.metadata.get('question', '')}, 'answer': {doc.page_content}, 'index': {doc.metadata.get('index', '')}}}"
        for doc in documents]
    vectors = embeddings.embed_documents(text_contents)
    data = [
        {"vector": vectors[idx], "content": doc}
        for idx, doc in enumerate(text_contents)
    ]
    return data

async def analyzer_train_datas(self, datas: list):
    """拼装训练数据"""
    qa_mapping = {}
    documents = []
    metas = []

    for i, item in enumerate(datas):
        # 为每个文档（问题-答案对）创建一个元数据，包含原始问题和索引
        qa_id = uuid.uuid4().hex  # 生成一个唯一ID
        # qa_mapping[qa_id] = item  # 存储映射到关系数据库
        documents.append(item['answer'])
        metas.append({
            "index": qa_id,
            "question": item["question"]
        })

    return await get_train_vectors(documents, metas)

datas = [
    {
        "question": "问题1",
        "answer": "答案1"
    },
    {
        "question": "问题2",
        "answemr": "答案2"
    }
]
# datas = await analyzer_train_datas(datas)
# times = datetime.now(timezone.utc).isoformat()
# for dt in datas:
#     dt["user_id"] = userid
#     dt["avatar_id"] = avatarid
#     dt["update_time"] = times
#
# ids = await Milvus.insert(datas)

# ----------------------------------------------------------------------------------------------------------------

from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.storage import SQLStore
from langchain_core.documents import Document
import sqlite3  # 或者使用其他数据库驱动

# 1. 定义切分器
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# 2. 初始化存储
embeddings = OpenAIEmbeddings()
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={},	# 配置 Milvus 连接参数, username, pwd
    collection_name="parent_documents",  # 指定集合名称
    drop_old=True  # 如果集合已存在，是否重新创建（根据你的需求调整）
)

# 使用持久化存储 存储父块，如 db
# 方法1：使用 SQLite（内置支持）
store = SQLStore("sqlite:///parent_docs.db")
# 方法2：如果需要使用真正的 MySQL
# store = SQLDocStore("mysql+pymysql://username:password@localhost:3306/database_name")

# 3. 创建检索器
# 将子文档的数据向量化存储到 Milvus 中，父文档的原始内容（较大块）存储在 doc store 中
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 4. 添加文档
docs = [Document(page_content="这是测试文档")]
retriever.add_documents(docs)

# 5. 检索
retrieved_docs = retriever.get_relevant_documents("你的问题")

# -----------------------------------------------------------------------------------------------------------

"""
推荐使用 Parent Document Retriever 的情况：
- 文档长度较长：原始文档超过1000字符
- 上下文依赖强：答案需要前后文才能准确理解
- 对答案质量要求高：需要生成连贯、完整的回答。生成时上下文更完整（由于返回父块）。
- 适用于长文档，能够平衡检索和生成的需求。
- 资源充足：需要维护两个存储（向量库和文档存储）。

推荐使用 将问题和答案组合成一个文本进行向量化 的情况：
- 文档较短：问答对相对独立，长度适中。更适用于QA对的形式。
- 资源有限：希望保持架构简单
- 问答对结构清晰：问题和答案有明确的对应关系
"""
