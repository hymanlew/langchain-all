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


"""
HuggingFaceEmbeddings 类在 LangChain 框架中封装了 sentence-transformers 库的功能
- 直接使用 sentence-transformers：更灵活，更多底层参数配置，可直接访问所有原生功能。但需要自己处理集成到框架中
- 使用 LangChain 的 HuggingFaceEmbeddings：更适合集成到 LangChain 生态系统中，只支持双向编码器（即SentenceTransformer模型）

因此LangChain的HuggingFaceEmbeddings只适用于生成文档和查询的嵌入向量（用于检索等任务），而交叉编码器通常用于对文本对进行相关性评分
（例如，在重排序任务中）。所以，如果需要同时使用两种类型的模型，或者在同一个流程中使用交叉编码器进行重排序，那么需要自定义一个类来支持。
"""
from typing import Optional, Dict, Any, List
import numpy as np
from langchain_core.embeddings import Embeddings
# Pydantic 增强的 dataclass（dataclasses），支持验证
from pydantic.dataclasses import dataclass
from dataclasses import field

"""
 它集成了两种类型的模型：双向编码器（Bi-encoder）和交叉编码器（CrossEncoder）
 无论使用哪一种编码器，加载的模型必须是要支持的

 双向编码器 (Bi-encoder)：
 - 用于标准嵌入任务，如文档检索、语义搜索。适用于大规模检索（先编码，后计算相似度）
 - 它将每个句子独立地编码为一个向量，然后通过向量之间的相似度（如余弦相似度）来衡量句子间的相似度。

 交叉编码器 (Cross-encoder)：
 - 用于句子分类、相关性评分、文本对匹配任务。适用于精排、重排序（直接计算文本对的相关性）
 - 同时编码两个句子，并输出一个分数（如相关性分数）。

 因此，交叉编码器模型和普通的嵌入模型是两种不同的模型架构，不能混用。
"""
@dataclass
class HuggingfaceEmbeddingsDemo(Embeddings):
    """Model name to use."""
    model_name: str = 'DEFAULT_MODEL_NAME'
    """Path to store models.
    Can be alLso set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    cache_folder: Optional[str] = None
    """Keyword arguments to pass to the model."""
    model_kwargs: Dict[str, Any] = field(default_factory = dict)
    # 编码时的参数字典，同样使用 field 设置默认工厂函数返回空字典
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import sentence_transformers
            from transformers import AutoConfig
            # 导入序列分类模型映射名称
            from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
        except ImportError as exc:
            raise ImportError("Could not import sentence_transformers python package."
                              "PLease install it with 'pip install sentence-transformers."
                              ) from exc

        config = AutoConfig.from_pretrained(self.model_name)

        # 检查模型是否为交叉编码器（cross）：
        # 1. 获取序列分类模型的所有架构名称列表
        # 2. 使用 numpy 的 intersect1d 计算交集
        # 3. 检查 config.architectures 中是否包含序列分类模型的架构
        # 4. 将结果转换为布尔值
        self.is_cross_encoder = bool(
            np.intersect1d(
                # 序列分类模型架构名称列表
                List(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
                # 当前模型的架构列表
                config.architectures,
            )
        )

        if self.is_cross_encoder:
            # 交叉编码器：同时编码两个文本，计算它们的相关性分数
            self.model = sentence_transformers.CrossEncoder(self.model_name, **self.model_kwargs,)
        else:
            # 双向编码器：分别编码文本，然后计算相似度
            self.model = sentence_transformers.SentenceTransformer(
                self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
            )

        # ensure outputs are tensors
        if "convert_to_tensor" not in self.encode_kwargs:
            self.encode_kwargs["convert_to_tensor"] = True

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[float]:
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        from torch import Tensor

        assert isinstance(self.model, SentenceTransformer), "Model is not of the type Bi-encoder"
        embeddings = self.model.encode(
            # 是否归一化嵌入向量（总是为True）
            texts, normalize_embeddings=True, **self.encode_kwargs
        )

        assert isinstance(embeddings, Tensor)
        return embeddings.tolist()

    """预测文本对的相关性分数"""
    def predict(self, texts: List[List[str]]) -> List[float]:
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch import Tensor

        assert isinstance(self.model, CrossEncoder), "Model is not of the type CrossEncoder"

        # 使用交叉编码器预测文本对的相关性分数
        predictions = self.model.predict(texts, **self.encode_kwargs)

        assert isinstance(predictions, Tensor)
        return predictions.tolist()

