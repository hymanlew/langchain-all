"""
文档类查询处理 @see 02-rag-retrieve, 04-pdf-load

双列存储（索引列 +内容向量列）：
- 内容向量列：存储原始文本通过嵌入模型（如BGE）生成的高维向量
- 索引列：通过TF-IDF、KeyBERT或LLM从内容向量中提取关键词，建立倒排索引

# 企业级Schema设计（以 Milvus为例）
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="keyword_index", dtype=DataType.VARCHAR, max_length=255)
    ]
)

先通过关键词在索引列快速筛选候选集（减少90%+计算量），再对候选集进行向量相似度精排
graph LR
A[用户查询] --> B(关键词提取)
B --> C{索引列匹配?}
C -->|是| D[召回候选集]
C -->|否| E[全量向量搜索]
D --> F[向量精排]
"""
# 语义分类器（OpenAI微调版）
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import pipeline

class SemanticClassifier:
    def __init__(self, model="gpt-4-turbo"):
        self.model = model
        self.label_map = {
            "title": ["文档标题", "报告名称", "文章题目"],
            "content": ["具体内容", "技术细节", "数据描述"]
        }
        # 使用LLM增强关键词提取（企业级实现）
        self.kw_extractor = pipeline("text2text-generation", model="BAAI/keybert-base")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def classify(self, query: str) -> list:
        """使用OpenAI进行意图分类"""



        base_kws = self.kw_extractor(query)  # 基础关键词

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": f"判断用户问题类型，输出'title'或'content':\n已知标题关键词:{self.label_map['title']}"
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.3
        )
        return list(set(base_kws + response.choices[0].message.content.split(",")))


# 数据召回服务
from typing import List, Dict
from milvus import MilvusClient
from typing import Union

# 或者直接问文档标题，那么怎么召回到文档内容？
# 如果用户的问题是随意问的，此时要如何分析语义到文档标题上？
class HybridRetrievalSystem:
    def __init__(self, milvus_host: str):
        self.classifier = SemanticClassifier()
        self.milvus = MilvusClient(uri=milvus_host)
        self.collection_name = "document_vectors"

    def process_query(self, query: str) -> Dict:
        """完整处理流程"""
        # 1. 意图分类
        query_list = self.classifier.classify(query)
        query = str(query_list)

        # 2. 内容路径
        vector_results = self.milvus.search(
            collection_name=self.collection_name,
            query_embeddings=[self._embed_query(query)],
            limit=5
        )

        return {
            "type": "content",
            "contents": [v["content"] for v in vector_results[0]],
            "titles": vector_results
        }

    def _embed_query(self, text: str) -> List[float]:
        """生成查询向量（实际项目替换为真实模型）"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model.encode(text).tolist()

