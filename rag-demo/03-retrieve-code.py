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


"""
Reranker 是检索增强生成(RAG)系统中的二次排序组件，用于对初步检索结果进行精细化重排序。与嵌入模型（生成向量）不同，Reranker 直接计算
查询与文档的相关性得分，通过深度语义理解提升排序质量。
是属于检索流程的中间件，位于初步检索（Milvus/TuGraph）与生成环节之间，形成"检索-重排-生成"管道

Rerank 模型分为基于 Transformer 的交叉编码器（如BERT）、LLM微调模型（如RankVicuna）（如） 

Rerank 模型就是指具体的算法组件实现，如：
- BGE-Reranker：基于RoBERTa微调的交叉编码器，高精度需求
- FlashRank-CPU：低延迟要求
- Cohere Rerank：轻量级API服务，多模态混合

# reranker-service.yaml
env:
- name: RERANKER_MODEL
  value: "BAAI/bge-reranker-large-fp16"  # 启用FP16优化
- name: FUSION_WEIGHTS
  value: '{"text":0.7, "graph":0.5, "image":0.4}'
resources:
  limits:
    gpu: 1  # 需要A10G以上GPU
    memory: 8Gi
"""
# # 企业级实现示例（结合 FlagEmbedding 库）
# from FlagEmbedding import FlagReranker
#
# # 初始化高性能 reranker（支持FP16加速）
# reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
# # 混合检索结果
# initial_results = vector_db.search(query, k=30) + graph_db.query(query, limit=10)
# # 重排序计算
# scored_pairs = [(query, doc) for doc in initial_results]
# rerank_scores = reranker.compute_score(scored_pairs, normalize=True)  # 归一化到0-1
# # 动态融合（加权分数）
# final_ranking = sorted(zip(initial_results, rerank_scores),
#                key=lambda x: x[1]*0.7 + x[0].original_score*0.3,  # 权重可调
#                reverse=True)

from FlagEmbedding import FlagReranker
from datetime import datetime, timedelta
import numpy as np

# -------------------- 1. 加载模型 --------------------
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)  # 可启用半精度加速

# -------------------- 2. 模拟召回数据 --------------------
query = "2024年诺贝尔文学奖得主"
candidate_docs = [
    {"id": 1, "content": "2024年诺贝尔文学奖授予挪威作家约恩·福瑟...", "publish_time": "2024-10-10"},
    {"id": 2, "content": "福瑟是挪威当代著名剧作家和小说家...", "publish_time": "2024-10-12"},
    {"id": 3, "content": "历届诺贝尔文学奖得主名单及作品介绍", "publish_time": "2023-05-01"},
    {"id": 4, "content": "2024年诺贝尔物理学奖揭晓", "publish_time": "2024-10-08"},
    # ... 更多文档
]

# -------------------- 3. 使用 BGE-Reranker 打分 --------------------
pairs = [[query, doc["content"]] for doc in candidate_docs]
scores = reranker.compute_score(pairs)  # 返回 list of float
# scores 可能为未归一化的 logits，例如 [-2.5, 3.2, 1.1, -1.8]

# 将分数添加到每个文档中
for i, doc in enumerate(candidate_docs):
    doc["rerank_score"] = scores[i]


# -------------------- 4. 归一化（如果需要与其他特征融合） --------------------
def min_max_norm(values):
    arr = np.array(values)
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v < 1e-9:
        return np.ones_like(arr)  # 所有值相同则返回1
    return (arr - min_v) / (max_v - min_v)


# 提取 rerank 分数并归一化
rerank_scores = [doc["rerank_score"] for doc in candidate_docs]
norm_rerank = min_max_norm(rerank_scores)

# 时效性得分：越新越高（例如：与当前日期相差天数，取负归一化）
# 原始分数 scores 可能是 [-2.5, 3.2, 1.1, -1.8]，而时效性分数 -days_diff 可能范围很大（如 [-5, -100]），直接加权会使时效性主导结果。归一化后两者都在 [0,1] 之间，加权才有意义
current_date = datetime.strptime("2024-10-15", "%Y-%m-%d")  # 假设当前日期
time_scores = []
for doc in candidate_docs:
    pub_date = datetime.strptime(doc["publish_time"], "%Y-%m-%d")
    days_diff = (current_date - pub_date).days
    # 时效性分：天数越少分越高，可用负天数归一化
    time_scores.append(-days_diff)  # 负天数，越新越大

norm_time = min_max_norm(time_scores)

# -------------------- 5. 加权融合 --------------------
weight_rerank = 0.7  # 相关性权重
weight_time = 0.3  # 时效性权重

final_scores = weight_rerank * norm_rerank + weight_time * norm_time

for i, doc in enumerate(candidate_docs):
    doc["final_score"] = final_scores[i]

# -------------------- 6. 排序并返回 --------------------
sorted_docs = sorted(candidate_docs, key=lambda x: x["final_score"], reverse=True)
# 若直接按分数排序，无需归一化
# sorted_docs = sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)
# 打印排序结果
for doc in sorted_docs:
    print(
        f"ID: {doc['id']}, Final: {doc['final_score']:.4f}, Rerank: {doc['rerank_score']:.4f}, Time: {doc['publish_time']}")
