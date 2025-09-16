"""
GraphRAG本身是技术框架，但实际落地需要依赖具体的工具链支持。其核心思想包括：

- 三元组抽取：通过 LLM 服务（语言模型 + embdding 模型）实现从文本中提取结构化知识（实体-关系-实体），并写入图数据库。
- 子图检索召回：基于图数据库查询与问题相关的局部子图。实现查询的关键词提取和泛化（大小写、别称、同义词等），并基于关键词实现子图遍历（DFS/BFS），搜索N跳以内的局部子图。
- 上下文生成：将局部子图数据格式化为文本，作为上下文和问题一起提交给大模型处理。

核心组件
- AI框架：DB-GPT（国产开源，支持分布式Graph RAG）
- 知识图谱引擎：OpenSPG（蚂蚁开源，支持中文语义增强）
- 图数据库：TuGraph（万亿边规模，LDBC性能冠军）
- 向量模型：bge-large-zh-v1.5（中文语义向量标杆）

协作流程示例：
1. 知识图谱构建阶段：OpenSPG 从文本中抽取三元组，构建图谱关系，并存入 TuGraph；
2. 检索阶段：DB-GPT调用TuGraph进行子图查询；
3. 生成阶段：DB-GPT将子图转化为文本，输入大模型生成答案。
"""
# 配置示例（DB-GPT框架）
from dbgpt.storage.graph import TuGraphStore, OpenSPGKnowledgeGraph
from dbgpt.rag import GraphRAGBuilder
from openspg.nn4k import NN4KConfig, TripletExtractor

# **混合存储设计**
# 1. 初始化NN4K配置（需提前部署）
NN4KConfig.load("nn4k_config.yaml")  # 包含zhipu-api密钥等信息

graph_store = TuGraphStore(
    uri="bolt://localhost:7687",
    user="tugraph",
    password="tugraph@123"
)

kg = OpenSPGKnowledgeGraph(
    graph_store=graph_store,
    spg_schema="finance_schema.yaml",  # 中文领域schema
    zh_enhancement=True  # 启用中文优化
)

# 中文数据处理
import re
import jieba
from typing import List
from langchain_text_splitters import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# 中文文本清洗
def clean_zh_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)  # 去HTML
    text = re.sub(r'\s+', ' ', text)     # 合并空格
    text = re.sub(r'[【】、；，（）“”‘’]', ' ', text)  # 中文标点处理
    return text.strip()

# 语义分块（适配中文）
# text2vec-large-chinese 通用分词处理
# BAAI/bge-large-zh-v1.5 支持领域术语增强(如金融、医疗)
zh_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
zh_splitter = SemanticChunker(
    embeddings=zh_embedding,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

def chunk_zh_text(text: str) -> List[str]:
    cleaned = clean_zh_text(text)
    # 先按段落粗分
    paragraphs = [p for p in cleaned.split('\n') if p]
    # 语义细分成300字左右的块
    chunks = []
    for para in paragraphs:
        if len(para) > 500:
            chunks.extend(zh_splitter.split_text(para))
        else:
            chunks.append(para)
    return chunks

# **知识图谱构建（中文三元组抽取）**
from langchain_core.documents import Document
from dbgpt.transformers import TripletExtractor

# 中文优化提示词模板
ZH_TRIPLET_TEMPLATE = """请从中文文本中抽取尽可能多的(主语, 谓语, 宾语)三元组，注意：
1. 保留中文原词，不要翻译
2. 处理中文特有的省略和指代
3. 识别中文复合关系如"阿里巴巴的创始人马云"应拆解为：
   (阿里巴巴, 创始人, 马云)

文本：{text}
三元组："""

# TripletExtractor 通过 NN4K 框架隐式调用智谱模型，可通过修改 NN4K 配置文件(nn4k_config.yaml)切换模型版本
extractor = TripletExtractor(
    llm_api="zhipu-api",  # 使用智谱AI
    prompt_template=ZH_TRIPLET_TEMPLATE
)


def build_zh_knowledge_graph(docs: List[Document]):
    for doc in docs:
        # OpenSPG 实际图存储处理流程（NN4K 框架内部）
        triplets = extractor.extract(text)  # 提取原始三元组
        enriched_triplets = kg.semantic_enhancement(triplets)  # 语义增强
        # 将三元组转成成图谱，并写入图数据库
        for subj, rel, obj in enriched_triplets:
            kg.insert_triplet(
                subject=standardize_entity(subj),  # 实体标准化
                relation=map_relation(rel),  # 关系映射
                object=standardize_entity(obj),  # 对象标准化
                source=source,
                confidence=calculate_confidence()  # 置信度评估
            )


"""
Milvus 替代 TuGraph 的方案对比
功能        TuGraph实现	            Milvus替代方案
三元组存储   原生图存储	                将三元组编码为向量 + 元数据存储 
关系查询	   Cypher/Gremlin查询语言    向量相似度搜索 + 后过滤 
语义推理	   基于规则的推理引擎	        向量空间中的线性代数运算 
"""
# 纯Milvus实现文档保存（具有父子关系，语义连续性 @see 04-pdf-load）
def _store_structured(self, triples):
    """将结构化数据编码为向量"""
    # 三元组文本化
    texts = [f"{s} {p} {o}" for s, p, o in triples]

    # 使用SPG文本编码
    vectors = self.spg.encode(texts)

    # 存储元数据
    for idx, triple in enumerate(triples):
        self.milvus.insert_metadata({
            "subject": triple[0],
            "predicate": triple[1],
            "object": triple[2],
            "vector_id": idx
        })

    return vectors[0] if vectors else None


# **混合检索实现**
from dbgpt.retrievers import HybridRetriever

class ZhHybridRetriever:
    def __init__(self, kg, vector_store):
        self.graph_retriever = kg.as_retriever(
            search_type="subgraph",
            search_kwargs={"depth": 2}
        )
        self.vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

    def retrieve(self, query: str):
        # 向量检索
        vector_results = self.vector_retriever.get_relevant_documents(query)
        # 图谱检索
        cypher_query = f"""
        MATCH (n)-[r]->(m)
        WHERE any(label IN labels(n) WHERE n.`zh_name` CONTAINS '{query}' 
               OR m.`zh_name` CONTAINS '{query}')
        RETURN n, r, m LIMIT 10
        """
        graph_results = self.graph_retriever.query(cypher_query)

        return self._fusion_results(vector_results, graph_results)
