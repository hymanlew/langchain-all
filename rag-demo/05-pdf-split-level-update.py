import uuid
from typing import List, Dict
from pymilvus import Collection, FieldSchema, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""
在长文档处理中，**存储父级ID**和**动态上下文注入**两种方案各有优劣，具体选择需结合业务场景。以下是深度对比和企业级混合方案实现：

### **一、方案对比与选型建议**
| **维度**         | **存储父级ID方案**                          | **动态上下文注入方案**                  |
|------------------|--------------------------------------------|----------------------------------------|
| **适用场景**     | 需要严格维护文档层级（如法律合同、技术手册） | 问答系统等需要上下文连贯性的场景        |
| **查询复杂度**   | 需多次查询重建层级                          | 单次查询即可获得上下文                  |
| **存储开销**     | 额外存储父ID/路径字段（约10%空间增长）      | 文本冗余存储（约20-30%空间增长）        |
| **更新成本**     | 修改父节点需更新所有子节点                  | 独立更新，但需重新计算上下文            |
| **典型应用**     | 文档管理系统、知识图谱                      | RAG、聊天机器人                        |

**推荐选择逻辑**：
- **优先父级ID**：文档结构复杂且需频繁按层级检索
- **优先上下文注入**：侧重语义连贯性且文档结构扁平
- **混合方案**：企业级推荐两者结合（见下文）


### **二、企业级混合方案实现**
#### **1. 核心设计**
- **存储父级ID**：维护基础层级关系
- **动态上下文**：在文本切片中注入前序关键句
- **元数据增强**：添加`context_window`和`hierarchy_path`字段

双模式互补：
用parent_id维护严谨层级
用context_window保证语义连贯性
"""
class HybridDocumentProcessor:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "(?<=。)"]
        )

    def _extract_key_sentences(self, text: str, window_size=2) -> str:
        """提取前序关键句作为上下文（基于语义重要性）"""
        sentences = text.split('。')
        return '。'.join(sentences[-window_size:])

	"""
	提取前序关键句作为上下文方案：
	1，LLM摘要生成（如GPT-4）的云API成本确实较高（约1.5元/千token），但可通过以下方式优化：
	   本地轻量化模型：使用7B-20B级开源模型（如DeepSeek-7B、GLM-4-Flash）私有化部署，硬件成本可控制在1万-5万元。
	   缓存机制：对高频查询结果缓存，减少重复计算。
	from langchain_core.prompts import ChatPromptTemplate
	from langchain_community.chat_models import ChatOpenAI

	def extract_key_sentences_llm(text: str, llm_model) -> str:
		"""使用LLM生成前序内容摘要"""
		prompt = ChatPromptTemplate.from_template(
			"请用1-2句话概括以下文本的核心内容，保留关键实体和结论：\n\n{text}"
		)
		chain = prompt | llm_model
		return chain.invoke({"text": text}).content
	
	
	2，TF-IDF方案：适用于非实时场景，成本仅为LLM的1/10。
	from sklearn.feature_extraction.text import TfidfVectorizer
	import numpy as np

	def extract_key_sentences(text: str, window_size=3) -> str:
		"""通过TF-IDF权重提取前序关键句"""
		sentences = [s.strip() for s in text.split('。') if s.strip()]
		if not sentences:
			return ""
		
		# 计算句子TF-IDF权重
		vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
		tfidf_matrix = vectorizer.fit_transform(sentences)
		sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
		
		# 选择权重最高的句子
		top_indices = np.argsort(-sentence_scores)[:window_size]
		return '。'.join([sentences[i] for i in sorted(top_indices)])
	
	
	2，BERT方案：适用于非实时场景，成本仅为LLM的1/10。
	from sklearn.cluster import KMeans
	import numpy as np

	def extract_key_sentences_bert(text: str, embedding_model, window_size=2) -> str:
		"""通过BERT嵌入聚类提取中心句"""
		sentences = [s.strip() for s in text.split('。') if s.strip()]
		if len(sentences) <= window_size:
			return '。'.join(sentences)
		
		# 获取句子嵌入
		embeddings = embedding_model.embed_documents(sentences)
		
		# K-Means聚类找中心点
		kmeans = KMeans(n_clusters=window_size, random_state=42)
		kmeans.fit(embeddings)
		closest_indices = [np.argmin(np.linalg.norm(embeddings - center, axis=1)) 
						  for center in kmeans.cluster_centers_]
		
		return '。'.join([sentences[i] for i in sorted(closest_indices)])
		
		
	3，规则引擎+关键词提取：对结构化文本（如合同条款）效率更高。
	
	4. Spacy Text Splitter 方法
	Spacy 是一个用于执行自然语言处理（NLP）各种任务的库。它具有文本拆分器功能，能够在进行文本分割的同时，保留分割结果的上下文信息。
	import spacy
	input_text = "文本分块是自然语言处理（NLP）中的一项关键技术，其作用是将较长的文本切
	割成更小、更易于处理的片段。这种分割通常是基于单词的词性和语法结构，例如将文本拆分
	为名词短语、动词短语或其他语义单位。这样做有助于更高效地从文本中提取关键信息。"
	
	nlp = spacy.load( "zh_core_web_sm" )
	doc = nlp(input_text)
	for s in doc.sents:
		print (s)
	
	输出如下 >>>
	[
	'文本分块是自然语言处理（NLP）中的一项关键技术，其作用是将较长的文本切割成更
	小、更易于处理的片段。',
	"这种分割通常是基于单词的词性和语法结构，例如将文本拆分为名词短语、动词短语或其
	他语义单位。",
	"这样做有助于更高效地从文本中提取关键信息。"
	]


	4，集成到动态上下文注入
	class EnhancedContextProcessor:
		def __init__(self, strategy="tfidf", embedding_model=None, llm_model=None):
			self.strategy = strategy
			self.embedding_model = embedding_model
			self.llm_model = llm_model

		def _extract_key_sentences(self, text: str) -> str:
			if not text.strip():
				return ""
				
			if self.strategy == "tfidf":
				return extract_key_sentences(text)
			elif self.strategy == "bert":
				return extract_key_sentences_bert(text, self.embedding_model)
			elif self.strategy == "llm":
				return extract_key_sentences_llm(text, self.llm_model)
			else:
				raise ValueError(f"Unsupported strategy: {self.strategy}")

		def add_context(self, full_text: str, sub_chunk: str) -> str:
			context_window = full_text[:max(0, full_text.find(sub_chunk))]
			key_info = self._extract_key_sentences(context_window)
			return f"[CONTEXT: {key_info}] {sub_chunk}"
	"""
    def process_document(self, full_text: str) -> List[Dict]:
        # 第一轮：按结构分块（模拟从Markdown/XML解析）
        structural_chunks = self._parse_structure(full_text)
        
        # 第二轮：对每个结构块再分片
        all_records = []
        for chunk in structural_chunks:
            sub_chunks = self.text_splitter.split_text(chunk["text"])
            
            for i, sub_chunk in enumerate(sub_chunks):
                # 动态上下文注入
                context = self._extract_key_sentences(chunk["text"][:max(0, chunk["text"].find(sub_chunk))])
                enhanced_text = f"[CONTEXT: {context}] {sub_chunk}"
                
                all_records.append({
                    "id": str(uuid.uuid4()),
                    "text": sub_chunk,
                    "enhanced_text": enhanced_text,  # 带上下文的版本
                    "parent_id": chunk["parent_id"],
                    "level": chunk["level"],
                    "context_window": context,
                    "hierarchy_path": f"{chunk['hierarchy_path']}/{chunk['id']}" if chunk["parent_id"] else f"/{chunk['id']}"
                })
        
        return all_records

    def _parse_structure(self, text: str) -> List[Dict]:
        """模拟文档结构解析（实际项目替换为真实解析逻辑）"""
        # 示例：假设解析出章、节两级结构
        return [
            {"id": "c1", "text": "第一章 总则...", "parent_id": None, "level": 1},
            {"id": "s1_1", "text": "1.1 范围...", "parent_id": "c1", "level": 2},
            {"id": "s1_2", "text": "1.2 定义...", "parent_id": "c1", "level": 2}
        ]

# 初始化处理器
processor = HybridDocumentProcessor(embedding_model=HuggingFaceEmbeddings())

# 批量处理文档并生成记录
from concurrent.futures import ThreadPoolExecutor   
def batch_process(documents: List[str], workers=4):
   with ThreadPoolExecutor(max_workers=workers) as executor:
	   return list(executor.map(processor.process_document, documents))
records = batch_process("长文档全文内容...documents")


# 3. Milvus Schema配置
schema = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),       # 原始分片
    FieldSchema(name="enhanced_text", dtype=DataType.VARCHAR, max_length=65535), # 带上下文的文本
    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=36),
    FieldSchema(name="level", dtype=DataType.INT8),
    FieldSchema(name="context_window", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="hierarchy_path", dtype=DataType.VARCHAR, max_length=512)
], description="Hybrid document chunks")

# 创建集合
collection = Collection("hybrid_docs", schema)

# 层级路径前缀索引
collection.create_index(
   field_name="hierarchy_path",
   index_params={"index_type": "Trie"} # 针对字符串字段的索引
)
# 向量索引优化
collection.create_index(
   field_name="embedding",
   index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
)
   
# 插入数据前向量化
embeddings = processor.embedding_model.embed_documents([r["enhanced_text"] for r in records])
data = [
    [r["id"] for r in records],
    embeddings,
    [r["text"] for r in records],
    [r["enhanced_text"] for r in records],
    [r["parent_id"] for r in records],
    [r["level"] for r in records],
    [r["context_window"] for r in records],
    [r["hierarchy_path"] for r in records]
]
collection.insert(data)


# 4. 混合查询示例
def hybrid_search(collection: Collection, query: str, top_k: int = 5):
    # 向量搜索带上下文的文本
    query_embedding = processor.embedding_model.embed_query(query)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 128}},
        limit=top_k,
        output_fields=["text", "enhanced_text", "hierarchy_path", "parent_id"]
    )
    
    # 结果增强：关联父节点文本
    enriched_results = []
    for hit in results[0]:
        record = {
            "text": hit.entity.get("text"),
            "context": hit.entity.get("enhanced_text"),
            "score": hit.score
        }
        
        # 按需加载父节点内容（企业级建议用缓存优化）
        parent_id = hit.entity.get("parent_id")
        if parent_id:
            parent = collection.query(
                expr=f"id == '{parent_id}'",
                output_fields=["text"]
            )
            record["parent_text"] = parent[0]["text"] if parent else None
        
        enriched_results.append(record)
    
    return enriched_results




# 缓存层设计：
from redis import Redis
class HierarchyCache:
   def __init__(self):
	   self.redis = Redis(host='cache.enterprise.com', port=6379)
   
   def get_parent(self, parent_id: str) -> Dict:
	   if cached := self.redis.get(f"parent:{parent_id}"):
		   return json.loads(cached)
	   # ... 否则查询数据库 ...


# 灵活查询：
# 按层级精确查询
collection.query(expr='hierarchy_path like "/c1/s1_1%"')

# 语义搜索增强上下文
hybrid_search(collection, "如何定义项目范围?")

