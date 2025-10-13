"""
### 支持的搜索类型
- ANN 搜索：查找最接近查询向量的前 K 个向量。
- 过滤搜索：在指定的过滤条件下执行 ANN 搜索。
- 范围搜索：查找查询向量指定半径范围内的向量。
- 全文搜索：基于 BM25 的全文搜索。
- 混合搜索：基于多个向量场进行 ANN 搜索 + 以上搜索方式。
- Rerankers：根据附加标准或辅助算法调整搜索结果顺序，完善初始 ANN 搜索结果。
- 获取：根据主键检索数据。
- 查询：使田特定表达式检索数据。

**字段数据类型**
- FLOAT_VECTOR
- FLOAT16_VECTOR
- bfloat16_vector

**适用索引类型**
- 平面
- IVF_FLAT
- IVF_SQ8
- IVF_PQ（倒排乘积量化）
- GPU_IVF_FLAT
- GPU_IVF_PQ
- HNSW（分层图索引）
- DISKANI
"""
# 安装依赖，pymilvus 是一个 python SDK 集成了多个嵌入模型，rerankers 模型
# pip install pymilvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility,
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# 连接到本地 Milvus 服务
"""
connections 自动维护客户端，底层仍会创建 MilvusClient，但封装了连接池和状态管理

全文本搜索（BM25）是 Milvus 2.4 新增功能，MilvusClient 作为底层接口能更快支持新特性。
Collection 接口需要等待上层封装适配（通常晚1-2个版本），才能支持全文检索。

全文本搜索涉及分词器配置、BM25函数绑定等低级操作，MilvusClient 的 create_schema() 和 add_function() 更适合精细控制。
"""
connections.connect(
    alias="prod",  # 生产集群连接别名，服务连接名称
    host="10.0.0.1",
    port="19530",
    user="admin",
    password="Milvus"
)
connections.connect(
    alias="test",  # 测试集群连接别名
    host="10.0.0.2",
    port="19530",
    user="test",
    password="test"
)

# 多线程环境下，不同线程可安全使用不同别名的连接（底层自动隔离），线程安全
# 检查连接是否成功，并指定使用的连接，若不指定则默认是 default（创建连接时默认别名也是 default）
print(utility.get_server_version(using="prod"))

# 修改数据库密码
utility.reset_password('name', 'old-pass', 'new-pass')


# 创建集合（Collection）
# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),  # UUID主键
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),            # 向量维度需与模型匹配
    FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=65535),        # 文本内容
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),               # 原文档ID
    FieldSchema(name="chunk_index", dtype=DataType.INT32)                           # 切片序号
]

# 创建集合，并指定连接
collection_name = 'prod_collection'
schema = CollectionSchema(fields, description="Document chunks with embeddings")
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
prod_collection = Collection(
    name=collection_name,
    schema=schema,
    using="prod"
)
print(f"集合 {collection_name} 创建成功")


# 创建索引（加速搜索）
index_params = {
    "index_type": "IVF_FLAT",  # `IVF_FLAT`（中小规模）或 `HNSW`（快速查询）
    "metric_type": "L2",       # 相似度计算方式（L2 距离）
    "params": {"nlist": 128}   # 聚类中心数
}
# 为向量字段创建索引
prod_collection.create_index(
    field_name="embedding",
    index_params=index_params
)
print("索引创建完成")


# 插入数据
# 1. 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(long_text)

# 2. 向量化
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
embeddings = embedding_model.embed_documents(chunks)

# 3. 准备Milvus数据
"""
data 的每个字段列表共同描述多行数据，最终插入的行数由列表长度决定。
所有字段列表长度必须等于切片数量。
通过 doc_id 和 chunk_index 实现切片与原文档的关联，支撑后续检索和上下文还原。

data = [
    ["uuid1", "uuid2", "uuid3"],        # 主键ID（随机生成）
    [[0.1,0.2], [0.3,0.4], [0.5,0.6]],  # 向量（假设维度=2）
    ["text1", "text2", "text3"],        # 文本切片
    ["doc_123", "doc_123", "doc_123"],  # 原文档ID
    [0, 1, 2]                           # 切片序号
]
"""
data = [
    [str(uuid.uuid4()) for _ in chunks],  # 主键ID：每个切片一个唯一UUID（ID数量=列表长度=len(chunks)）
    embeddings,                           # 向量：外列表长度=切片数，内列表长度=向量维度（如768）的二维数组
    chunks,                               # 文本切片：["chunk1 text", "chunk2 text", ...]
    ["doc_id"] * len(chunks),             # 原文档ID：所有切片共享同一文档ID（列表长度=len(chunks)，列表内容一致，都为 doc_id）
    list(range(len(chunks)))              # 切片序号：生成从 0 开始的连续序号放入一个列表中，标记每个切片在原文档中的位置[0, 1, 2, ..., n-1]
]

# 4. 插入Milvus
# 每条记录（行）对应一个独立的向量 + 可选的标量字段（如文本片段、元数据）。
# 查询时返回的最小单位是行，若所有切片合并为一行，将无法精准检索到具体片段。
insert_result = prod_collection.insert(data)
prod_collection.flush()  # 确保数据持久化
print(f"已插入 {insert_result.insert_count} 条数据")


# 条件查询
prod_results = prod_collection.query(expr="id > 0")

# 修改 schema 添加字段
fields.append(FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=20))
schema = CollectionSchema(fields)
test_collection = Collection.create(collection_name, schema)

# 搜索时过滤类别
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = test_collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=5,
    expr='category == "fiction"',  # 过滤条件
    output_fields=["text", "category"]
)



# 相似性搜索
collection.load()

# 随机生成一个查询向量
query_vector = [[random.random() for _ in range(128)]]

# 执行搜索
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # 搜索参数
results = prod_collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=5,                # 返回前 5 个结果
    output_fields=["text"]  # 返回的元数据字段
)

# 解析结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, 距离: {hit.distance}, 文本: {hit.entity.get('text')}")
		


# 更新数据（需指定主键）
test_collection = Collection("book_collection")
expr = "id in [1, 2, 3]"  # 要更新的主键
new_texts = ["Updated text 1", "Updated text 2", "Updated text 3"]
test_collection.delete(expr)  # 先删除旧数据
test_collection.insert([[1, 2, 3], new_vectors, new_texts])  # 重新插入

# 删除数据
test_collection.delete(expr="id in [4,5,6]")
test_collection.flush()
