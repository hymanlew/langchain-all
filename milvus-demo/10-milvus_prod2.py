from pymilvus import connections, Collection, utility
import threading
from queue import Queue
from contextlib import contextmanager

class MilvusConnectionPool:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_pool(*args, **kwargs)
        return cls._instance
    
    def _init_pool(self, max_connections=10, **conn_params):
        self.max_connections = max_connections
        self.conn_params = conn_params
        self.pool = Queue(max_connections)
        self._main_alias = conn_params.get("alias", "default")
        
        # 确保主连接存在
        self._ensure_connection(self._main_alias)
        
        # 创建多个连接实例
        for i in range(max_connections):
            alias = f"{self._main_alias}_{i}"
            self._ensure_connection(alias)
            self.pool.put(alias)
    
    def _ensure_connection(self, alias):
        """确保连接存在并已认证"""
        if alias not in connections.list_connections():
            connections.connect(alias=alias, **self.conn_params)
    
	# @property 是 Python 中的一个内置装饰器，它的主要作用是将一个类的方法转换为属性，让开发者可以用访问属性的语法来调用方法。
    @property
    def main_alias(self):
        """获取主连接别名（用于创建集合等管理操作）"""
        return self._main_alias
    
    def get_connection(self):
        """从池中获取一个连接别名"""
        try:
            return self.pool.get(block=True, timeout=5)
        except Exception:
            raise RuntimeError("Failed to get connection from pool")
    
    def release_connection(self, alias):
        """释放连接回池中"""
        self.pool.put(alias)
    
    @contextmanager
    def get_collection(self, collection_name):
        """获取集合的上下文管理器"""
        alias = self.get_connection()
        try:
            collection = Collection(collection_name, using=alias)
            
            # 检查集合是否已加载 - 兼容 2.5.10 的方式
            if utility.load_state(collection_name=collection_name, using=alias).value == 1:
                collection.load()    
            yield collection
        finally:
            self.release_connection(alias)
    
    @contextmanager
    def get_collection_for_insert(self, collection_name):
        """专门用于插入操作的集合获取"""
        alias = self.get_connection()
        try:
            yield Collection(collection_name, using=alias)
        finally:
            self.release_connection(alias)


from pymilvus import FieldSchema, CollectionSchema, DataType

def create_collection(collection_name, dimension=128):
    """创建新集合"""
    # 使用主连接进行管理操作
    main_alias = pool.main_alias
    
    # 1. 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]
    
    # 2. 创建模式
    schema = CollectionSchema(fields, description="Document embedding collection")
    
    # 3. 检查集合是否存在
    if utility.has_collection(collection_name, using=main_alias):
        utility.drop_collection(collection_name, using=main_alias)
    
    # 4. 创建集合
    collection = Collection(name=collection_name, schema=schema, using=main_alias)
    
    # 5. 创建索引（可选）
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    
    return collection
	

def create_index(collection_name, field_name, index_params):
    """为指定字段创建索引"""
    with connections.connect(using=pool.main_alias) as conn:
        collection = Collection(collection_name, using=pool.main_alias)
        
        # 检查索引是否已存在
        index_info = collection.indexes
        current_index = next((idx for idx in index_info if idx.field_name == field_name), None)
        
        if not current_index:
            # 创建新索引
            return collection.create_index(field_name, index_params)
        elif current_index.params != index_params:
            # 删除旧索引并创建新索引
            collection.drop_index()
            return collection.create_index(field_name, index_params)
			
def get_index_info(collection_name, field_name):
    """获取字段的索引信息"""
    with connections.connect(using=pool.main_alias) as conn:
        collection = Collection(collection_name, using=pool.main_alias)
        indexes = collection.indexes
        
        for index in indexes:
            if index.field_name == field_name:
                return {
                    "index_type": index.index_type,
                    "params": index.params,
                    "metric_type": index.metric_type
                }
        return None
		

	
def load_collection(collection_name):
    """显式加载集合到内存"""
    with connections.connect(using=pool.main_alias) as conn:
        # 获取集合加载状态
        if utility.load_state(collection_name=collection_name, using=alias).value == 1:
                collection.load()

def release_collection(collection_name):
    """释放集合内存资源"""
    for collection in self.load_collections:
        collection.release()
		
def get_collection_stats(collection_name):
    """获取集合统计信息"""
    with connections.connect(using=pool.main_alias) as conn:
        # 获取集合描述信息
        collection_info = utility.describe_collection(collection_name, using=pool.main_alias)
        
        # 获取集合实体数量
        num_entities = utility.num_entities(collection_name, using=pool.main_alias)
        
        # 获取加载状态
        load_state = utility.get_load_state(collection_name, using=pool.main_alias)
        
        return {
            "description": collection_info,
            "num_entities": num_entities,
            "load_state": load_state
        }
		
def create_partition(collection_name, partition_name):
    """创建新分区"""
    with connections.connect(using=pool.main_alias) as conn:
        collection = Collection(collection_name, using=pool.main_alias)
        return collection.create_partition(partition_name)

def drop_partition(collection_name, partition_name):
    """删除分区"""
    with connections.connect(using=pool.main_alias) as conn:
        collection = Collection(collection_name, using=pool.main_alias)
        return collection.drop_partition(partition_name)
		

		
def insert_data(collection_name, vectors, metadata_list, texts):
    """向集合中插入数据"""
    # 使用专门的插入上下文管理器
    with pool.get_collection_for_insert(collection_name) as collection:
        # 准备实体数据
        entities = [
            vectors,             # embedding 向量列表
            metadata_list,       # JSON 元数据列表
            texts                # 文本列表
        ]
        
        # 插入数据
        insert_result = collection.insert(entities)
        
        # 刷新确保数据立即可搜索
        collection.flush()
        
        return insert_result.primary_keys
    
    # 注意：插入后不需要加载集合，因为加载状态是独立的
	
def vector_search(collection_name, query_vector, top_k=10, search_params=None):
    """执行向量相似度搜索"""
    if search_params is None:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16}
        }
    
    with pool.get_collection(collection_name) as collection:
        # 定义搜索参数
        search_args = {
            "data": [query_vector],
            "anns_field": "embedding",
            "param": search_params,
            "limit": top_k,
            "output_fields": ["id", "text", "metadata"]  # 返回的字段
        }
        
        # 执行搜索
        results = collection.search(**search_args)
        
        # 处理结果 - 兼容 2.5.10 的返回格式
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata")
                })
        
        return formatted_results

def hybrid_search(collection_name, query_vector, filter_expr, top_k=10):
    """混合搜索：向量相似度 + 标量过滤"""
    with pool.get_collection(collection_name) as collection:
        # 定义搜索参数
        search_params = {
            "data": [query_vector],
            "anns_field": "embedding",
            "param": {"metric_type": "L2", "params": {"nprobe": 16}},
            "limit": top_k,
            "expr": filter_expr,  # 标量过滤表达式
            "output_fields": ["id", "text", "metadata"]
        }
        
        # 执行搜索
        results = collection.search(**search_params)
        
        # 处理结果
        return [{
            "id": hit.id,
            "distance": hit.distance,
            "text": hit.entity.get("text"),
            "metadata": hit.entity.get("metadata")
        } for hits in results for hit in hits]

		
def query_data(collection_name, expr, output_fields=None):
    """执行基于标量字段的查询"""
    if output_fields is None:
        output_fields = ["id", "text", "metadata"]
    
    with pool.get_collection(collection_name) as collection:
        # 执行查询
        query_result = collection.query(
            expr=expr,
            output_fields=output_fields
        )
        
        return query_result

		
def delete_data(collection_name, id_list):
    """删除指定ID的数据点"""
    expr = f"id in {id_list}"
    
    with pool.get_collection_for_insert(collection_name) as collection:
        # 执行删除
        delete_result = collection.delete(expr=expr)
        
        # 刷新确保删除生效
        collection.flush()
        
        return delete_result


# 初始化连接池
pool = MilvusConnectionPool(
    host="localhost", 
    port=19530,
    db_name="default",
    max_connections=20
)

# 创建新集合
create_collection("doc_embeddings", dimension=768)

# 插入数据
vectors = [[0.1]*768, [0.2]*768]  # 768维向量
metadata = [{"source": "doc1"}, {"source": "doc2"}]
texts = ["First document content", "Second document content"]
inserted_ids = insert_data("doc_embeddings", vectors, metadata, texts)

# 确保集合已加载
load_collection("doc_embeddings")

# 执行搜索
query_vector = [0.15]*768
results = vector_search("doc_embeddings", query_vector, top_k=5)

# 混合搜索
filter_expr = "metadata['source'] == 'doc1'"
results = hybrid_search("doc_embeddings", query_vector, filter_expr)

# 查询数据
query_result = query_data("doc_embeddings", "id in [1,2,3]")

# 删除数据
delete_data("doc_embeddings", [1,2])

# 释放资源
release_collection("doc_embeddings")

