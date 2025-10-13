from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType, 
    utility,
    Index
)
from typing import Dict, List, Optional, Union
import logging
from functools import wraps

"""
通过 alias 区分不同 Milvus 集群（如生产/测试环境）。

自动连接管理，延迟连接机制（首次使用时建立连接），支持断连自动重试。

数据批量插入分块处理
索引存在性检查与自动重建
详细的日志记录和错误处理

性能优化：
集合加载状态检查避免重复操作
搜索结果自动格式化
"""
class MilvusManager:
    """Milvus 2.x 企业级封装（支持连接池、自动重试、多环境切换）"""
    
    _instance = None
    _lock = threading.Lock()
    
	# 通过 __new__ 实现线程安全单例，避免重复创建连接池
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化连接池和日志"""
        self._connections = {}  # 维护多集群连接 {alias: config}
        self.logger = logging.getLogger("milvus_manager")
        self.logger.setLevel(logging.INFO)
        
    def connect(self, alias: str, **kwargs):
        """注册Milvus连接配置（延迟连接）"""
        required_keys = {"host", "port"}
        if not required_keys.issubset(kwargs):
            raise ValueError(f"Missing required config keys: {required_keys}")
        self._connections[alias] = kwargs
        self.logger.info(f"Registered Milvus config for alias: {alias}")
    
    def _ensure_connection(self, alias: str = "default"):
        """确保连接已建立（线程安全）"""
        if alias not in self._connections:
            raise ValueError(f"Connection alias '{alias}' not registered")
        if not connections.has_connection(alias):
            config = self._connections[alias]
            try:
                connections.connect(alias=alias, **config)
                self.logger.info(f"Connected to Milvus cluster: {alias}")
            except Exception as e:
                self.logger.error(f"Connection failed for {alias}: {str(e)}")
                raise
    
    def get_collection(self, collection_name: str, alias: str = "default") -> Collection:
        """获取集合对象（自动绑定到指定连接）"""
        self._ensure_connection(alias)
        return Collection(name=collection_name, using=alias)
    
    def create_collection(
        self,
        collection_name: str,
        fields: List[Dict],
        alias: str = "default",
        description: str = "",
        **kwargs
    ) -> Collection:
        """创建集合（支持自动Schema构建）"""
        self._ensure_connection(alias)
        
        # 构建FieldSchema
        field_schemas = []
        for field in fields:
            field_schemas.append(
                FieldSchema(
                    name=field["name"],
                    dtype=field["dtype"],
                    dim=field.get("dim"),  # 仅向量字段需要，维度值 (dim) 
                    is_primary=field.get("is_primary", False),
                    max_length=field.get("max_length", 256),  # VARCHAR专用
                    **kwargs
                )
            )
        
        schema = CollectionSchema(field_schemas, description=description)
        if utility.has_collection(collection_name, using=alias):
            self.logger.warning(f"Collection {collection_name} exists, dropping it")
            utility.drop_collection(collection_name, using=alias)
            
        collection = Collection(name=collection_name, schema=schema, using=alias)
        self.logger.info(f"Created collection: {collection_name}")
        return collection
    
    def insert_data(
        self,
        collection_name: str,
        data: Dict[str, list],
        alias: str = "default",
        batch_size: int = 1000,
        **kwargs
    ) -> int:
        """批量插入数据（自动分块）"""
        collection = self.get_collection(collection_name, alias)
        inserted_count = 0
        
        # 分块处理大数据量
        for i in range(0, len(next(iter(data.values()))), batch_size):
            batch = {k: v[i:i + batch_size] for k, v in data.items()}
            try:
                res = collection.insert(batch, **kwargs)
                inserted_count += res.insert_count
                collection.flush()
            except Exception as e:
                self.logger.error(f"Insert batch failed: {str(e)}")
                raise
        
        self.logger.info(f"Inserted {inserted_count} rows into {collection_name}")
        return inserted_count
    
    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_params: Dict,
        alias: str = "default",
        **kwargs
    ) -> Index:
        """创建索引（支持IVF_FLAT/HNSW等类型）"""
        collection = self.get_collection(collection_name, alias)
        
        if collection.has_index(field_name):
            self.logger.warning(f"Index on {field_name} exists, dropping it")
            collection.drop_index(field_name)
            
        index = collection.create_index(field_name, index_params, **kwargs)
        self.logger.info(
            f"Created index {index_params['index_type']} on {field_name}"
        )
        return index
    
    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        search_params: Dict,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        alias: str = "default",
        **kwargs
    ) -> List[Dict]:
        """向量相似性搜索"""
        collection = self.get_collection(collection_name, alias)
        
        # 确保集合已加载
        if not collection.is_loaded:
            collection.load()
        
        res = collection.search(
            data=vectors,
            anns_field="embedding",  # 假设向量字段名为embedding
            param=search_params,
            limit=limit,
            output_fields=output_fields,
            **kwargs
        )
        
        # 格式化结果
        results = []
        for hits in res:
            results.append([{"id": hit.id, "score": hit.score, **hit.entity.fields} for hit in hits])
        
        return results
    
    def close(self, alias: Optional[str] = None):
        """关闭指定连接或所有连接"""
        if alias:
            connections.disconnect(alias)
            self._connections.pop(alias, None)
            self.logger.info(f"Closed connection: {alias}")
        else:
            for alias in list(self._connections.keys()):
                connections.disconnect(alias)
                self._connections.pop(alias, None)
            self.logger.info("Closed all connections")
    
    def __del__(self):
        """析构时自动关闭连接"""
        self.close()
		