import threading
import atexit
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Literal
from pymilvus.orm import utility
from pymilvus import connections, DataType, FieldSchema, Collection, CollectionSchema, FunctionType, Function, \
    AnnSearchRequest, RRFRanker
from config.config_loader import load_config
from config.logger import setup_logging

TAG = __name__
config = load_config()
logger = setup_logging()


class MilvusClientPool:
    """Milvus 连接池"""
    def __init__(self):
        self.host = str(config.get("Milvus", {}).get("host", "localhost"))
        self.port = int(config.get("Milvus", {}).get("port", 19530))
        self.db_name = config.get("Milvus", {}).get("db_name", "default")
        self.collection_names = set(config.get("Milvus", {}).get("collection_name", []))
        self.pool_size = config.get("Milvus", {}).get("pool_size", 0)
        self.timeout = config.get("Milvus", {}).get("timeout", 10)
        self.aliases = [f"conn_{i}" for i in range(self.pool_size)]
        self.lock = threading.Lock()
        self.load_collections = []

        # 初始化所有连接
        for alias in self.aliases:
            try:
                connections.connect(
                    uri=f"http://{self.host}:{self.port}",
                    user='',
                    password='',
                    db_name=self.db_name,
                    alias=alias,
                    timeout=self.timeout,
                    keep_alive=True,
                )
            except Exception as e:
                logger.bind(tag=TAG).error(f"create_connect error - {alias}: {e}")
                pass

        self.available = self.aliases.copy()
        for collect in self.collection_names:
            if collect == "device_face":
                self._init_identify_collections(collection_name=collect)
            if collect == "avatar_voice":
                self._init_avoice_collections(collection_name=collect)
            if collect == "avatar_train":
                self._init_atrain_collections(collection_name=collect)
        # 应用退出时调用
        atexit.register(self._shutdown)

    @contextmanager
    def get_collection(self, collection_name, timeout=None):
        """获取 MilvusClient 的上下文管理器"""
        if timeout is None: timeout = self.timeout
        alias = self._acquire(timeout)
        try:
            collection = Collection(collection_name, using=alias)
            if utility.load_state(collection_name=collection_name, using=alias).value == 1:
                collection.load()
                self.load_collections.append(collection)
            yield collection
        except Exception as e:
            logger.bind(tag=TAG).error(f"get_collection error: {e}")
            raise e
        finally:
            self._release(alias)

    def _acquire(self, timeout):
        """获取 MilvusClient 链接，带超时 second """
        alias = None
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if self.available:
                    alias = self.available.pop()
                    break
            time.sleep(0.1)
        if not alias:
            raise TimeoutError(f"获取连接超时 - {timeout}s")
        return alias

    def _release(self, alias):
        with self.lock:
            self.available.append(alias)

    def _init_identify_collections(self, collection_name, timeout=None):
        """初始化集合"""
        if timeout is None: timeout = self.timeout
        alias = self._acquire(timeout)
        if not utility.has_collection(collection_name, alias):
            logger.bind(tag=TAG).info(f"create_collection info: {collection_name}")
            fields = [
                FieldSchema(name="id", auto_id=True, dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="device_id", dtype=DataType.VARCHAR, max_length=52),
                FieldSchema(name="client_id", dtype=DataType.VARCHAR, max_length=52),
                FieldSchema(name="user_json", dtype=DataType.JSON, nullable=True, max_length=512),
                FieldSchema(name="face_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="audio_vector", dtype=DataType.FLOAT_VECTOR, dim=192),
                FieldSchema(name="is_delete", dtype=DataType.BOOL, default_value=False),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)

            collection = Collection(name=collection_name, schema=schema, using=alias, properties={"num_shards":1, "consistency_level":"Bounded"})
            collection.create_index(field_name="device_id", index_params={"index_type": "AUTOINDEX"})
            collection.create_index(field_name="client_id", index_params={"index_type": "AUTOINDEX"})
            collection.create_index(field_name="face_vector", index_params={"index_type":"HNSW", "metric_type":"COSINE", "params":{"M": 48, "efConstruction": 360}})
            collection.create_index(field_name="audio_vector", index_params={"index_type":"HNSW", "metric_type":"COSINE", "params":{"M": 24, "efConstruction": 180}})
            logger.bind(tag=TAG).info(f"初始化集合成功 - {collection_name}")

    def _init_avoice_collections(self, collection_name, timeout=None):
        """初始化集合"""
        if timeout is None: timeout = self.timeout
        alias = self._acquire(timeout)
        if not utility.has_collection(collection_name, alias):
            logger.bind(tag=TAG).info(f"create_collection info: {collection_name}")
            fields = [
                FieldSchema(name="id", auto_id=True, dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=52),
                FieldSchema(name="audio_text", dtype=DataType.VARCHAR, nullable=True, max_length=52),
                FieldSchema(name="audio_file", dtype=DataType.VARCHAR, max_length=52),
                FieldSchema(name="audio_vector", dtype=DataType.FLOAT_VECTOR, dim=192),
                FieldSchema(name="is_delete", dtype=DataType.BOOL, default_value=False),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)

            collection = Collection(name=collection_name, schema=schema, using=alias,
                                    properties={"num_shards": 1, "consistency_level": "Bounded"})
            collection.create_index(field_name="user_id", index_params={"index_type": "AUTOINDEX"})
            collection.create_index(field_name="audio_vector",
                                    index_params={"index_type": "HNSW", "metric_type": "COSINE",
                                                  "params": {"M": 24, "efConstruction": 180}})
            logger.bind(tag=TAG).info(f"初始化集合成功 - {collection_name}")

    def _init_atrain_collections(self, collection_name, timeout=None):
        """初始化集合"""
        if timeout is None: timeout = self.timeout
        alias = self._acquire(timeout)
        if not utility.has_collection(collection_name, alias):
            logger.bind(tag=TAG).info(f"create_collection info: {collection_name}")
            fields = [
                FieldSchema(name="id", auto_id=True, dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=52),
                FieldSchema(name="avatar_id", dtype=DataType.VARCHAR, max_length=52, is_partition_key=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True,
                            analyzer_params={"type": "chinese"}),
                FieldSchema(name="content_bm25", dtype=DataType.SPARSE_FLOAT_VECTOR),  # 稀疏向量
                FieldSchema(name="update_time", dtype=DataType.VARCHAR, max_length=35),
                FieldSchema(name="is_delete", dtype=DataType.BOOL, default_value=False),
            ]
            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=["content"],
                output_field_names=["content_bm25"],
                function_type=FunctionType.BM25,
            )
            schema = CollectionSchema(fields=fields, functions=[bm25_function], enable_dynamic_field=True)
            collection = Collection(name=collection_name, schema=schema, using=alias, num_partitions=128,
                                    properties={"num_shards": 1, "consistency_level": "Bounded"})
            collection.create_index(field_name="user_id", index_params={"index_type": "AUTOINDEX"})
            collection.create_index(field_name="avatar_id", index_params={"index_type": "AUTOINDEX"})
            collection.create_index(field_name="vector",
                                    index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 4096}})
            collection.create_index(field_name="content_bm25",
                                    index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {"inverted_index_algo": "DAAT_MAXSCORE"}})
            logger.bind(tag=TAG).info(f"初始化集合成功 - {collection_name}")

    async def insert(self, dtype: Literal["image", "audio"], datas: list[dict]):
        """ 检查当前 client id 是否已有数据，有则做更新 """
        user = await self.search_by_user("client_id", datas[0].get("client_id"))
        with self.get_collection("device_face") as collection:
            if not user:
                collection.insert(datas)
                collection.flush()
                msg = "插入身份数据成功"
            else:
                if dtype == "image":
                    user["face_vector"] = datas[0]["face_vector"]
                else:
                    user["audio_vector"] = datas[0]["audio_vector"]
                user["user_json"] = datas[0]["user_json"]
                collection.upsert(user)
                msg = "更新身份数据成功"
            logger.bind(tag=TAG).info(f"{msg} {datas[0]['device_id']} - {datas[0]['client_id']}")
        return "OK"

    async def search_by_face(self, device_id, query_vector):
        print("============ 查询 face ================")
        try:
            with self.get_collection("device_face") as client:
                res = client.search(
                    anns_field="face_vector",
                    data=[query_vector],
                    expr=f"device_id == '{device_id}'",
                    limit=3,
                    param={"metric_type": "COSINE", "params": {"ef": 220}},
                    output_fields=["score", "device_id", "client_id", "user_json"],
                )
                sorted_res = sorted(res[0], key=lambda x: x.score, reverse=True)
                if len(sorted_res) > 0 and sorted_res[0].score >= 0.65:
                    return sorted_res[0].user_json
                return dict()
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询 milvus face 异常")
            raise e

    async def search_by_audio(self, device_id, query_vector):
        print("============ 查询 audio ================")
        try:
            with self.get_collection("device_face") as client:
                res = client.search(
                    anns_field="audio_vector",
                    data=[query_vector],
                    expr=f"device_id == '{device_id}'",
                    limit=3,
                    param={"metric_type": "COSINE", "params": {"ef": 180}},
                    output_fields=["device_id", "client_id", "user_json"],
                )
                sorted_res = sorted(res[0], key=lambda x: x.score, reverse=True)
                if len(sorted_res) > 0 and sorted_res[0].score >= 0.55:
                    return sorted_res[0].user_json
                return dict()
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询 milvus audio 异常")
            raise e

    async def search_by_user(self, dtype: Literal["device_id", "client_id"], data):
        try:
            with self.get_collection("device_face") as client:
                res = client.query(
                    expr=f"{dtype} == '{data}'",
                    output_fields=["id", "device_id", "client_id", "user_json", "face_vector", "audio_vector"],
                    limit=1
                )
                if res:
                    return res[0]
                else:
                    return None
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询 milvus audio 异常")
            raise e

    async def insert_avoice(self, datas: list[dict]):
        """ 插入语音文件 """
        try:
            with self.get_collection("avatar_voice") as collection:
                results = collection.insert(datas)
                msg = "新增分身语音数据成功"
                logger.bind(tag=TAG).info(f"{msg} user_id - {datas[0]['user_id']}")
                return results.primary_keys[0]
        except Exception as e:
            logger.bind(tag=TAG).error(f"插入 milvus voice 异常 user_id - {datas[0]['user_id']}")
            raise e

    async def search_by_voice(self, data):
        try:
            with self.get_collection("avatar_voice") as client:
                res = client.query(
                    expr=f"id == {data}",
                    output_fields=["id", "user_id", "audio_text", "audio_vector", "audio_file", "is_delete"],
                    limit=1
                )
                if res:
                    return res[0]
                else:
                    return None
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询 milvus voice 异常 - {data}")
            raise e

    async def del_by_voice(self, data):
        try:
            with self.get_collection("avatar_voice") as client:
                res = client.delete(f"id == {data}")
                return res.delete_count
        except Exception as e:
            logger.bind(tag=TAG).error(f"删除 milvus voice 异常 - {data}")
            raise e

    async def insert_train(self, datas: list[dict]):
        with self.get_collection("avatar_train") as collection:
            collection.insert(datas)
            collection.flush()
            msg = "新增分身训练数据成功"
            logger.bind(tag=TAG).info(f"{msg} - {datas[0]['user_id']} - {datas[0]['avatar_id']}")
        return "OK"

    async def search_by_train(self, user_id, avatar_id, query_text, query_embedding):
        try:
            # 检索使用时，分身自动继承综合分身的，训练数据，及所有标签等元数据（分身及综合 一起查），并取最新时间的数据为准
            # 训练数据优先级，取最新时间的数据，提示词中标注
            with self.get_collection("avatar_train") as client:
                # 稀疏向量检索（BM25全文匹配）
                sparse_params = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}
                sparse_request = AnnSearchRequest(
                    [query_text], "content_bm25", sparse_params, limit=15, expr=f"avatar_id in ['{avatar_id}', '{user_id}-000']",
                )
                # 稠密向量检索
                dense_params = {"metric_type": "IP", "params": {"nprobe": 400}}
                dense_request = AnnSearchRequest(
                    [query_embedding], "vector", dense_params, limit=15, expr=f"avatar_id in ['{avatar_id}', '{user_id}-000']",
                )
                res = client.hybrid_search(
                    reqs=[sparse_request, dense_request],
                    rerank=RRFRanker(k=30),
                    limit=10,
                    output_fields=["id", "user_id", "avatar_id", "content", "update_time", "is_delete"],
                )
                # 假设我们只保留分数大于等于0.06的文档
                sorted_res = [hit for hit in res[0] if hit.score >= 0.06]
                sorted_res = sorted(sorted_res, key=lambda x: datetime.fromisoformat(x.entity.get('update_time')), reverse=True)
                return [hit.get('content') for hit in sorted_res]

            '''
            答案生成：
            context = "\n\n".join([doc["entity"]["content"] for doc in hybrid_results])

prompt = f"""Answer the following question based on the provided context.
If the context doesn't contain relevant information, just say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context.",
        },
        {"role": "user", "content": prompt},
    ],
)

print(response.choices[0].message.content)

            '''
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询 milvus train 异常 - {avatar_id} - {query_text}")
            raise e

    async def search_all_train(self, avatar_id):
        try:
            with self.get_collection("avatar_train") as client:
                res = client.query(
                    expr=f"avatar_id == '{avatar_id}'",
                    order_by="update_time ASC",
                    output_fields=["id", "user_id", "avatar_id", "content", "update_time", "is_delete"],
                )
                if res and len(res[0]) > 0:
                    return [hit.get('content') for hit in res[0]]
                return None
        except Exception as e:
            logger.bind(tag=TAG).error(f"查询 milvus train 异常 - {avatar_id}")
            raise e

    async def del_by_train(self, avatar_id):
        try:
            with self.get_collection("avatar_train") as client:
                res = client.delete(f"avatar_id == '{avatar_id}'")
                return res.delete_count
        except Exception as e:
            logger.bind(tag=TAG).error(f"删除 milvus train 异常 - {avatar_id}")
            raise e

    def _shutdown(self):
        """关闭所有连接（应用退出时调用）"""
        for collection in self.load_collections:
            collection.release()

        for alias in self.aliases:
            try:
                if connections.has_connection(alias): connections.disconnect(alias)
            except ConnectionError as e:
                pass
        logger.bind(tag=TAG).info("Milvus 连接池已关闭")


# 全局连接池实例
MilvusClientProvider = MilvusClientPool()
