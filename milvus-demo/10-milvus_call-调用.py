import logging
from pymilvus import DataType, WeightedRanker, AnnSearchRequest, RRFRanker
from milvus_manager import MilvusManager

'''
MilvusManager 是历史性解决方案，多见于 Milvus 1.x 或早期 2.x 项目，本质是开发者补丁而非版本特性。
MilvusManager 并非 Milvus 官方库中的标准类名，而是开发者自定义或第三方封装的工具类，用于管理 Milvus 连接和操作。其设计背景与 Milvus 版本演进密切相关
'''
logging.basicConfig(level=logging.INFO)

def main():
    # 初始化Milvus管理器（单例）
    milvus = MilvusManager()
    
    # 注册生产环境和测试环境连接配置
    milvus.connect(
        alias="prod",
        host="10.0.0.1",
        port="19530",
        user="admin",
        password="your_password"
    )
    
    milvus.connect(
        alias="test",
        host="10.0.0.2",
        port="19530",
        user="test",
        password="test"
    )
    
    # ========== 1. 创建集合 ==========
    fields = [
        {"name": "id", "dtype": DataType.INT64, "is_primary": True},
        {"name": "embedding", "dtype": DataType.FLOAT_VECTOR, "dim": 128},
        {"name": "title", "dtype": DataType.VARCHAR, "max_length": 512}
    ]
    # 创建真实连接
    collection = milvus.create_collection(
        collection_name="product_embeddings",
        fields=fields,
        alias="prod",
        description="Product recommendation vectors"
    )
    
    # ========== 2. 插入测试数据 ==========
    import numpy as np
    
    data = {
        "embedding": np.random.rand(1000, 128).tolist(),  # 1000个128维向量
        "title": [f"Product_{i}" for i in range(1000)]
    }
    
    milvus.insert_data(
        collection_name="product_embeddings",
        data=data,
        alias="prod",
        batch_size=500  # 分两批插入
    )
    
    # ========== 3. 创建索引 ==========
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    
    milvus.create_index(
        collection_name="product_embeddings",
        field_name="embedding",
        index_params=index_params,
        alias="prod"
    )
    
    # ========== 4. 向量搜索 ==========
    query_vector = np.random.rand(1, 128).tolist()  # 单个查询向量
    search_params = {"nprobe": 16}
    
    results = milvus.search(
        collection_name="product_embeddings",
        vectors=query_vector,
        search_params=search_params,
        limit=5,
        output_fields=["title"],
        alias="prod"
    )
    for item in results[0]:
        print(f"ID: {item['id']}, Title: {item['title']}, Score: {item['score']:.4f}")

    # 5. 文本匹配搜索示例
    print("\n=== 文本匹配搜索 ===")
    filter = "TEXT_MATCH(text, 'text_1 text_2')"  # 搜索包含text_1或text_2的文档
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        filter=filter,
        limit=3,
        search_params={"metric_type": "L2"},
        output_fields=["text"]
    )

    # 7. 基本分组搜索示例
    query_vector = [random.random() for _ in range(128)]
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=5,  # 返回5个不同的文档组
        group_by_field="docId",  # 按文档ID分组
        output_fields=["docId", "chunk"]
    )

    # 8. 配置组大小的分组搜索示例
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=3,  # 返回3个不同的文档组
        group_by_field="docId",
        group_size=2,  # 每个组返回2个最相似的结果
        strict_group_size=True,  # 严格确保每个组有2个结果
        output_fields=["docId", "chunk"]
    )

    # 9. 范围搜索示例
    # 使用 L2 距离度量，设置范围搜索参数
    # 注意：对于 L2 距离，range_filter 应该小于 radius
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=10,  # 增加限制以显示更多结果
        search_params={
            "metric_type": "L2",
            "params": {
                "radius": 1.0,  # 外圈半径
                "range_filter": 0.5  # 内圈半径
            }
        },
        output_fields=["color"]
    )
    print(f"范围搜索结果,搜索范围: 距离在 {0.5} 到 {1.0} 之间的向量")
    for hits in results:
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}, 颜色: {hit['entity']['color']}")

    # 10. 使用 SearchIterator 进行搜索
    query_vector = [random.random() for _ in range(128)]
    iterator = client.search_iterator(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        search_params={"metric_type": "L2"},
        batch_size=1000,  # 每批返回1000条结果
        limit=20000,  # 总共返回20000条结果
        output_fields=["color"]
    )
    # 使用迭代器获取结果
    all_results = []
    while True:
        result = iterator.next()
        if not result:
            iterator.close()
            break

        # 将结果转换为字典并添加到结果列表
        for hit in result:
            all_results.append(hit.to_dict())

    print(f"总共获取到 {len(all_results)} 条结果")
    print("\n前5条结果:")
    for result in all_results[:5]:
        print(f"ID: {result['id']}, 距离: {result['distance']}, 颜色: {result['entity']['color']}")

    # 10.1 使用 Get 方法查询指定 ID 的数据
    get_results = client.get(
        collection_name=COLLECTION_NAME,
        ids=[0, 1, 2],
        output_fields=["vector", "color"]
    )
    for result in get_results:
        print(f"ID: {result['id']}, 颜色: {result['color']}")

    # 10.2 使用 Query 方法进行条件查询
    query_results = client.query(
        collection_name=COLLECTION_NAME,
        filter="color like \"color_1%\"",  # 查询颜色以 color_1 开头的记录
        output_fields=["id", "color"],
        limit=5
    )
    for result in query_results:
        print(f"ID: {result['id']}, 颜色: {result['color']}")

    # 10.3 使用 QueryIterator 进行分页查询
    iterator = collection.query_iterator(
        batch_size=10,
        expr="color like \"color_1%\"",
        output_fields=["id", "color"]
    )
    print("QueryIterator 查询结果:")
    while True:
        result = iterator.next()
        if not result:
            iterator.close()
            break
        for item in result:
            print(f"ID: {item['id']}, 颜色: {item['color']}")

    # ========== 5. 清理资源 ==========
    # 程序退出时会自动调用close()，此处演示手动关闭
    milvus.close("test")  # 仅关闭测试环境
    # milvus.close()      # 关闭所有连接

# 混合搜索
def hybrid_search(query, category=None, environment=None, search_type=None, rerank_method=None):
    limit = 5
    weights = {"sparse": 0.7, "dense": 1.0}
    rrf_k = 60  # RRF 参数
    query_embeddings = [query]

    # 构建过滤表达式
    conditions = []
    if category:
        conditions.append(f'category == "{category}"')
    if environment:
        conditions.append(f'environment == "{environment}"')
    expr = " && ".join(conditions) if conditions else None

    search_params = {"metric_type": "IP", "params": {}}
    if expr:
        search_params["expr"] = expr

    if search_type == "hybrid":
        dense_req = AnnSearchRequest(
            data=[query_embeddings["dense"][0]],
            anns_field="dense_vector",
            param=search_params,
            limit=limit
        )
        sparse_req = AnnSearchRequest(
            data=[query_embeddings["sparse"]._getrow(0)],
            anns_field="sparse_vector",
            param=search_params,
            limit=limit
        )

        if rerank_method == "weighted":
            rerank = WeightedRanker(weights["dense"], weights["sparse"])
            print(f"\n使用加权重排，权重：稀疏={weights['sparse']}, 密集={weights['dense']}")
        else:  # rrf
            rerank = RRFRanker(rrf_k)
            print(f"\n使用 RRF 重排，k={rrf_k}")
        # WeightedRanker的参数顺序必须与reqs列表中的请求顺序一一对应

        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=rerank,
            limit=limit,
            output_fields=["text", "id", "title", "category", "location", "environment"]
        )[0]
    else:
        field = "dense_vector" if search_type == "dense" else "sparse_vector"
        vec = query_embeddings["dense"][0] if search_type == "dense" else query_embeddings["sparse"][0]
        results = collection.search(
            data=[vec],
            anns_field=field,
            param=search_params,
            limit=limit,
            output_fields=["text", "id", "title", "category", "location", "environment"]
        )[0]

    return results


if __name__ == "__main__":
    main()
