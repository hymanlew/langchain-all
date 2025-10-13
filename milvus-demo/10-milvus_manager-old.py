
'''
MilvusManager 是历史性解决方案，多见于 Milvus 1.x 或早期 2.x 项目，本质是开发者补丁而非版本特性。
MilvusManager 并非 Milvus 官方库中的标准类名，而是开发者自定义或第三方封装的工具类，用于管理 Milvus 连接和操作。其设计背景与 Milvus 版本演进密切相关
'''
import logging
from milvus_manager import MilvusManager

# 配置日志
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
    
    print("Top 5 similar products:")
    for item in results[0]:
        print(f"ID: {item['id']}, Title: {item['title']}, Score: {item['score']:.4f}")
    
    # ========== 5. 清理资源 ==========
    # 程序退出时会自动调用close()，此处演示手动关闭
    milvus.close("test")  # 仅关闭测试环境
    # milvus.close()      # 关闭所有连接

if __name__ == "__main__":
    main()
