import pytesseract
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import pandas as pd
import fitz  # PyMuPDF

"""
父子关系的生成逻辑：
1. 父子关系的定义
父节点：高层级结构（如标题、章节名）。
子节点：隶属于父节点的内容（如段落、列表项）。

关系类型：
显式存储：在元数据中直接标注父节点ID（推荐）。
隐式推导：通过层级缩进或序号（如1.1 → 1.1.1）推断。

2. 不存储标题时的解决方案
若仅存储内容（无标题），可通过以下方式生成父子关系：
文本位置关联：用chunk_index标记顺序，子节点的父节点为前一个更高层级的切片。
语义相似度：通过向量相似度匹配父子（需额外计算）。
关键词提取：从内容中提取隐含主题作为虚拟父节点。
"""
from typing import List, Dict
import uuid

# 文档解析与多级分块
class DocumentNode:
    def __init__(self, text: str, level: int, parent_id: str = None):
        self.id = str(uuid.uuid4())
        self.text = text
        self.level = level          # 0=根, 1=章, 2=节, ...
        self.parent_id = parent_id  # 直接父节点ID
        self.path = f"/{self.id}" if parent_id is None else f"{parent_path}/{self.id}"

def parse_hierarchical_document(content: str) -> List[DocumentNode]:
    """解析Markdown/HTML等格式的文档，返回多级节点"""
    # 示例：模拟从Markdown提取结构（实际项目可用BeautifulSoup等库）
    nodes = []
    root = DocumentNode(text="ROOT", level=0)
    nodes.append(root)
    
    # 第一层级（章）
    chapter1 = DocumentNode(text="第一章", level=1, parent_id=root.id)
    chapter2 = DocumentNode(text="第二章", level=1, parent_id=root.id)
    nodes.extend([chapter1, chapter2])
    
    # 第二层级（节）
    section1_1 = DocumentNode(text="1.1 节", level=2, parent_id=chapter1.id)
    section2_1 = DocumentNode(text="2.1 节", level=2, parent_id=chapter2.id)
    nodes.extend([section1_1, section2_1])
    
    # 第三层级（段落/表格）
    paragraph1 = DocumentNode(text="这是1.1节的段落...", level=3, parent_id=section1_1.id)
    table1 = DocumentNode(text="表格内容...", level=3, parent_id=section2_1.id)
    nodes.extend([paragraph1, table1])
    
    return nodes


# 向量化与存储
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# 定义多级嵌套的Collection Schema
schema = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=36),
    FieldSchema(name="level", dtype=DataType.INT8),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512)  # 全路径索引
], description="Hierarchical document chunks")

# 创建集合（企业环境需添加分片和副本配置）
collection = Collection("hierarchical_docs", schema)

# 插入数据示例
nodes = parse_hierarchical_document("...")
embeddings = embedding_model.embed_documents([n.text for n in nodes])  # 假设已初始化模型

collection.create_index(
    field_name="path",
    index_params={"index_type": "Trie", "metric_type": "L2"}
)


data = [
    [n.id for n in nodes],
    embeddings,
    [n.text for n in nodes],
    [n.parent_id for n in nodes],
    [n.level for n in nodes],
    [n.path for n in nodes]
]

# 企业级批量插入（每次500-1000条）
from tqdm import tqdm
batch_size = 500
for i in tqdm(range(0, len(nodes), batch_size)):
    batch = nodes[i:i+batch_size]
    collection.insert(prepare_batch_data(batch))


# 多级查询与树形重建
def get_full_hierarchy(collection: Collection, doc_root_id: str) -> Dict:
    """根据根节点ID重建完整树形结构"""
    # 第一步：查询所有属于该文档的节点
	# 结合向量搜索和层级过滤
	search_params = {
		"expr": 'level == 3',  # 只搜索最底层段落
		"anns_field": "embedding",
		"param": {"nprobe": 128},
		"limit": 10
	}
    search_params['expr'] += f' and path like "/{doc_root_id}%"'  # 利用全路径快速过滤
    results = collection.query(
        expr=search_params,
        output_fields=["id", "text", "level", "parent_id"]
    )
    
    # 第二步：构建树形结构
    tree = {}
    id_to_node = {n["id"]: n for n in results}
    for node in results:
        if node["parent_id"] is None:
            tree[node["id"]] = {"text": node["text"], "children": {}}
        else:
            parent = id_to_node.get(node["parent_id"])
            if parent:
                if "children" not in parent:
                    parent["children"] = {}
                parent["children"][node["id"]] = {
                    "text": node["text"], 
                    "level": node["level"],
                    "children": {}
                }
    return tree

# 使用示例
hierarchy_tree = get_full_hierarchy(collection, "根节点ID")

# 输出示例
{
    "root_id": {
        "text": "ROOT",
        "children": {
            "chapter1_id": {
                "text": "第一章",
                "level": 1,
                "children": {
                    "section1_1_id": {
                        "text": "1.1 节",
                        "level": 2,
                        "children": {
                            "paragraph1_id": {
                                "text": "这是段落内容...",
                                "level": 3,
                                "children": {}
                            }
                        }
                    }
                }
            }
        }
    }
}
