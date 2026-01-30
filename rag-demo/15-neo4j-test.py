# 准备Neo4j数据库连接
from neo4j import GraphDatabase
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化DeepSeek客户端
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Neo4j连接配置
uri = "bolt://localhost:7687"  # 默认Neo4j Bolt端口
username = "neo4j"
password = os.getenv("NEO4J_PASSWORD")  # 从环境变量获取密码

# 初始化Neo4j驱动
driver = GraphDatabase.driver(uri, auth=(username, password))


def get_database_schema():
    """查询数据库的元数据信息"""
    with driver.session() as session:
        # 查询节点标签
        node_labels_query = """
        CALL db.labels() YIELD label
        RETURN label
        """
        node_labels = session.run(node_labels_query).data()

        # 查询关系类型
        relationship_types_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        relationship_types = session.run(relationship_types_query).data()

        # 查询每个标签的属性
        properties_by_label = {}
        for label in node_labels:
            properties_query = f"""
            MATCH (n:{label['label']})
            WITH n LIMIT 1
            RETURN keys(n) as properties
            """
            properties = session.run(properties_query).data()
            if properties:
                properties_by_label[label['label']] = properties[0]['properties']

        return {
            "node_labels": [label['label'] for label in node_labels],
            "relationship_types": [rel['relationshipType'] for rel in relationship_types],
            "properties_by_label": properties_by_label
        }


# 获取数据库结构
schema_info = get_database_schema()
print("\n数据库结构信息：")
print("节点类型：", schema_info["node_labels"])
print("关系类型：", schema_info["relationship_types"])
print("\n节点属性：")
for label, properties in schema_info["properties_by_label"].items():
    print(f"{label}: {properties}")

# 准备SNOMED CT Schema描述
schema_description = f"""
你正在访问一个SNOMED CT图数据库，主要包含以下节点和关系：

节点类型：
{', '.join(schema_info["node_labels"])}

关系类型：
{', '.join(schema_info["relationship_types"])}

节点属性：
"""
for label, properties in schema_info["properties_by_label"].items():
    schema_description += f"\n{label}节点属性：{', '.join(properties)}"

# 准备SNOMED CT Schema描述
schema_description = """
你正在访问一个SNOMED CT图数据库，主要包含以下节点和关系：

节点类型：
1. Concept (概念节点)
   - conceptId: 概念唯一标识符
   - fullySpecifiedName: 完整概念名称
   - preferredTerm: 首选术语
   - active: 是否激活
   - effectiveTime: 生效时间
   - moduleId: 模块ID

2. Description (描述节点)
   - descriptionId: 描述唯一标识符
   - term: 术语文本
   - typeId: 描述类型ID
   - languageCode: 语言代码
   - active: 是否激活

3. Relationship (关系节点)
   - relationshipId: 关系唯一标识符
   - typeId: 关系类型ID
   - active: 是否激活

关系类型：
1. IS_A: 表示概念之间的层级关系
2. HAS_DESCRIPTION: 概念与其描述之间的关系
3. HAS_RELATIONSHIP: 概念之间的其他关系
"""

# 设置查询
user_query = "查找与'Diabetes'相关的所有概念及其描述"

# 准备生成Cypher的提示词
prompt = f"""
以下是SNOMED CT图数据库的结构描述：
{schema_description}
用户的自然语言问题如下：
"{user_query}"

请生成Cypher查询语句，注意以下几点：
1. 关系方向要正确，例如：
   - ObjectConcept 拥有 Description，所以应该是 (oc:ObjectConcept)-[:HAS_DESCRIPTION]->(d:Description)
   - 不要写成 (d:Description)-[:HAS_DESCRIPTION]->(oc:ObjectConcept)
2. 使用MATCH子句来匹配节点和关系
3. 使用WHERE子句来过滤条件，建议使用toLower()函数进行不区分大小写的匹配
4. 使用RETURN子句来指定返回结果
5. 请只返回Cypher查询语句，不要包含任何其他解释、注释或格式标记（如```cypher）
"""

# 调用LLM生成Cypher语句
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system",
         "content": "你是一个Cypher查询专家。请只返回Cypher查询语句，不要包含任何Markdown格式或其他说明。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

# 清理Cypher语句，移除可能的Markdown标记
cypher = response.choices[0].message.content.strip()
cypher = cypher.replace('```cypher', '').replace('```', '').strip()
print(f"\n生成的Cypher查询语句：\n{cypher}")


# 执行Cypher查询并获取结果
def run_query(tx, query):
    result = tx.run(query)
    return [record for record in result]


with driver.session() as session:
    results = session.execute_read(run_query, cypher)
    print(f"\n查询结果：{results}")

# 生成自然语言描述
if results:
    nl_prompt = f"""
    查询结果如下：
    {results}
    请将这些数据转换为自然语言描述，使其易于理解。
    原始问题是：{user_query}

    要求：
    1. 使用通俗易懂的语言
    2. 包含所有查询到的数据信息
    3. 如果有专业术语，请适当解释
    """
    response_nl = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个医学信息专家，负责将SNOMED CT查询结果转换为易懂的自然语言描述。"},
            {"role": "user", "content": nl_prompt}
        ],
        temperature=0.7
    )
    description = response_nl.choices[0].message.content.strip()
    print(f"自然语言描述：\n{description}")
else:
    print("未找到相关数据。")

# 关闭数据库连接
driver.close()
