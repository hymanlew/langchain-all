from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader


# 1，CSV
# with open(csv_file_path, mode='r', encoding='utf-8') as f:
#     csv_reader = csv.DictReader(f)
#     for row in csv_reader:

loader = CSVLoader(file_path='file/weather_district_id.csv', encoding='utf-8')
data = loader.load()
for record in data[:2]:
    print(record)


# 2，PDF
# 每一页对应一个 document，并且提取图片中的文字
loader = PyPDFLoader(file_path='file/test.pdf', extract_images=True)
data = loader.load()
print(data)


# 3，HTML
loader = WebBaseLoader(
    web_paths=('https://fastapi.tiangolo.com/zh/features/',),
    encoding='utf-8',
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=('md-content',)))
)
docs = loader.load()
print(docs)


# 4，JSON
"""
jq_schema：定义如何解析JSON数据，使用 jq 语法提取特定字段。
    - .messages[] 表示提取 messages 数组中的每个元素（如聊天记录）。
    - 若需提取嵌套字段，可用 .array.key.subkey。
    - 若需提取多个嵌套字段，可用 .array | {key, subkey}。
    
metadata_func：自定义函数，用于从JSON记录中提取元数据并附加到文档。
    - 函数需接收两个参数：record（当前JSON对象）和 metadata（要返回的元数据），返回更新后的元数据字典。

text_content：控制是否将提取的内容视为纯文本。
    - False 表示保留原始数据结构（如字典），适合后续结构化处理。
    - True 则强制转换为字符串。

jq 安装：需提前安装 jq 包：pip install jq，否则会报错。
JSONL文件支持：若文件为JSON Lines格式（每行一个JSON对象），需添加参数 json_lines=True。
内容键（content_key）：若需从复杂对象中提取特定字段作为文档内容（如仅提取 content 字段），需指定 content_key="content"。
"""
def create_metadata(record: dict, metadata: dict) -> dict:
    metadata['sender_name'] = record.get('sender_name')
    metadata['timestamp_ms'] = record.get('timestamp_ms')
    return metadata

loader = JSONLoader(
    file_path='file/test.json',
    jq_schema='.messages[]',
    # jq_schema='.messages[].content',
    # jq_schema='.messages[] | {content, sender_name}',
    metadata_func=create_metadata,
    text_content=False
)
data = loader.load()
print(data)


# 5，MD 文件
# 整个 md 文件内容是一个 document
# loader = UnstructuredMarkdownLoader(file_path='./file/test_translated.md')
# 将 md 文件内容按照 element（段落）拆分成一个个 document
loader = UnstructuredMarkdownLoader(file_path='./file/test_translated.md', mode='elements')
data = loader.load()
print(data)
