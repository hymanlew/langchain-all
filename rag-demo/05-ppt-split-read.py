from unstructured.partition.ppt import partition_ppt

# 解析 PPT 文件
ppt_elements = partition_ppt(filename="90-文档-Data/黑悟空/黑神话悟空.pptx")
print("PPT 内容：")
for element in ppt_elements:
    print(element.text)

from langchain_core.documents import Document

# 转换为 Documents 数据结构
documents = [
    Document(page_content=element.text,
             metadata={"source": "data/黑神话悟空PPT.pptx"})
    for element in ppt_elements
]

# 输出转换后的 Documents
print(documents[0:3])


