from unstructured.partition.pdf import partition_pdf

# 指定文件路径
file_path = "/home/huangj2/Documents/20250202_RAG_Book/data/山西文旅/云冈石窟-en.pdf"

# 解析PDF
elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",  # 使用高精度模式，普通模式是fast
    chunking_strategy="by_title",  # 按标题分块, 普通模式是Basic
    max_characters=2500,  # 每个分块的最大字符数
    new_after_n_chars=2300,  # 在达到指定字符数后开始新分块
    infer_table_structure=True,  # 推断表格结构
    extract_images=True,  # 提取图像
    image_format="png",  # 图像格式
    include_metadata=True  # 包含元数据
)

# 展示解析后的元素
for i, element in enumerate(elements):
    print(f"Element {i+1}:")
    print(element)
    print("元数据:", element.metadata.to_dict())  # 打印元素的元数据
    print("-" * 80)

# 创建一个元素ID到元素的映射
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

for element in elements:
    if element.category == "Table":  # 只打印表格数据
        print("\n表格数据:")
        print("表格元数据:", vars(element.metadata))  # 使用vars()显示所有元数据属性
        print("表格内容:")
        print(element.text)  # 打印表格文本内容

        # 获取并打印父节点信息
        parent_id = getattr(element.metadata, 'parent_id', None)
        if parent_id and parent_id in element_map:
            parent_element = element_map[parent_id]
            print("\n父节点信息:")
            print(f"类型: {parent_element.category}")
            print(f"内容: {parent_element.text}")
            if hasattr(parent_element, 'metadata'):
                print(f"父节点元数据: {vars(parent_element.metadata)}")  # 同样使用vars()显示所有元数据
        else:
            print(f"未找到父节点 (ID: {parent_id})")
        print("-" * 50)

text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]

