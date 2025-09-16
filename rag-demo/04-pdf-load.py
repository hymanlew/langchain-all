import pytesseract
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import pandas as pd
import fitz  # PyMuPDF

"""
多引擎校验机制：
引擎类型	    适用场景	    准确率提升策略
PaddleOCR	中文印刷体	自定义字典库 
Tesseract5 	英文/数字	LSTM模型优化 
PP-StructureV2	复杂表格	单元格合并算法 
"""
class PDFParser:
    def __init__(self, ocr_engine="paddle"):
        # 多引擎支持，百度开源 OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch") if ocr_engine == "paddle" else None
        self.tesseract_config = r'--oem 3 --psm 6'  # Tesseract优化配置

    def parse_pdf(self, file_path):
        """主解析流程"""
        doc = fitz.open(file_path)
        results = {"text": [], "tables": [], "images": []}

        for page in doc:
            # 文本层提取（保留原始结构）
            text = self._extract_native_text(page)
            results["text"].append(text)

            # 表格检测与重建
            tables = self._detect_tables(page)
            results["tables"].extend(tables)

            # 图像OCR处理
            images = self._extract_images(page)
            results["images"].extend([self._ocr_image(img) for img in images])

        return self._post_process(results)


    # 高精度表格解析
    def _detect_tables(self, page):
        """基于深度学习检测表格边界"""
        # 使用OpenCV检测表格线
        import cv2
        pix = page.get_pixmap(dpi=300)
        img = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGB2GRAY)

        # 表格线增强（企业级优化）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 使用PP-StructureV2表格识别
        table_engine = PPStructure(show_log=True)
        return table_engine(img)

    # 多模态图像处理
    def _ocr_image(self, image):
        """混合OCR引擎处理"""
        # PaddleOCR优先（中文优化）
        result = self.ocr.ocr(image, cls=True)
        if not result:
            # 备用Tesseract引擎
            text = pytesseract.image_to_string(image, config=self.tesseract_config)
            return {"text": text, "confidence": 0.7}

        # 结构化输出
        return {
            "text": "\n".join([line[1][0] for line in result]),
            "positions": [line[0] for line in result],
            "confidence": min(line[1][1] for line in result)
        }

    # 后处理与质量控制
    def _post_process(self, results):
        """企业级后处理流水线"""
        # 表格内容验证
        for table in results["tables"]:
            if table["confidence"] < 0.8:  # 低置信度表格触发复核
                table["content"] = self._manual_verify(table["html"])

        # 文本结构重建（解决PDF阅读顺序问题）
        results["text"] = self._reconstruct_reading_order(results["text"])

        # 输出标准化
        return {
            "version": "1.0",
            "metadata": {"pages": len(results["text"])},
            "content": results
        }


# --------------------------------

from typing import List, Dict
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import re

"""
Unstructured 以语义分割和OCR见长（额外配置 ORC 库，Poppler（PDF 分析）），元素级分割（段落/标题/表格等），适合需要精细化处理的场景；支持 PDF、Word、HTML等
PyMuPDF 是页面级或区域级提取，需额外配置 ORC 库，适合规则文档的批量操作。两者可协同使用，但需根据数据特性权衡性能与精度。

from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
# 本地文件处理（保留元数据）
loader = UnstructuredFileLoader(
    "contract.pdf",
    mode="elements",  # 分块保留标题、表格等结构
    post_processors=[clean_extra_whitespace]  # 企业级文本清洗
)
docs = loader.load()

# 安装 PDF 解析器
pip install -qu "unstructured[pdf]"
pip install -qu "unstructured[md]" nltk
也可以使用 ChatPDF 库，它会首先读取PDF文件，将其转换为可处理的文本格式，例如txt格式；

2. ChatPDF会对提取出来的文本进行清理和标准化，例如去除特殊字符、分段、分句等，以便于
后续处理。这一步可以使用自然语言处理技术，如正则表达式等；

工具介绍：
• Layout-parser：
	• 优点：最大的模型（约800MB）精度非常高
	• 缺点：速度慢一点
• PaddlePaddle-ppstructure：
• 	优点：模型比较小，效果也还行
• unstructured：
• 	缺点：fast模式效果很差，基本不能用，会将很多公式也识别为标题。其他模式或许可行，笔者没有尝试

"""

# 结构化切片策略
class SemanticSlicer:
    def __init__(self):
        self.header_stack = []  # 维护标题层级栈

    def slice_pdf(self, file_path: str) -> List[Dict]:
        """PDF切片并保留结构关系"""
        doc = fitz.open(file_path)
        slices = []

        for page in doc:
            # 提取带格式文本（保留标题标记）
            text = page.get_text("dict")
            blocks = self._process_blocks(text["blocks"])

			# 存储长文档切片时，父子关系通常用于表示文档的层级结构（如标题与段落、章节与子章节）
            # 根据文档的标题层级，手动构建切片上下文
            for block in blocks:
                if block["type"] == "header":
					# 显式存储父级关系：在元数据中直接标注父节点ID（parent：层级及内容）
                    self._update_header_stack(block["text"], block["level"])
                    slices.append({
                        "content": "",
                        "parent": self.header_stack[-2] if len(self.header_stack) > 1 else None,
                        "current_header": self.header_stack[-1]
                    })
                else:
					# 
                    if slices:
                        slices[-1]["content"] += block["text"] + "\n"

        return self._post_process(slices)

    """
    标题/子标题的存储必要性 在企业实践中，一定会存储标题和结构化信息。
    """
    def _process_blocks(self, blocks):
        """识别标题块和内容块"""
        processed = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # 根据字体大小判断标题层级
                        if span["size"] > 14:  # 标题阈值
                            processed.append({
                                "type": "header",
                                "text": span["text"],
                                "level": int(span["size"] / 2)  # 简化层级计算
                            })
                        else:
                            processed.append({
                                "type": "text",
                                "text": span["text"]
                            })
        return processed

    """
    语义连续性保障,连续性保障机制
    技术手段	    实现方式	    企业级优化点
    标题栈维护	通过字体大小识别标题层级，动态维护父子关系   支持多级标题（H1-H6）的自动关联
    上下文衔接	在每个切片头部插入前序片段的关键句	      基于BERT的句子重要性分析 
    跨片段关系验证	检查父标题是否存在，自动修复断裂的层级	      结合知识图谱的实体消歧 
    
    无标题数据的处理，并保证语义连贯，父子关系
    - 先根据内容特点，建立向量库的索引映射（Schema 设计），并通过分类ID关联外部元数据库
    - 内容嵌入：使用领域优化的嵌入模型（如 BGE、text2vec）将正文转化为向量
    - 动态上下文注入：在切片时保留前序文本的关键句
    - 如果是图谱数据：则可以通过实体链接将内容与知识图谱节点关联
    """
    def _post_process(self, slices: List[Dict]) -> List[Dict]:
        """切片后处理（保证语义连贯）"""
        for i in range(1, len(slices)):
            # 前序片段结尾补全
            if len(slices[i - 1]["content"]) > 100:  # 长文本处理
                last_sentences = ". ".join(slices[i - 1]["content"].split(". ")[-2:])
                slices[i]["content"] = f"上下文衔接:{last_sentences}\n{slices[i]['content']}"

            # 父子关系验证
            if slices[i]["parent"] and not any(s["current_header"] == slices[i]["parent"] for s in slices[:i]):
                slices[i]["parent"] = None  # 修复断裂的层级

        return slices
		

"""
文章的切分及关键信息抽取：
- 1，关键信息: 为各语义段的关键信息集合，或者是各个子标题语义扩充之后的集合（pdf多级标题识别及提取见
下一篇文章）
- 2，语义切分方法1：利用NLP的篇章分析（discourse parsing）工具，提取出段落之间的主要关系，譬如上述
极端情况2展示的段落之间就有从属关系。把所有包含主从关系的段落合并成一段。 这样对文章切分完之后
保证每一段在说同一件事情.
- 3，语义切分方法2：除了discourse parsing的工具外，还可以写一个简单算法利用BERT等模型来实现语义分
割。BERT等模型在预训练的时候采用了NSP（next sentence prediction）的训练任务，因此BERT完全可以
判断两个句子（段落）是否具有语义衔接关系。这里我们可以设置相似度阈值t，从前往后依次判断相邻两个
段落的相似度分数是否大于t，如果大于则合并，否则断开。当然算法为了效率，可以采用二分法并行判定，
模型也不用很大，笔者用BERT-base-Chinese在中文场景中就取得了不错的效果。

- 4，文档加工：
• 一种是使用更好的文档拆分的方式（如项目中已经集成的达摩院的语义识别的模型及进行拆分）；
• 一种是改进填充的方式，判断中心句上下文的句子是否和中心句相关，仅添加相关度高的句子；
• 另一种是文本分段后，对每段分别及进行总结，基于总结内容语义及进行匹配；

多语言问题，paper的内容是英文的，用户的query和生成的内容都是中文的，这里有个语言之间的对齐问
题，尤其是可以用中文的query embedding来从英文的text chunking embedding中找到更加相似的top-k是个
具有挑战的问题


ChatPDF首先读取PDF文件，将其转换为可处理的文本格式，例如txt格式；
2. ChatPDF会对提取出来的文本进行清理和标准化，例如去除特殊字符、分段、分句等，以便于
后续处理。这一步可以使用自然语言处理技术，如正则表达式等；
"""

