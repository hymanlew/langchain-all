# 导入相关的库
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.load import dumps, loads

llm = ChatDeepSeek(model="deepseek-chat")

# 加载文档
doc_dir = "90-文档-Data/山西文旅"
def load_documents(directory):
    """读取目录中的所有文档（包括PDF、TXT、DOCX)"""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue  # 跳过不支持的文件类型
        documents.extend(loader.load())
    return documents
docs = load_documents(doc_dir)
# 文本切块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)
# 获取嵌入并创建向量索引
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
retriever = vectorstore.as_retriever()

# RRF算法（多检索系统结果融合）
# 对于每个文档在每一个排序列表中的位置，计算一个得分（倒数排名分数，即分数越高越靠前），然后将这个文档在所有列表中的得分相加，得到一个文档的融合总分。最后将所有文档按照总分重新排序。
# 具体公式：对于每个文档，在每个列表中如果出现，则加 1/(rank + k)，其中rank是文档在该列表中的排名（从0开始），k是一个常数（通常取60）。
# 如果文档在某个列表中没有出现，则不加分。

# 较大的 k：减小排名差异的影响（更平滑）
# 较小的 k：放大排名差异的影响（更敏感）
# 典型值范围：k=60（信息检索领域常用值）
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        # items 返回 k, v 元组，x[1] 就是第二个元素 v
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

# 生成多个搜索查询
template = """你是一个帮助用户生成多个搜索查询的助手。\n
请根据以下问题生成多个相关的搜索查询：{question} \n
输出（4个查询）："""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

questions = [
    "山西有哪些著名的旅游景点？",
    "云冈石窟的历史背景是什么？",
    "五台山的文化和宗教意义是什么？"
]
# 进行检索和RRF处理
# retriever.map() 从单个输入，转换成，可以多个输入
for question in questions:
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    
    print(f"\n【问题】{question}")
    print(f"文档数量：{len(docs)}")
    for doc, score in docs[:3]:  # 显示前3个结果
        print(f"文档内容：{doc.page_content[:200]}...")  # 只展示前200个字符
