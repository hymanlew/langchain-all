import requests
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from datasets import Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# 使用 Ollama 服务
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
llm = ChatOpenAI(model="gwen2:7b-instruct-g4_0")
template = """您是问答任务的助理。使用以下检索到的上下文来回答问题。
如果你不知道答案，就说你不知道。
最多使用三句话，不超过100字，保持答案简洁。
Question: {question}
Context: {context}
Answer: """
prompt = ChatPromptTemplate.from_template(template)


# 从网络查询数据，建索引，构建数据集
def website_data():
    loader = WebBaseLoader("https://baike.baidu.com/item/AIGC-box")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(chunks[0].page_content)
    return chunks

# embedding 知识库，保存到向量数据库
def embedding_data(chunks):
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_langchain_db")
    retriever = vector_store.as_retriever()
    return vector_store, retriever, embeddings

def generate_testset_from_online_files():
    chunks = website_data()
    vector_store, retriever, embedding = embedding_data(chunks)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 有时候会评估失败，最要注意的是知识内容可能有一些特殊字符没有清洗干净，影响了Json数据格式报错
    # RAGAS 作为一个无需参照的评估框架，其评估数据集相对简单。准备一些 question和 ground_truths 的配对，并从中推导出其他所需信息操作如下:
    questions = [
        "艾伦图灵的论文叫什么?",
        "人工智能生成的画作在佳士得拍卖行卖了什么价格?",
        "目前企业在使用相关 AIGC 能力时，主要有哪五种方式?"
    ]
    ground_truths = [
        ["计算机器与智能(Computing Machinery and Intelligence)"],
        ["43.25万美元"],
        ["直接使用、Prompt、LoRA、Finetune、Train"]
    ]
    answers = []
    contexts = []
    # 答案和相关上下文文档（知识库中的数据）都是 langchain 通过检索生成，relevant 相关性
    for query in questions:
        answers.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

    # RAGAS 评估需要以下四个数据
    # 如果不关注 context_recall 指标，就不必提供 ground_truths 数据。在这种情况下，你只需准备 question 即可评估 RAG
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }

    # Ragas测评需要使用标准的Datasets数据格式，因此需要提前将自定义数据集进行封装
    # 传统方式：将字典转换为数据集,单轮问答评估场景,简单直接
    dataset = Dataset.from_dict(data)
    return dataset

# -------------------------------------------------------------------------------------------------------------------

# 加载本地原始文档，建索引，构建数据集
from ragas import Dataset
from langchain_community.document_loaders import DirectoryLoader

def local_data(directory_path="your-directory"):
    """加载本地目录中的文档并进行处理"""
    try:
        loader = DirectoryLoader(directory_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print(f"成功加载本地文档，共 {len(documents)} 个文件，分割为 {len(chunks)} 个文本块")
        if chunks:
            print("第一个文本块内容预览:")
            print(chunks[0].page_content[:200] + "...")
        return chunks
    except Exception as e:
        print(f"加载本地文档时出错: {str(e)}")
        return None

def generate_testset_from_docs(documents, llm, retriever, test_size=10):
    # 使用已经定义的 RAG 链来生成问题和答案
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 从文档中提取关键信息来创建问题
    questions = []
    ground_truths = []
    
    # 这里是一个简化的方法，实际应用中可以根据文档内容设计更复杂的问题生成策略
    for i, doc in enumerate(documents[:test_size]):
        # 从文档中提取关键事实作为答案
        content = doc.page_content
        # 创建基于内容的简单问题
        if len(content) > 100:
            question = f"根据文档，关于这个主题有什么重要信息？"
            questions.append(question)
            ground_truths.append([content[:200] + "..." if len(content) > 200 else content])
    
    # 生成答案和上下文
    answers = []
    contexts = []
    for query in questions:
        answers.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
    
    # 创建数据集
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    return dataset

# 基于本地文件生成测试数据集
def generate_testset_from_local_files(directory_path="your-directory", test_size=10):
    """从本地文件生成测试数据集"""
    print(f"正在从本地目录 {directory_path} 生成测试数据集...")
    
    # 加载本地文档
    chunks = local_data(directory_path)
    if not chunks:
        print("没有成功加载到文档，无法生成测试数据集")
        return None
    
    # 为本地文档创建向量存储和检索器
    vector_store, retriever, embeddings = embedding_data(chunks)
    
    # 生成测试数据集
    test_dataset = generate_testset_from_docs(chunks, llm, retriever, test_size)
    
    print(f"成功生成包含 {len(test_dataset)} 个问题的测试数据集")
    return test_dataset

# -----------------------------------------------------------------------------------------------------------------

from datasets import load_dataset
def generate_testset_from_datasets(test_size=10):
    """从在线文件生成测试数据集"""
    eval_dataset = load_dataset("explodinggradients/earning_report_summary", split="train")
    return eval_dataset

# -----------------------------------------------------------------------------------------------------------------

"""
评估 RAG
首先从 ragas.metrics 导入要使用的所有度量标准，然后将度量标准和已准备好的数据集传入 evaluate() 函数即可。
评估用的大语言模型及 embedding 模型可以是本地部署的，也可以为线上模型，可以参考 [评估框架] 中的文档

Ragas 提供了五种评估指标包括：
- 忠实度 (faithfulness)：衡量生成的答案(answer)与给定上下文(context)的事实一致性，数据一致性
- 答案相关性(Answer relevancy)：评估生成的答案(answer)与用户问题(question)之间相关程度，是否完整地回答了所有问题
- 上下文精度(Context precision)：评估在所有上下文(contexts)中与基本事实(ground-truth)相关的条目，是否排名较高，是否在上下文的顶部
- 上下文召回率(Context recall)：衡量检索到的上下文(Context)与提供的真实答案(ground truth)的一致程度，是否完整，全部地召回了相关文档
- 上下文相关性(Context relevancy)：衡量检索到的上下文(Context)的相关性，是否与用户问题(question)相关。在最新版本中移除了。

忠实度：计算过程：
- 使用 LLM 从答案中抽取主张。
- 使用 LLM 验证每个主张是否可以从上下文中推断出来。
- 使用公式计算 答案中主张 / 上下方主张 = score。

答案相关性：计算过程：
分数通过从答案中逆向推理变体问题，并计算与原始问题的余弦相似度，来衡量原始答案中的信息是否都与问题相关。

上下文精度(Context precision)：
衡量的是检索到的所有文档块中，真正相关的文档块所占的比例以及它们在查到的所有文档中的排名位置。
而在有父子文档情况下检索时，是通过比较子文档与问题的相似度，然后返回相似度较高的子文档所在的父文档，
相关文档的比例高，且最重要的文档被排在了最前面。这为生成模型产出高质量答案奠定了坚实基础。

好的评估指标应该与产品的目标对齐，而这取决于业务需求，并持续根据结果进行系统优化。例如：
- 智能客服（关注：上下文召回率，精度，答案相关性，忠诚度）
- 情感对话（关注：上下文召回率，精度，上下文相关性）
"""
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    # context_relevancy, 已移除
)

run_config = RunConfig(
    max_retries=10, # 重试次数
    max_wait=60, # 重试等待时间
    log_tenacity=True # 是否记录日志
)

# -------------------------------------------------------------------------------------------------------------------
"""
分析指标表现并优化：
- 如果上下文召回率低，考虑增加检索结果数量或使用更先进的检索算法
- 如果上下文精确度低，可能需要优化检索排序或过滤无关文档
- 如果答案正确性低，可能需要改进提示词或微调生成模型
- 如果响应相关性低，检查提示词是否引导模型聚焦于问题
"""
if __name__ == "__main__":
    # 使用示例：基于本地文件生成测试数据集
    local_directory = "path/to/your/documents"

    # 生成测试数据集
    web_dataset = generate_testset_from_online_files(test_size=5)
    local_file_dataset = generate_testset_from_local_files(local_directory, test_size=5)
    ready_dataset = generate_testset_from_datasets(test_size=5)

    if local_file_dataset:
        # 使用生成的测试数据集进行评估
        result = evaluate(
            dataset=local_file_dataset,
            llm=LangchainLLMWrapper(llm),
            embeddings=embeddings,
            run_config=run_config,
            # 根据需要写所要关注的评估指标
            metrics=[
                context_precision,  # 准确率
                context_recall,  # 召回率
                faithfulness,  # 忠实度
                answer_relevancy,  # 相关性
            ],
        )
        print(result)

        # 以二维表格的形式，打印出示例中的 RAGAS 分数
        df = result.to_pandas()
        print(df)
