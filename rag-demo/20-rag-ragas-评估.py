"""
# 注意，安装最新版本可能出现 Failed to parse output，Returning None 错误
# 具体看最新版本是否修复
"""
import requests
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
# 注释掉未使用的导入
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from datasets import Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 准备知识库数据，建索引
def prepare_data():
    loader = WebBaseLoader("https://baike.baidu.com/item/AIGC-box")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(chunks[0].page_content)
    return chunks

# embedding 知识库，保存到向量数据库
def embedding_data(chunks):
    # rag_embeddings = OpenAIEmbeddings ()
    rag_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vector_store = Chroma.from_documents(documents=chunks, embedding=rag_embeddings, persist_directory="./chroma_langchain_db")
    retriever = vector_store.as_retriever()
    return vector_store, retriever, rag_embeddings

# 使用 Ollama 服务
llm = OllamaLLM(model="gwen2:7b-instruct-g4_0")
template = """您是问答任务的助理。使用以下检索到的上下文来回答问题。
如果你不知道答案，就说你不知道。
最多使用三句话，不超过100字，保持答案简洁。
Question: {question}
Context: {context}
Answer: """
prompt = ChatPromptTemplate.from_template(template)
chunks = prepare_data()
vector_store, retriever, embedding = embedding_data(chunks)

# 生成答案
def ragas_eval():
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
    # 将字典转换为数据集
    dataset = Dataset.from_dict(data)
    return dataset


"""
评估 RAG
首先从 ragas.metrics 导入要使用的所有度量标准，然后将度量标准和已准备好的数据集传入 evaluate() 函数即可。
评估用的大语言模型及 embedding 模型可以是本地部署的，也可以为线上模型，可以参考 [评估框架] 中的文档

Ragas 提供了五种评估指标包括：
- 忠实度 (faithfulness)：衡量生成的答案(answer)与给定上下文(context)的事实一致性，数据一致性
- 答案相关性(Answer relevancy)：评估生成的答案(answer)与用户问题(question)之间相关程度，是否完整地回答了所有问题
- 上下文精度(Context precision)：评估在所有上下文(contexts)中与基本事实(ground-truth)相关的条目，是否排名较高，是否在上下文的顶部
- 上下文召回率(Context recall)：衡量检索到的上下文(Context)与人类提供的真实答案(ground truth)的一致程度，是否完整地召回了相关文档
- 上下文相关性(Context relevancy)：衡量检索到的上下文(Context)的相关性，是否与用户问题(question)相关

上下文精度(Context precision)：此参数参考性较低，因为在有父子文档情况下检索时，是通过比较子文档与问题的相似度，然后返回相似度较高的子文档所在的父文档，
而子文档在父文档中的位置不一定是在顶部，也很有可能是在父文档的中部或者底部，因此这就会导致较低的 context_precision 得分。
"""
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

run_config = RunConfig(
    max_retries=10,
    max_wait=60,
    log_tenacity=True
)

result = evaluate(
    dataset=ragas_eval(),
    llm=llm,
    embeddings=embedding,
    run_config=run_config,
    # 根据需要写所要关注的评估指标
    metrics=[
        context_precision,  # 准确率
        context_recall,     # 召回率
        faithfulness,       # 忠实度
        answer_relevancy,   # 相关性
    ],
)
print(result)

# 以二维表格的形式，打印出示例中的 RAGAS 分数
df = result.to_pandas()
print(df)

