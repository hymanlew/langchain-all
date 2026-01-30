#1 为3篇博客文章创建索引
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 添加到向量数据库
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

#2 检索评分器
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# 数据模型
class GradeDocuments(BaseModel):
    """对检索文档相关性的二元评分。"""
    binary_score: str = Field(
        description="文档与问题相关为'yes'，不相关为'no'"
    )

# 带有函数调用的语言模型
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 提示模板
system = """你是一个评估检索文档与用户问题相关性的评分员。 \n 
    如果文档包含与问题相关的关键词或语义含义，则将其评为相关。 \n
    给出一个二元评分'yes'或'no'来表示文档是否与问题相关。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档: \n\n {document} \n\n 用户问题: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

#3 设置生成模型
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# 提示模板
prompt = hub.pull("rlm/rag-prompt")

# 语言模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 后处理
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 链式调用
rag_chain = prompt | llm | StrOutputParser()

# 运行
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

#4 设置问题重写器
# 语言模型
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# 提示模板
system = """你是一个问题重写者，将输入的问题转换为更适合网络搜索的版本。 \n 
     分析输入并尝试推理出潜在的语义意图/含义。"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "这是初始问题: \n\n {question} \n 请重新表述为一个改进的问题。",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

#5 设置网络搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

#6 设置CRAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

#7 定义图状态
from typing import List 
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    表示我们图的状态。

    属性:
        question: 问题
        generation: 语言模型生成
        web_search: 是否添加搜索
        documents: 文档列表
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]

from langchain.schema import Document

def retrieve(state):
    """
    检索文档

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 向状态添加新的键，documents，包含检索到的文档
    """
    print("---检索---")
    question = state["question"]

    # 检索
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    生成答案

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 向状态添加新的键，generation，包含语言模型生成的内容
    """
    print("---生成---")
    question = state["question"]
    documents = state["documents"]

    # RAG生成
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    确定检索到的文档是否与问题相关。

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 更新documents键，只保留经过筛选的相关文档
    """

    print("---检查文档与问题的相关性---")
    question = state["question"]
    documents = state["documents"]

    # 对每个文档评分
    filtered_docs = []
    web_search = "No"
    has_relevant_docs = False
    
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---评分: 文档相关---")
            filtered_docs.append(d)
            has_relevant_docs = True
        else:
            print("---评分: 文档不相关---")
            continue
    
    # 只有在没有任何相关文档时才进行网络搜索
    if not has_relevant_docs:
        web_search = "Yes"
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    """
    转换查询以生成更好的问题。

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 用重新表述的问题更新question键
    """

    print("---转换查询---")
    question = state["question"]
    documents = state["documents"]

    # 重写问题
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    使用网络搜索工具获取额外信息。

    参数:
        state (dict): 包含当前状态
            - question: 问题
            - documents: 文档列表

    返回:
        state (dict): 用追加的网络搜索结果更新documents键
    """

    print("---网络搜索---")
    question = state["question"]
    documents = state["documents"]

    # 网络搜索
    search_results = web_search_tool.invoke(question)
    # 将搜索结果列表转换为字符串
    search_results_str = "\n".join([str(result) for result in search_results])
    web_results = Document(page_content=search_results_str)
    documents.append(web_results)

    return {"documents": documents, "question": question}

### 边缘处理

def decide_to_generate(state):
    """
    决定是生成答案还是重新生成问题。

    参数:
        state (dict): 当前图状态

    返回:
        str: 下一个要调用的节点的二元决策
    """

    print("---评估已评分文档---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # 所有文档都已被check_relevance过滤
        # 我们将重新生成一个新的查询
        print(
            "---决策: 所有文档与问题都不相关，转换查询---"
        )
        return "transform_query"
    else:
        # 我们有相关文档，所以生成答案
        print("---决策: 生成---")
        return "generate"

#8 编译图
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("retrieve", retrieve)  # 检索
workflow.add_node("grade_documents", grade_documents)  # 评分文档
workflow.add_node("generate", generate)  # 生成
workflow.add_node("transform_query", transform_query)  # 转换查询
workflow.add_node("web_search_node", web_search)  # 网络搜索

# 构建图
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# 编译
app = workflow.compile()

#9 使用图回答问题

from pprint import pprint

# 运行
inputs = {"question": "What are the types of agent memory?"}
# inputs = {"question": "为何山西省旅游资源丰富?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # 节点
        pprint(f"节点 '{key}':")
        # 可选：在每个节点打印完整状态
        # pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# 最终生成
pprint(value["generation"])


