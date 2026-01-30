from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_deepseek import ChatDeepSeek


"""
数据模型，逻辑路由
"""
class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源"""
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="根据用户问题，选择最适合回答问题的数据源",
    )


def create_router():
    """创建并返回路由模型"""
    # 带函数调用的大模型
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # 提示模板
    system = """你是将用户问题路由到合适数据源的专家。
根据问题所涉及的编程语言，将其路由到相关的数据源。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    # 定义路由器
    return prompt | structured_llm


def route_question(question: str) -> str:
    """路由用户问题到合适的数据源"""
    router = create_router()
    result = router.invoke({"question": question})
    return result.datasource


if __name__ == "__main__":
    # 测试问题
    test_question = "Python中的列表和元组有什么区别？"
    result = route_question(test_question)
    print(f"问题: {test_question}")
    print(f"路由结果: {result}")


"""
语义路由
"""
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 定义两个提示模板
combat_template = """你是一位精通黑悟空战斗技巧的专家。
你擅长以简洁易懂的方式回答关于黑悟空战斗的问题。
当你不知道问题的答案时，你会坦诚相告。

以下是一个问题：
{query}"""

story_template = """你是一位熟悉黑悟空故事情节的专家。
你擅长将复杂的情节分解并详细解释。
当你不知道问题的答案时，你会坦诚相告。

以下是一个问题：
{query}"""

# 初始化嵌入模型
embeddings = OpenAIEmbeddings()
prompt_templates = [combat_template, story_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# 定义路由函数
def prompt_router(input):
    # 对用户问题进行嵌入
    query_embedding = embeddings.embed_query(input["query"])
    # 计算相似度
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # 选择最相似的提示模板
    print("使用战斗技巧模板" if most_similar == combat_template else "使用故事情节模板")
    return PromptTemplate.from_template(most_similar)

# 创建处理链
chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)

# 示例问题
print(chain.invoke("黑悟空是如何打败敌人的？"))
