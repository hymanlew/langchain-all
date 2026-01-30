import logging
from typing import List
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import PromptTemplate

# 加载游戏相关文档并构建向量数据库
loader = TextLoader("90-文档-Data/黑悟空/黑悟空设定.txt", encoding='utf-8')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents=splits, embedding= embed_model)

# 自定义输出解析器
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # 过滤空行

output_parser = LineListOutputParser()
# 自定义查询提示模板
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一个资深的游戏客服。请从5个不同的角度重写用户的查询，以帮助玩家获得更详细的游戏指导。
                请确保每个查询都关注不同的方面，如技能选择、战斗策略、装备搭配等。
                用户原始问题：{question}
                请给出5个不同的查询，每个占一行。""",
)
# 设定大模型处理管道
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_chain = QUERY_PROMPT | llm | output_parser
# 使用自定义提示模板的MultiQueryRetriever
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(),
    llm_chain=llm_chain,
    parser_key="lines"
)
# 进行多角度查询
query = "那个，我刚开始玩这个游戏，感觉很难，请问这个游戏难度级别如何，有几关，在普陀山那一关，嗯，怎么也过不去。先学什么技能比较好？新手求指导！"
# 调用RePhraseQueryRetriever进行查询分解
docs = retriever.invoke(query)
print(docs)


# HyDE 文档生成模板
template = """请撰写一段与以下问题相关的游戏内容：
问题：{question}
内容："""
prompt_hyde = ChatPromptTemplate.from_template(template)

# 创建生成假设文档的链
llm = ChatDeepSeek(model="deepseek-chat")
generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser()
)

# 生成假设文档
question = "黑神话悟空中的主角有哪些主要技能？"
generated_doc = generate_docs_for_retrieval.invoke({"question": question})
print("\n=== 生成的假设文档 ===")
print(generated_doc)

# 初始化向量存储检索器，检索相关文档
retriever = vectorstore.as_retriever()
retrieval_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain.invoke({"question": question})
print("\n=== 检索到的相关文档 ===")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n文档 {i}:")
    print(doc.page_content)

# 最终回答生成模板
answer_template = """根据以下内容回答问题：
{context}
问题：{question}
回答："""
answer_prompt = ChatPromptTemplate.from_template(answer_template)
# 创建最终的问答链
final_rag_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)
# 获取最终答案
final_answer = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
print("\n=== 最终答案 ===")
print(final_answer)
