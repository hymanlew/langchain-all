from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

"""
工作原理：HyDE 在标准的检索流程前，增加了一个关键的“想象”步骤：

1. 生成假设性文档：
- 接收到用户查询后，不直接用它去检索。而是先将查询交给一个 指令执行能力强的生成式 LLM，让 LLM 凭空生成一个假设的、可能不准确的答案。
2. 嵌入假设性文档：将这个生成的、假设性的答案（即假设性文档）通过嵌入模型转换为一个稠密向量。

3. 用假设性向量进行检索：
- 使用这个假设性文档的向量（而不是原始查询的向量）去向量数据库中进行相似性搜索。
- 检索出与这个假设性文档最相似的真实文档。

4. 标准生成步骤：
- 将检索到的真实文档和原始问题一起交给生成器，生成最终的真实、准确的答案。
"""
# 1. 初始化组件
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# 手动定义查询重写的提示模板
rewrite_template = """请将以下用户查询重写为更清晰、正式、易于检索的版本。
保持原意不变，但使其更加具体和完整。

原始查询: {question}

要求：
1. 保持核心意图不变
2. 补充可能的隐含信息
3. 使用更规范的表达方式
4. 如果查询模糊，请使其更具体

重写后的查询:"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=rewrite_template
)

# 2. 定义生成假设文档的提示模板
hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="请为以下问题生成一个假设性的答案段落。即使不确定，也请写出一个可能合理的答案：\n\n问题：{question}"
)

# 3. 创建生成假设文档的链
# LLMChain 还可以通过合适的提示模板实现查询重写功能，进行查询重写，
# 将用户的原始、模糊或不完整的查询，重写成一个更清晰、更正式、更易于检索的查询
hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)


# 4. 创建 HyDE 检索器（这里需要一些自定义逻辑，但思路如下）
class HyDERetriever:
    def __init__(self, vectorstore, hyde_chain):
        self.vectorstore = vectorstore
        self.hyde_chain = hyde_chain

    def get_relevant_documents(self, question):
        # Step 1: 生成假设文档
        hypothetical_doc = self.hyde_chain.run(question)
        # Step 2: 使用假设文档的向量进行检索
        docs = self.vectorstore.similarity_search(hypothetical_doc, k=5)
        return docs


# 5. 使用 HyDE 检索器
hyde_retriever = HyDERetriever(vectorstore, hyde_chain)

# 6. 现在，您可以将 hyde_retriever 用于您的 QA 链
# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=hyde_retriever, ...)
# answer = qa_chain.run("什么是光合作用？")
