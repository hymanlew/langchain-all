import re
from operator import itemgetter
from langchain.load import dumps, loads
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda

"""
**查询重写** `multi-query`：换种说法（可以重写3遍 or 5遍），表达查询意图。
**查询融合** `RAG-fusion`：将多个查询的关联文档进行融合（去重、Ranking Fusion），将最相关的文档排在最前面，输入给 LLM，获取最终答案。
**子查询** `sub-question`：复杂查询，依赖 LLM 生成多个子查询，然后分别检索，最后合并结果。
**后退查询** `step-back query`：将原始查询，转换为更通用的查询，然后检索，获取关联文档，输入给 LLM，获取最终答案。
**假设性文档嵌入** `HYDE`：让 llm 先生成一份书面的回答（`假设性回答`），并以此作为`查询嵌入`后，获取对应关联文档；再用 `原始查询` + 关联文档，获取最终生成的内容。
"""
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# remove <think> part in the text
def remove_think_tags(text):
    """remove <think> part in the text"""
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    return cleaned_text.strip()


# ------------------------ 查询重写 Multi Query -------------------------------
rewrite_template = """请将以下用户查询重写为更清晰、正式、易于检索的版本。
保持原意不变，但使其更加具体和完整。

原始查询: {question}

要求：
1. 保持核心意图不变
2. 补充可能的隐含信息
3. 使用更规范的表达方式
4. 如果查询模糊，请使其更具体

重写后的查询:"""

# 使用 LLM 重写查询语句（包含 Role、Goal、Constraints），返回多个查询语句
rewrite_template2 = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_rewrite = ChatPromptTemplate.from_template(rewrite_template2)

generate_queries = (
    prompt_rewrite
    | llm
    | StrOutputParser()
    | remove_think_tags
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# 使用重写得到的 5 个 Query，分别检索 Retrieve，并将关联文档进行`去重`：
# `dumps`、`loads`：LangChain 的序列化工具，对象转换为 JSON 字符串，并反序列化回来。
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})

# 使用上面得到的关联文档，输入给 LLM，获取最终答案：
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# itemgetter 用于排序或从序列中提取元素，但在这里它被用来提取字典中特定键的值。
final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)
final_rag_chain.invoke({"question":question})


# ------------------------ 查询融合 RAG-fusion -------------------------------
# 直接复用上面的 查询重写，或使用 LangChain Hub 的 rag-fusion-query-generation 模型
# from langchain import hub
# prompt = hub.pull("langchain-ai/rag-fusion-query-generation")

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)

            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]

            # Update the score of the document using the RRF formula: 1 / (rank + k)
            # The core of RRF: documents ranked higher (lower rank value) get a larger score
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples,
    # each containing the document and its fused score
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
# docs = retrieval_chain_rag_fusion.invoke({"question": question})

template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
        {"context": retrieval_chain_rag_fusion,
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
)
final_rag_chain.invoke({"question": question})


# ------------------------ 子查询 Sub-Question -------------------------------
# 适用场景：有些复杂问题包含了多个子问题，无法在一个步骤中解决。例如，What are the main components of an LLM-powered agent, and how do they interact? 这实际就是 2 个问题。
# 实际上，查询拆解为多个子查询后，不同的查询之间，可能存在 2 类关系：前后依赖、相互独立。
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_sub_question = ChatPromptTemplate.from_template(template)

generate_queries_chain = (
            prompt_sub_question
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | remove_think_tags)

question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_chain.invoke({"question": question})

# 使用上面得到的 questions，分别检索，并使用 RAG 获取答案
from langchain import hub
prompt_rag = hub.pull("rlm/rag-prompt")

def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """RAG on each sub-question"""
    sub_questions = sub_question_generator_chain.invoke({"question": question})

    # Initialize a list to hold RAG chain results
    rag_results = []

    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)

        # Use retrieved documents and sub-question in RAG chain
        answer = ((prompt_rag | llm | StrOutputParser() | remove_think_tags)
                  .invoke({"context": retrieved_docs, "question": sub_question}))
        rag_results.append(answer)
    return rag_results, sub_questions

# Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_chain)

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

context = format_qa_pairs(questions, answers)

template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
)
final_rag_chain.invoke({"context": context, "question": question})


# -------------------------- 后退查询 Step Back ---------------------------------
# 将原始查询`后退`一步，重新构造查询，然后检索，获取答案。
# 对原始查询进行概念和原则的抽象化处理，从而引导更加深入的推理过程；一般会去掉不必要的细节，从而引导更加深入的推理过程
# 采用 小样本学习（few-shot），来引导 LLM 进行后退查询。
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
# step-back chain
generate_queries_step_back = prompt | llm | StrOutputParser() | remove_think_tags

# 使用上面得到的 query，进行检索，并使用 RAG 获取答案
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)
chain.invoke({"question": question})


# ------------------------ HYDE 假设性文档嵌入 -------------------------------
"""
HyDE 解决的主要问题是：查询相关文档时，查询用语 跟 文档内容之间，存在术语不统一、词表不一致的问题。
HyDE 工作原理：在标准的检索流程前，增加了一个关键的“想象”步骤：

1. 生成假设性文档：
- 接收到用户查询后，不直接用它去检索。而是先将查询交给一个 指令执行能力强的生成式 LLM，让 LLM 凭空生成一个假设的、可能不准确的答案。
2. 嵌入假设性文档：将这个生成的、假设性的答案（即假设性文档）通过嵌入模型转换为一个稠密向量。

3. 用假设性向量进行检索：
- 使用这个假设性文档的向量（而不是原始查询的向量）去向量数据库中进行相似性搜索。
- 检索出与这个假设性文档最相似的真实文档。

4. 标准生成步骤：
- 将检索到的真实文档和原始问题一起交给生成器，生成最终的真实、准确的答案。
"""
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="请为以下问题生成一个假设性的答案段落。即使不确定，也请写出一个可能合理的答案：\n\n问题：{question}"
)

# 创建生成假设文档的链
# LLMChain 还可以通过合适的提示模板实现查询重写功能，进行查询重写，
# 将用户的原始、模糊或不完整的查询，重写成一个更清晰、更正式、更易于检索的查询
hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)

generate_hyde_chain = (
    prompt_hyde
    | llm
    | StrOutputParser()
    | remove_think_tags
)

def get_relevant_documents(self, question):
    # Step 1: 生成假设文档
    hypothetical_doc = hyde_chain.run(question)
    # Step 2: 使用假设文档的向量进行检索
    docs = vectorstore.similarity_search(hypothetical_doc, k=5)
    return docs

def get_relevant_final(self, question):
    # 使用上面得到的假设性文档，进行检索，并使用 RAG 获取答案
    retrieval_chain = generate_hyde_chain | retriever
    retrieved_docs = retrieval_chain.invoke({"question": question})

    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
            prompt
            | llm
            | StrOutputParser()
    )
    docs = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
    return docs


# 5. 使用 HyDE 检索器
hyde_retriever = HyDERetriever(vectorstore, hyde_chain)

# 6. 现在，您可以将 hyde_retriever 用于您的 QA 链
# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=hyde_retriever, ...)
# answer = qa_chain.run("什么是光合作用？")
