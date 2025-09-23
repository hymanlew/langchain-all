import asyncio
import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from typing import List, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.pydantic_v1 import BaseModel, Field
# from display_graph import display_graph


# Load local PDF document
file_path = os.path.join(os.getcwd(), "chapter11", "Faiss by FacebookAI.pdf")
pdf_loader = PyPDFLoader(file_path)
pdf_documents = pdf_loader.load()

# Split PDF into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_pdf_documents = text_splitter.split_documents(pdf_documents)

# 1. Index 3 websites by adding them to a vector DB
urls = [
    "https://github.com/facebookresearch/faiss",
    "https://github.com/facebookresearch/faiss/wiki",
    "https://github.com/facebookresearch/faiss/wiki/Faiss-indexes"
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
split_web_documents = text_splitter.split_documents(docs_list)

all_documents = split_pdf_documents + split_web_documents
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
vectorstore = Chroma.from_documents(
    documents=all_documents,
    collection_name="rag-chroma",
    embedding=embeddings,
)
print("All documents indexed in Chroma successfully.")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks


# 3. define the graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]


# Retrieval Grader setup
class GradeDocuments(BaseModel):
    # 文件是否与问题相关
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


# Answer Grader setup
class GradeAnswer(BaseModel):
    # 答案是否正确
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


# Hallucination Grader setup
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the documents, 'yes' or 'no'")


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retrieval_prompt = ChatPromptTemplate.from_template("""
You are a grader assessing if a document is relevant to a user's question.
Document: {document} 
Question: {question}
Is the document relevant? Answer 'yes' or 'no'.
""")
retrieval_grader = retrieval_prompt | model.with_structured_output(GradeDocuments)

# 2. Prepare the RAG chain
prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

hallucination_prompt = ChatPromptTemplate.from_template("""
You are a grader assessing if an answer is grounded in retrieved documents.
Documents: {documents} 
Answer: {generation}
Is the answer grounded in the documents? Answer 'yes' or 'no'.
""")
hallucination_grader = hallucination_prompt | model.with_structured_output(GradeHallucinations)

answer_prompt = ChatPromptTemplate.from_template("""
You are a grader assessing if an answer addresses the user's question.
Question: {question} 
Answer: {generation}
Does the answer address the question? Answer 'yes' or 'no'.
""")
answer_grader = answer_prompt | model.with_structured_output(GradeAnswer)


# 4. Retrieve node
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Grades documents based on relevance to the question.
    Only relevant documents are retained in 'relevant_docs'.
    """
    question = state["question"]
    documents = state["documents"]
    relevant_docs = []

    for doc in documents:
        response = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if response.binary_score == "yes":
            relevant_docs.append(doc)

    return {"documents": relevant_docs, "question": question}

# 5. Generate node
def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state):
    """
    Rephrases the query for improved retrieval if initial attempts do not yield relevant documents.
    """
    transform_prompt = ChatPromptTemplate.from_template("""
    You are a question re-writer that converts an input question to a better version optimized for retrieving relevant documents.
    Original question: {question} 
    Please provide a rephrased question.
    """)
    question_rewriter = transform_prompt | model | StrOutputParser()
    question = state["question"]
    # Rephrase the question using LLM
    transformed_question = question_rewriter.invoke({"question": question})
    return {"question": transformed_question, "documents": state["documents"]}

def decide_to_generate(state):
    """
    Decides whether to proceed with generation or transform the query.
    """
    if not state["documents"]:
        return "transform_query"  # No relevant docs found; rephrase query
    return "generate"  # Relevant docs found; proceed to generate

def grade_generation_v_documents_and_question(state):
    """
    Checks if the generation is grounded in retrieved documents and answers the question.
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Step 1: Check if the generation is grounded in documents
    hallucination_check = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if hallucination_check.binary_score == "no":
        return "not supported"  # Regenerate if generation isn't grounded in documents

    # Step 2: Check if generation addresses the question
    answer_check = answer_grader.invoke({"question": question, "generation": generation})
    return "useful" if answer_check.binary_score == "yes" else "not useful"


# 6. Define the workflow
def create_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)

    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"}
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {"not supported": "generate", "useful": END, "not useful": "transform_query"}
    )
    return workflow.compile(checkpointer=MemorySaver())


# 7. Run the workflow
async def run_workflow():
    app = create_workflow()
    config = {
        "configurable": {"thread_id": "1"},
        "recursion_limit": 50
    }

    inputs = {"question": f"What are flat indexs?"}
    try:
        async for event in app.astream(inputs, config=config, stream_mode="values"):
            if "error" in event:
                print(f"Error: {event['error']}")
                break
            print(event)
    except Exception as e:
        print(f"Workflow execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_workflow())

