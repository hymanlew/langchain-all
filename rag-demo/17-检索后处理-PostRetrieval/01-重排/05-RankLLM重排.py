from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
import torch
# 加载文档并进行分割
documents = TextLoader("data/山西文旅/云冈石窟.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx
# 生成嵌入并创建检索器
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh") 
retriever = FAISS.from_documents(texts, embed_model).as_retriever(search_kwargs={"k": 20})

# # 清理GPU缓存
# torch.cuda.empty_cache()
# 设置RankLLM重排序器
compressor = RankLLMRerank(top_n=3, model="gpt", gpt_model="gpt-4o-mini")
# 创建上下文压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
# 执行查询并获取重排后的文档
query = "云冈石窟有哪些著名的造像？"
compressed_docs = compression_retriever.invoke(query)
# 输出结果
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
pretty_print_docs(compressed_docs)
# # 清理模型
# del compressor
