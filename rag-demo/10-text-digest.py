import os

"""
文本自动摘要
对一组文档 (PDF、页面、客户问题等) 进行内容总结。鉴于大型语言模型 LLMS 在理解和综合文本方面的熟练程度，
所以它是完成这项任务的绝佳工具。

总结或组合文档的三种方式：
1. 填充(Stuff)，它简单地将文档连接成一个提示。只适用小文本，且不超过大模型一次处理的 token 数。因为它是一次性读取并处理的;
2. 映射-归约(Map-reduce)，将文档分成批次，总结这些批次，然后总结这些总结。适用于大型文本，几十万上百万的文字。
3. 细化(Refine)，通过顺序迭代文档来更新滚动摘要，类似于 map-reduce

整体流程可简化为：文档切片 - prompt - LLM（分别处理每一个切片文档）- LLM（将切片摘要分批处理）- LLM（直到最终摘要结果）
pip install tiktoken chromadb
"""
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# 加载一个在线文档。使用 WebBaseLoader 来加载博客文章：
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章

# Stuff 的第一种写法
# chain = load_summarize_chain(model, chain_type='stuff')

# Stuff 的第二种写法
# 定义提示
prompt_template = """针对下面的内容，写一个简洁的总结摘要:
"{text}"
简洁的总结摘要:"""
prompt = PromptTemplate.from_template(prompt_template)

# LLMChain 会提示过期提醒，但是它跟下面是绑定的，所以不建议改
llm_chain = LLMChain(llm=model, prompt=prompt)

# 注意 Stuff 方式如果接收数据大于模型的最大接收数时，它会自动截断，所以摘要的就不准确
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='text')
result = stuff_chain.invoke(docs)
print(result['output_text'])

# -------------------------------------

# 第二种：Map-Reduce
# 第一步：切割阶段，每一个小 docs 为 1000 个token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 第二步： map阶段
map_template = """以下是一组文档(documents)
"{docs}"
根据这个文档列表，请给出总结摘要:"""
map_prompt = PromptTemplate.from_template(map_template)
map_llm_chain = LLMChain(llm=model, prompt=map_prompt)

# 第三步： reduce 阶段: 组合 combine 和 最终的 reduce
reduce_template = """以下是一组总结摘要:
{docs}
将这些内容提炼成一个最终的、统一的总结摘要:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)

'''
reduce 的思路:
如果 map 之后文档的累积 token 数超过了 4000个，那就要递归地将文档以 <=4000 个token，按批次传递给
StuffDocumentsChain 来创建批量摘要。
一旦这些批量摘要的累积大小小于 4000 个token，就将它们全部传递给 StuffDocumentsChain 最后一次，以创建最终摘要。
'''
combine_chain = StuffDocumentsChain(llm_chain=reduce_llm_chain, document_variable_name='docs')
reduce_chain = ReduceDocumentsChain(
    # 最终调用的链
    combine_documents_chain=combine_chain,
    # 中间汇总的链
    collapse_documents_chain=combine_chain,
    # 将文档分组的最大令牌数
    token_max=4000
)

# 第四步：合并所有链
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_llm_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name='docs',
    return_intermediate_steps=False
)

# 第五步： 调用最终的链
result = map_reduce_chain.invoke(split_docs)
print(result['output_text'])

# ---------------------------------------

# 第三种：Refine
'''
Refine: RefineDocumentsChain 类似于 map-reduce：
文档链通过循环遍历输入文档并逐步更新其答案来构建响应。对于每个文档，它将当前文档和最新的中间答案传递给LLM链，以获得新的答案。
'''
# 第一步： 切割阶段，每一个小 docs 为 1000 个token
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 指定chain_type为： refine
chain = load_summarize_chain(model, chain_type='refine')

result = chain.invoke(split_docs)
print(result['output_text'])

# --------------------------------------

# 第三种：带提示模板的 refine 方式（不常用，最多的是第二种，map-reduce）
# 定义提示
prompt_template = """针对下面的内容，写一个简洁的总结摘要:
"{text}"
简洁的总结摘要:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "\n"
    "Given the new context, refine the original summary in Chinese"
    "If the context isn't useful, return the original summary."
)
# refine_template = (
#     "你的工作是做出一个最终的总结摘要。\n"
#     "我们提供了一个到某个点的现有摘要:{existing_answer}\n"
#     "我们有机会完善现有的摘要，基于下面更多的文本内容\n"
#     "------------\n"
#     "{text}\n"
#     "------------\n"
# )
refine_prompt = PromptTemplate.from_template(refine_template)

chain = load_summarize_chain(
    llm=model,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=False,
    input_key="input_documents",
    output_key="output_text",
)

result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
print(result["output_text"])
