from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.synthesizers import default_query_distribution
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# 使用 TestsetGenerator 生成大规模、多样化的测试集，对系统进行压力测试和整体评分。
# 它能模拟真实世界的查询多样性，帮你全面评估系统的抗压能力和整体表现
# 官方文档：https://docs.ragas.org.cn/en/stable/getstarted/rag_testset_generation/#knowledgegraph-creation
# 官方文档：https://docs.ragas.org.cn/en/stable/howtos/applications/singlehop_testset_gen/#query-generation-using-synthesizers

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

# 创建默认的查询分布
query_distribution = default_query_distribution(generator_llm)

# 如果你想自定义不同类型问题的比例，可以调整合成器的权重
# 以下是默认分布，你可以根据需要调整
# from ragas.testset.synthesizers import (
#     SingleHopSpecificQuerySynthesizer,
#     MultiHopAbstractQuerySynthesizer,
#     MultiHopSpecificQuerySynthesizer
# )
# query_distribution = {
#     SingleHopSpecificQuerySynthesizer: 0.5,   # 单跳具体问题：50%
#     MultiHopAbstractQuerySynthesizer: 0.25,   # 多跳抽象问题：25%
#     MultiHopSpecificQuerySynthesizer: 0.25,   # 多跳具体问题：25%
# }

# 使用自定义的查询分布生成测试集
test_dataset_diverse = generator.generate(
    testset_size=10,
    query_distribution=query_distribution
)

# 生成 user_input=question, reference_contexts，reference=真实参考答案，
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
dataset.to_pandas()


# 使用测试集评估你的 RAG 系统，并假设 RAG 系统封装成了一个函数 `my_rag_chain`
my_rag_chain = None
questions = test_dataset_diverse.user_input.values

answers = []
contexts_list = [] # 注意：这里准备存储每个问题对应的多个上下文文档

for question in questions:
    # 假设 my_rag_chain 返回答案和检索到的上下文列表
    answer, contexts = my_rag_chain.invoke(question)
    answers.append(answer)
    contexts_list.append(contexts) # 注意：这里每个问题的contexts是一个列表

# 将你的RAG系统的输出构造成Ragas评估所需的数据集
evaluation_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts_list, # 注意：这里传入的是contexts_list
    "ground_truths": test_dataset_diverse.reference.values
}
evaluation_dataset = EvaluationDataset.from_list([evaluation_data])

metrics = [
    faithfulness,       # 忠实度：答案是否基于给定上下文
    answer_relevancy,   # 答案相关性：答案与问题的匹配程度
    context_precision,  # 上下文精确度：检索到的上下文是否相关
    context_recall,     # 上下文召回率：是否检索到了所有必要信息:cite[10]
]
evaluation_result = evaluate(evaluation_dataset, metrics=metrics)
evaluation_result.to_pandas()
