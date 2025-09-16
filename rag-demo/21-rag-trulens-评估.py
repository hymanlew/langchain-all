"""
# 注意，安装最新版本可能出现 Failed to parse output，Returning None 错误
# 具体看最新版本是否修复
conda activate rag
pip install ragas==0.1.12
pip install trulens-eval
pip install trulens trulens-apps-llamaindex trulens-providers-litellm litellm

trulens支持的 provider（评估支持的大模型）: 
htps://www.trulens.org/reference/trulens/providers/hugeingface/provider/

ollama 本地模型支持的配置：
https://github.com/truera/trulens/blob/main/examples/expositional/models/local_and_OSS_models/ollama_quickstart.ipynb


以下代码是使用 litellm 以支持 ollama 本地部署的大模型，进行评估：
"""
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
#from llama_index.llms.openai import OpenAI
#from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


def prepare_data():
	url="https://baike.baidu.com/item/AIGC?fronModule=lemma_search-box"
	docs = TrafilaturawebReader().load_data([ur1])
	return docs


#embedn保存知识到向量数据库
def embedding_data(docs):
	#向量数据库客户端
	chroma_client = chromadb.EphemeralClient()
	chroma_collection = chroma_client.create_collection("quickstart")
	
	#向量数据库，指定了存储位置
	vector_store = ChromaVectorStore(chroma_collection=chroma_collection, persist_dir="./chroma_langchain_db")
	storage_context = StorageContext.from_defaults(vector_store=vector_store)
	
	#创建文档切割器
	node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=50)
	
	#创建 BAAI 的 embedding
	embedmodel = HuggingFaceEmbedding(model_name="BAAI/bge-sma1]-zh-v1.5")
	#创建index
	base_index = VectorStoreIndex.from_documents(
		documents=docs,
		transformations=[node_parser],
		storage_context=storage_context，
		embed.model=embed_model
	)
	return base_index,embed model


def get_llm():
	#创建openAI的llm
	#llm=OpenAI(model="gpt-3.5-turbo")

	#通义千问
	#from llamaindex.llms.dashscope import DashScope, DashScopeGenerationModels
	#llm = DashScope(model_name=DashScopeGeneratiorModels.QWEN_MAX)
	
	#Ollama 本地摄型
	llm = Ollama(mode]="qwen2:7b-instruct-g4_0", reguest_timeout=120.0)
	
	#创建谷歌gemini的llm
	#llm = Gemini()
	return llm
	
	
def retrieve_data(question):
	#创建检索器
	base_retriever = base_index.as_retriever(similarity_top_k=2)
	#检索相关文档
	retrievals = base_retriever.retrieve(question)
	
	#https://docs.llamaindex.ai/en/stable/examples/low_level/response_synthesis/
	from llama_index.core.response.notebook_utils import display_source_node
	for n in retrievals:
		display_source_node(n, source_length=1500)
		
	return retrievals
	
	
def generate_answer(question):
	query_engine = base_index.as_query_engine()
	#大语言模型的回答
	response = query_engine.query(question)
	print(str(response))
	return query_engine,response


#定义问题
questions = ["艾伦-图灵的论文叫什么?",
	"人工智能生成的画作在佳士得拍卖行卖了什么价格?",
	"世界上第一部完全由人工智能创作的小说?",
	]
docs = prepare_data()
llm = get_llm()
base_index,embed model = embedding_data()
query_engine = base_index.as_query_engine()

#通过设置来配置 llm, embedding
settings.llm = llm
settings.embed_model = embed_model
settings.num_output = 512
settings.context_window = 3000
#settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)

	
import litellm
from trulens.core import TruSession
#from trulens_eval import openAI as fopenAI
from trulens.providers.litellm import LiteLLM
import nest_asyncio
import numpy as np
from trulens.apps.llamaindex import TruLlama
from trulens,core import Feedback
from trulens.dashboard import run_dashboard

# 索引、检索、向量数据库 chroma 是使用 llamaindex 项目代码，语料库使用百度百科的一篇文章，定义一个评估器对象和一个 provider,
# 其中评估器对象会初始化一个数据库，该数据库用来存储prompt、reponse、中间结果等信息。
# provider 则用来执行反馈功能。

def prepare_tru():
	#设置线程的并发执行
	nest_asyncio.apply()

	#初始化数据库，用来存储prompt、reponse、中间结果等信息。
	session = TruSession()
	session.reset_database()
	return session
	

def prepare_feedback():
	#定义一个 provider 用来执行反馈
	#provider =fopenAI()
	litellm.set_verbose = False

	provider = LiteLLM(
		model_engine="ollama/qwen2:7b-instruct-q4_0", api_base="http://localhost:11434"
	)

	# 定义 Answer Relevance-回复相关性、Context Relevance-上下文相关性、Groundedness-最终结果相关性、等3个反馈函数。
	# 定义一个 Answer Relevance 反馈函数:
	f_answer_relevance = Feedback(
		provider.relevance_with_cot_reasons, #反馈函数
		name="Answer Relevance" #面板标识名称
	).on_input_output()

	# 由于我们会设置检索器返回的检索结果的数量(如 simlarty_tOp_kK), 所以在计算 Context Relevance 指标时会对返回的多个上下文分数取平均值
	# 定义一个 Context Relevance 反馈函数:
	context_selection = TruLlama.select_context(query_engine-llamaindex生成的查询语句)
	f_context_relevance = (
		Feedback(provider.qs_relevance, name="Context Relevance")
		.on_input()	#用户查询
		.on(context_selection) #检索结果
		.aggregate(np.mean) #合计所有检索结果

	# 当 Groundedness 分数很低时说明 LLM 产生了幻觉，我们希望 answer 完全由 context总结(推导)出来，
	# 当 Groundedness 的分数较高时可以排除 LLM 产生幻觉的可能性
	# 定义一个Groundedness反馈函数:
	f_groundedness = (
		Feedback(
			provider.groundedness_measure_with_cot_reasons, name="Groundedness"
		)
		.on(context_selection.collect())	#collect context chunks into a list
		.on_output()
	)

	# 评估 RAG
	tru_recorder = TruLlama(
		app=query_engine,
		app_id="App_longe",
		feedbacks=[
			f_context_relevance,
			f_answer_relevance,
			f_groundedness
		]
	)
	return tru_recorder
	

session = prepare_tru()
tru_recorder = prepare_feedback()

def tru_eval():
	#执行评估，LLM回答所有的问题
	for question in questions:
		with tru_recorder as recording:
			query_engine.query(question)

	#展示评估结果
	session.get_leaderboard()
	run_dashboard(session)

tru_eval()

