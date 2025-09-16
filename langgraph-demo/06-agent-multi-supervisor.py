"""
简单逻辑的或少量 AGENT，使用协作 AGENT
当有大量任务 agent 时，使用主管 AGENT
当有更多 agent 时，一个主管 AGENT 也不够了，就要使用分层多 Agent(Hierarchical Agent)

参考: https://github.com/langchaln.ai/langgraph/blob/maln/docs/docs/tutorlals/multil_agent/hierarchical_agent_teams.ipynb
在前面 Agent 主管中，引入了单个主管节点的概念，以在不同的工作节点之间路由工作。但如果单个工人的工作变得过于复杂，工人人数太多怎么办？对于某些应用程序，
如果工作是分层分布的，系统可能会更有效。
可以通过组合不同的子图并创建顶级主管和中级主管来实现这一点。我们称之为分层团队。因为子Agent 在某种程度上可以被视为团队，并由主管 Agent 将他们连接起来。


代码说明：
operator：Python 标准库，提供运算符的函数式接口（如 operator.add 对应 + 的操作）

typing：Python 类型提示支持库。
	Annotated 用于添加元数据的类型注解，
		def read_document(
			file_name: Annotated[
				str,
				Field(..., description="File path", regex=r"^/data/.*\.txt$")
			]
		) -> str: 可以直接在类型注解中说明参数用途，此时 file_name 不只是表示字符串，而且还说明了用途，及格式校验
		
	Literal 用于创建字面量类型（如 Literal["a","b"] 表示值只能是 "a"或"b"）。

functools.partial 作用是冻结函数的部分参数，生成一个参数更少的新函数。
	def power(base, exponent): return base ** exponent
	power(2, 3) = 8，power(2, 4) = 16，传统调用方式：每次需传两个参数
	square = partial(power, exponent=2) 新方式固定 base=2，创建新函数 square，相当于 power(base, 2)
	square(5) 调用时传一个参数即可，正常计算得 (即5²)
	research_node = functools.partial(agent_node_run, agent=research_agent, name="webscraper")
	创建了一个新函数 research_node，它只需要接收state参数，因为agent和name已被固定。


代码例子：
定义Agent访问web和写入文件的工具，定义一些实用程序来帮助创建图形和Agent
创建和定义每个团队(网络研究+文档写作)，把一切都组合在一起。
每个团队将由一名或多名gent组成，每个Agent都有一个或多个工具。
"""
#------------------ 公共服务函数 -----------------------

llm = ChatOpenAI(model="gpt-4o-mini")

# 消息历史裁剪器，用于控制输入模型的 token 数量（避免超出上下文窗口限制）
trimmer = trim_messages(
    max_tokens=100000,      # 允许的最大总token数
    strategy="last",        # 裁剪策略：保留最后N条消息
    token_counter=llm,      # 使用llm的token计数方法
    include_system=True     # 是否计入系统消息
)

'''
state 参数是在执行时由框架（langgraph 图执行器）传入的，通常来自前一个节点的输出或初始输入，上个节点当前状态/上下文数据
不是由 LLM 处理的，而是通过图的执行流程管理
并且在有向图（Graph）执行流程中，节点间的参数传递和类型转换，也是由图执行引擎自动处理。

统一接口：大多数图执行引擎要求所有节点函数遵循统一的接口规范（如都接收 state: dict 参数，这里是 TypedDict）。
状态封装：所有参数会被封装到一个共享的 state 对象中（通常是字典或特定类实例），执行引擎会自动传递完整的 state，节点按需从中提取自己需要的字段。且类型安全也是由节点自己处理。
类型转换：高级框架（如使用Pydantic的LCEL）会自动验证和转换类型，但需预先定义好Schema。

Pydantic 是 Python 中广泛使用的数据验证库，通过继承 BaseModel 定义 Schema，自动处理类型转换和验证
from pydantic import BaseModel, ValidationError
class UserSchema(BaseModel):
    id: int
    name: str = "default_name"
    email: str  # 必填字段
    age: int | None = None  # 可选字段

# 使用Schema验证数据
try:
    user = UserSchema(id="123", email="test@example.com")  # 自动将字符串id转为int
    print(user.dict())  # 输出: {'id': 123, 'name': 'default_name', 'email': 'test@example.com', 'age': None}
except ValidationError as e:
    print(e.json())  # 验证失败时输出错误详情
'''
def agent_node_run(state, agent, name):
	result = agent.invoke(state)
	return {
		"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
	}
	

#主管定义函数，可以创建子团队的主管，也可以创建大主管
#定义一个主管，并执行路由操作，返回下一节点的节点名
def create_team_supervisor(llm: ChatOpenAl, system_prompt, members) -> str:
	"""An LLM-based router ."""
	options = ["FINISH"] + members
	"""
	这里不需要实际定义 route 函数。这里的 route 只是一个工具调用的标识符，而非需要实现的 Python 函数。
	- 首先 name 字段仅描述模型可以调用的工具结构（名称、参数、描述），
	- 重要的是模型决策：LLM 根据对话上下文选择是否调用工具（如返回 {"next": "FINISH"}）
	- 是依据 description 字段 + 对话上下文（例如 prompt）来决策
	- 另外 JsonOutputToolsParser() 最后解析模型返回的 JSON 结构，不会实际执行函数
	"""
	function_def = {
		"type": "function",
        "function": {
			"name": "route",
            "description": "Select the next role to act or FINISH the conversation",
            "parameters": {
                "type": "object",
                "properties": {
                    "next": {
                        "type": "string",
                        "enum": options,
                        "description": "The next team member to act or FINISH"
                    }
                },
                "required": ["next"],
            },
		},
	}
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			Messagesplaceholder(variable_name="messages"),
			(
				"system",
				"Given the conversation above, who should act next? or should we FINISH? Select one of: {options}",
			)
		]
	).partial(options=", ".join(options))
	return (
		prompt | trimmer
		| llm.bind_tools(tools=[function_def], tool_choice="route")
		| JsonOutputFunctionsParser()
	)



#---------------- 建立访问web 网络研究团队 --------------------

import functools
import operator
from angchaincore.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent

#需要 api_key, 不然会报错
tavi1y_tool =TavilySearchResults(max_results=5)

# 网页转 doc 工具
@tool
def scrape_webpages(urls: List[str]) -> str:
	"""Use requests and bs4 to scrape the provided web pages for detailed information."""
	loader = webBaseLoader(ur1s)
	docs = loader.load()
	return "\n\n".join(
		[
			f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
			for doc in docs
		]
	)

#ResearchTeam graph state
class ResearchTeamState(TypedDict):
	#A message is added after each team member finishes
	messages: Annotated[List[BaseMessage], operator.add]
	#The team members are tracked so they are aware of the others's ki1l-sets
	team_members: List[str]
	#Used to route work. The supervisor calls a function that will update this every time it makes a decision
	next: str


search_agent = create_react_agent(llm, tools=[tavily_tool])
search_node = functools.partial(agent_node_run, agent=search_agent, name="Search")

research_agent = create_react_agent(llm, tools=[scrape_webpages])
research_node = functools.partial(agent_node_run, agent=research_agent, name="webscraper")

# 创建子团队主管
supervisor_agent = create_team_supervisor(
	llm,
	"You are a supervisor tasked with managing a conversation between the following workers: Search, webScraper."
	"Given the following user request, respond with the worker to act next, Each worker will perform a task and "
	"respond with their results and status. when finished, respond with FINISH.",
	["Search", "webScraper"],
)
research_graph = StateGraph(ResearchTeamstate)
research_graph.add_node("Search", search_node)
research_graph.add_node("webScraper", research_node)
research_graph.add_node("supervisor", supervisor_agent)

# Define the control flow
research_graph.add_edge("Search", "supervisor")
research_graph.add_edge("webScraper", "supervisor")
research_graph.add_conditiona1_edges(
	"supervisor",
	lambda x: x["next"],
	{"Search":"Search", "webScraper": "webScraper", "FINISH": END},
)
research_graph.add_edge(START, "supervisor")
research_graph_chain = research_graph.compile()


# The following functions interoperate between the top level graph state and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def msg_func(message: str):
	results = {
		"messages": [HumanMessage(content=message)]
	}
	return results
	
research_chain = msg_func | research_graph_chain

#可以显示 graph 结构
from IPython.display import Image, display
display(Image(research_graph_chain.get_graph(xray=True).draw_mermaid_png()))

#运行研究团队
for s in research_chain.stream(
	"when is Taylor Swift's next tour?",
	{"recursion_limit": 100}
):
	if "_end_" not in s:
		print(s)
		print("---")



#------------------- 建立文档团队 -------------------------
'''
使用类似的方法创建下面的文档编写团队。我们将为每个Agent提供对不同文件写入工具的访问权限。
请注意，这里为 Agent 提供文件系统访问权限，这在所有情况下都是不安全的。
'''
import operator
from pathlib import Path
from typing import Annotated, List
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import too1
from pathlib import path
from tempfile import TemporaryDirectory
from typing import Dict, optional
from langchain_experimental.utilities import pythonREPL
from typing_extensions import TypedDict
from typing import list, optional
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import chatPromptTemplate, Messagesplaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph. START
from langchain_core.messages import HamanMessage, trim_messages


TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(TEMP_DIRECTORY.name)

# 生成大纲工具
@tool
def create_outline(
	points: Annotated[List[str], "List of main points or sections."],
	file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "path of the saved outline file."]:
	"""Create and save an outline."""
	with(WORKING_DIRECTORY/file_name).open("w") as file:
		for i, point in enumerate(points):
			file.write(f"{i + 1}, {point}\n")
	return f"outline saved to {file_name}"
	
	
# 文档阅读工具
@tool
def read_document(
	file_name: Annotated[str, "File path to save the document."],
	start: Annotated[optional[int], "The start line. Default is 0"]= None,
	end: Annotated[optional[int], "The end line. Default is None"]= None,
) -> str:
	"""Read the specified document."""
	with(WORKING_DIRECTORY/file_name).open("r") as file:
		lines = file.readlines()
	if start is not None:
		start = 0
	return "\n".join(lines[start:end])
	
	
# 写文档工具
@tool
def write_document(
	content: Annotated[str, "Text content to be written into the document ."],
	file_name: Annotated[str, "File path to save the document ."],
) -> Annotated[str, "Path of the saved document file."]:
	"""Create and save a text document."""
	with(WORKING_DIRECTORY/file_name).open("w") as file:
		file.write(content)
	return f"Document saved to {file_name}"
	

# 编辑文档工具
@tool
def edit_document(
	file_name: Annotated[str, "path of the document to be edited."],
	inserts: Annotated[
		Dict[int, str],
		"Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
	],
) -> Annotated[str, "path of the edited document file,"]:
	"""Edit a document by inserting text at specific line numbers."""
	with(WRKING_DIRECTORY/file_name).open("r") as file:
		lines = file.readlines()
		
	sorted_inserts = sorted(inserts.items())
	for line_number, text in sorted_inserts:
		if 1 <= line_number <= len(lines) - 1:
			lines.insert(line_number-1, text + "\n")
		else:
			return f"Error: Line number {line_number} is out of range."
			
	with(WORKING_DIRECTORY/file_name).open("w") as file:
		file.writelines(lines)
		
	return f"Document edited and saved to {file_name}"
	

#Document writing team graph state
class DocWritingState(TypedDict):
	# This tracks the team's conversation internally
	messages: Annotated[List[BaseMessage], operator.add]
	# This provides each worker with context on the others' skill sets
	team_members: str
	#This is how the supervisor tells langgraph who to work next
	next: str
	# This tracks the shared directory state
	current_files: str
	

# warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()

# 代码执行工具
@tool
def python_repl(
	code: Annotated[str, "The pythan code to execute to generate your chart ."],
):
	"""use this to execute python code, If you want to see the output of a value,you should print it out with “print(...)`, This is visible to the user."""
	try:
		result = repl.run(code)
	except BaseException as e:
		return f"failed to execute. Error: {repr(e)}"
		
	return f"Successfully executed:\n'''pythoa\n{code}\n'''\nstdout: {result}"


#这将在每个 worker agent开始工作之前运行，以使他们更加了解当前的状态，及工作目录。
# This will be run before each worker agent begins work，It makes it so they are more aware of the current state# of the working directory.
def prelude(state):
	written_files = []
	if not WORKING_DIRECTORY.exists():
		WORKING_DIRECTORY.mkdir()
	try:
		written_files = [
			f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*”)
		]
	except Exception:
		pass
	if not written_files:
		return {**state, "current_files": "No files written."}
	return {
		**state,
		"current_files": "\nBelow are files your team has written to the directory:\n" + 
		"\n".join([f"- {f}" for f in written_files]),
	}


# Injects current direry working state before each ca11
doc_writer_agent = create_react_agent(llm, tools=[write_document, edit_document, read_document])
context_aware_doc_writer_agent = prelude | doc_writer_agent
doc_writing_node = functools.partial(
	agent_node_run, agent=context_aware_doc_writer_agent, name="Docwriter"
)

note_taking_agent = create_react_agent(llm, tools=[create_outline, read_document])
context_aware_note_taking_agent = prelude | note_taking_agent
note_taking_node = functools.partial(
	agent_node_run, agent=context_aware_note_taking_agent, name="NoteTaker"
)

chart_generating_agent = create_react_agent(llm, tools=[read_document, python_repl])
context_aware_chart_generating_agent = prelude | chart_generating_agent
chart_generating_node = functools.partial(
	agent_node_run, agent=context_aware_note_taking_agent, name="chartGenerator"
)


#创建子团队主管
'''
你是一名主管，负责管理以下 workers: {team_members}。给定以下用户请求:
与worker一起响应以采取下一步行动。
每个worker将执行一个任务并回复其结果和状态。
完成后，用FINISH回复。
'''
doc_writing_supervisor = create_team_supervisor(
	llm,
	"You are a supervisor tasked with managing a conversation between the following workers: {team_members}, "
	"Given the following user request. respond with the worker to act next, Each worker will perform a task and respond "
	"with their resuits and status. when finished, respond with FINISH.",
	["Docwriter", "NoteTaker", "ChartGenerator"],
)

#添加节点，添加边
# Note that we have unrolled the loop for the sake of this doc
authoring_graph = StateGraph(DocWritingState)
authoring_graph.add_node("Docwriter", doc_writing_node)
authoring_graph.add_node("NoteTaker", note_taking_node)
authoring_graph.add_node("chartGenerator", chart_generating_node)
authoring_graph.add_node("supervisor", doc_writing_supervisor)

#Add the edges that always occur
authoring_graph.add_edge("Docwriter", "supervisor")
authoring_graph.add_edge("NoteTaker", "supervisor")
authoring_graph.add_edge("chartGenerator", "supervisor")

#Add the edges where routing applies
authoring_graph.add_conditiona1_edges(
	"supervisor",
	lambda x: x["next"],
	{
		"Docwriter": "Docwriter",
		"NoteTaker": "NoteTaker",
		"ChartGenerator": "chartGenerator",
		"FINISH": END,
	},
)

authoring_graph.add_edge(START, "supervisor")
authoring_graph_chain = authoring_graph.compile()


#以下函数在顶层图形状态之间进行互操作, 以及研究子图的状态
#这使得每个图的状态不会混合在一起
# The following functions interoperate between the top level graph state# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def msg_func_doc(message: str, members: List[str]):
	results = {
		"messages": [HumanMessage(content=message)]
		"team_members": ",".join(members),
	}
	return results

#We reuse the enter/exit functions to wrap the graph
authoring_chain = (
	functools.partial(msg_func_doc, members=authoring_graph.nodes) | authoring_graph_chain
)

#运行文档团队
for s in authoring_chain.stream(
	"write an outline for poem and then write the poem to disk.",
	{"recursion_limit": 100},
):
	if"_end_" not in s:
		print(s)
		print("---")



#-------------------- 添加主管层 -------------------------
'''
在这个设计中，我们执行自上而下的规划策略。我们已经创建了两个图，但必须决定如何在两者之间路由工作。
我们将创建第三个图来编排前两个图，并添加一些连接器来定义如何在不同图之间共享此顶级状态。
'''
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI


#创建大主管
supervisor_node = create_team_supervisor(
	llm,
	"You are a supervisor tasked with managing a conversation between the following teams: {team_members}, "
	"Given the following user request, respond with the worker to act next. Each worker will perform a task "
	"and respond with their results and status. when finished, respond with FINISH.",
	["ResearchTeam", "PaperWritingTeam"],
)

#Top-level graph state
class State(TypedDict):
	messages: Annotated[List[BaseMessage], operator.add]
	next: str
	
	
def get_last_message(state: State) -> str:
	return state["messages"][-1].content
	
def get_result_message(response: dict):
	return {"messages": [response["messages"][-1]]}
	
# Define the graph.
super_graph = StateGraph(State)
#First add the nodes, which will do the work
super_graph.add_node("ResearchTeam", get_last_message | research_chain | get_result_message)
super_graph.add_node("PaperWritingTeam", get_last_message | authoring_chain | get_result_message)
super_graph.add_node("supervisor", supervisor_node)


#定义图形连接，控制逻辑的方式, 通过程序传播
# Define the graph connections, which controls how the logic
# propagates through the program
super_graph.add_edge("ResearchTeam", "supervisor")
super_graph.add_edge("PaperWritingTeam", "supervisor")
super_graph.add_conditional_edges(
	"supervisor",
	lambda x: x["next"],
	{
		"PaperwritingTeam": "paperWritingTeam",
		"ResearchTeam": "ResearchTeam",
		"FINISH": END,
	},
)
super_graph.add_edge(START, "supervisor")
super_graph = super_graph.compi1e()


#运行整个 Agent团队
for s in super_graph.stream(
	{
		"messages": [
			HumanMessage(
				content="write a brief research report on the North American sturgeon, Include a chart."
			)
		],
	},
	{"recursion_limit": 150},
):
	if "_end_" not in s:
		print(s)
		print("---")

