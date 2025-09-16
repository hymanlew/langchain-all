from typing import Literal, List, Dict, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
import gradio as gr
import asyncio
import logging
from mcp_config import Config
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.agent_toolkits import SQLDatabaseToolkit 
from langchain community.utilities import SQLDatabase

# 自定义的包
from sql_graph.my_state import SQLState


# 初始化LLM，本地部署的
llm = ChatopenAI(
	temperature=0,
	model="qwen3-8b",
	api_key="EMPTY",
	api_base="http://localhost:6006/v1"
	# 启用深度思考模式
	extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPService:
    """
    MCP 服务封装类（线程安全设计）
    功能：
    1. 管理多服务器连接池
    2. 提供同步/异步接口
    3. 内置重试机制
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls, config: Dict):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config)
        return cls._instance
    
    def _initialize(self, config: Dict):
        """初始化连接池"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._logger = logging.getLogger("mcp_service")
        
    async def _get_client(self):
        """获取客户端连接（自动重试）"""
        retries = 3
        for i in range(retries):
            try:
                async with MultiServerMCPClient(self.config) as client:
                    return client
            except Exception as e:
                if i == retries - 1:
                    raise
                self._logger.warning(f"MCP连接失败，正在重试... ({i+1}/{retries})")
                await asyncio.sleep(1)
    
    async def get_tools_async(self, service_name: str) -> List[BaseTool]:
        """异步获取工具集"""
        async with self._lock:
            client = await self._get_client()
            return await client.get_tools(service_name)
    
    def get_tools_sync(self, service_name: str) -> List[BaseTool]:
        """同步获取工具集（供非异步环境使用）"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_tools_async(service_name))
    
    @classmethod
    async def shutdown(cls):
        """优雅关闭"""
        if cls._instance:
            cls._instance._executor.shutdown()
            cls._instance = None



# 业务层调用示例
def should_continue(state: SQLState) -> Literal[END, "check_query"]:
	"""条件路由的，动态边"""
	messages = state["messages"]
	last_message = messages[-1]
	if not last_message.tool_calls:
		return END
	else:
		return "check_query'


# 异步调用，就是协程
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def make_graph():
    try:
		"""定义，并且编译工作流"""
		# 初始化服务（单例模式）
		client = MCPService(Config.MCP_TEXT2SQL_CONFIG)
		tools = await client.get_tools()
		
		#所有表名列表的工具
		#next() 是 Python 的内置函数，用于从迭代器中获取下一个元素，只取一个
		list_tables_tool = next(tool for tool in tools if tool.name == "list_tables_tool")
		
		#执行sql的工具
		db_query_tool = next(tool for tool in tools if tool.name == "db_guery_tool")
		
		"""
		Python允许在函数/方法内部定义其他函数（称为嵌套函数或局部函数）。作用域规则：
		内层函数（call_list_tables）只能在外层函数（make_graph）内部访问。
		内层函数可以访问外层函数的局部变量（闭包特性）。
		"""
		def call_list_tables(state: SQLState):
			"""第一个节点"""
			tool_call = {
				"name": "list_tables_tool",
				"args": {},
				"id": "abc123",
				"type": "tool_call",
			}
			tool_call_message = AIMessage(content="", tool_calls=[tool_call])
			
			# 调用工具
			#tool_message =list_tables_tool.invoke(tool_call)
			#response = AIMessage(f"所有可用的表: {tool_message.content}")
			#return {"messages": [tool_call_message, tool_message, response]}
			return {"messages":[tool call message]}
        
		
		# 第二个节点
		list_tables_tool = ToolNode([list_tables_tool], name="list_tables_tool")
		
				
		#获取表结构的工具
		get_schema_tool = next(tool for tool in tools if tool.name == 'sql_db_schema')
		#测试工具调用
		#print(get_schema_tool.invoke('employees'))

		def call_get_schema(state: SQLState):
			"""第三个节点"""
			#注意: LangChain 强制要求所有模型都接受 tool_choice="any"
			#以及 tool_choice = <工具名称字符申> 这两种参数
			llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
			response = llm_with_tools.invoke(state["messages"])
			return {"messages": [response]}


		#第四个节点: 直接使用 langgraph 提供的 ToolNode
		get_schema_node = ToolNode([get_schema_tool], name="get_schema")
		generate_query_system_prompt = """
		你是一个设计用于与SQL数据库交互的智能体。给定一个输入问题，创建一个语法正确的{dialect}查询来运行，
		然后查看查询结果并返回答案。除非用户明确指定他们希望获取的示例数量，否则始终将查询限制为最多{top_k}个结果。
		
		你可以按相关列对结果进行排序，以返回数据库中最有趣的示例。
		永远不要查询特定表的所有列，只询问与问题相关的列。
		不要对数据库执行仟何 DML 语句(INSERT、UPDATE、DELETE、DROP等)。
		""".fromat(
			dialect = db.dialect,
			top_k = 5,
		)
		
		query_check_system ="""您是一位注重细节的SQL专家。请仔细检查SQLite查询中常见错误，包括:
		- Using NOT IN with NULL values
		- Using UNION when UNION ALL should have been usedUsing BETWEEN for exclusive ranges
		- Data type mismatch in predicates
		- Properly quoting identifiers
		- Using the correct number of arguments for functions
		- Casting to the correct data type
		- Using the proper columns for joins
		如果发现上述任何错误，请重写查询。如果没有错误，请原样返回查询语句。
		检查完成后，你调用适当的工具来执行查询。
		"""


		def generate_query(state: SQLState):
			"""第五个节点:生成SQL语句"""
			system_message = {
				"role":"system",
				"content": generate_query_system_prompt,
			}
			# 这里不强制工具调用，允许模型在获得解决方案时自然响应
			llm_with_tools = llm.bind_tools([db_query_tool])
			resp = llm_with_tools.invoke([system_message] + state['messages'])
			return {'messages': [resp]}
			
			
		def check_query(state: SQLState):
			"""第六个节点: 检查SQL语句"""
			system_message = {
				"role":"system",
				"content": query_check_system,
			}
			tool_call = state["messages"][-1].tool_calls[0]
			
			# 得到生成后的 SQL
			user_message = {"role": "user", "content": tool_call["args"]["query"]}
			llm_with_tools = llm.bind_tools([db_query_tool], tool_choice='any')
			response = llm_with_tools.invoke([system_message, user_message])
			response.id = state["messages"][-1].id
			return {"messages": [response]}
			
		
		#第七个节点
		run_query_node = ToolNode([db_query_tool], name="run_query")
		workflow = StateGraph(SQLState)
		workflow.add_node(call_list_tables)
		workflow.add_node(list_tables_tool)
		workflow.add_node(call_get_schema)
		workflow.add_node(get_schema_node)
		workflow.add_node(generate_query)
		workflow.add_node(check_query)
		workflow.add_node(run_query_node)
		
		workflow.add_edge(START, "call_list_tables")
		workflow.add_edge("call_list_tables", "list_tables_tool")
		workflow.add_edge("list_tables_tool", "call_get_schema")
		workflow.add_edge("call_get_schema", "get_schema")
		workflow.add_edge("get_schema", "generate_query")
		workflow.add_conditional_edges('generate_query', should_continue)
		workflow.add_edge("check_query", "run_query")
		workflow.add_edge("run_query", "generate_query")
		
		graph = workflow.compile()
		yield graph
		

    finally:
        await MCPService.shutdown()



if __name__ == '__main__':
    # 生产环境启动
    instance.launch(**{
        "auth": Config.GRADIO_AUTH,
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "debug": False
    })
	
	
	