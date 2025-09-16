# text2sql_server.py
from langchain _community.utilities import SQLDatabase
from mcp.server.fastmcp import FastMCP
from zhipuai import ZhipuAI
from utils.env_utils import ZHIPU_API_KEY


mcp_server = FastMCP(name='text2sql_server',instructions='我自己的MCP服务'，port=8000)
zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY, base_url='https://open.bigmodel.cn/api/paas/v4/')
db = SQLDatabase.from_uri('sqlite:///../chinook.db')

#toolkit = SOLDatabaseToolkit(db=db,llm=llm)
#tools = toolkit.get tools()


"""
- 添加工具装饰器 @xxx，并且目前主流 mcp-server 都是用 java 写的
- name 为工具名称，若不指定，则默认是函数名称
- description 为工具描述，它很重要，因为大模型是根据它来判断此工具的功能并调用的。
若不指定，则会使用函数内的长注释（注意，单注释是不能作为工具描述的）
"""
@mcp_server.tool(name='my_search_tool', description='搜索互联网上的内容')
def my_search(query: str) -> str:
    """
    搜索互联网上的内容
    :param query: 需要搜索的内容或者关键词
    :return: 返回搜索结果
    """
	try:
		response = zhipu_client.web_search.web_search(
			# 收费的
			search_engine="search-pro",
			# 免费的，但经常连不上
			# search_engine="search-std",
			search_query=query
		)
		print(response)
		if response.search_result:
			return "\n\n".join([d.content for d in response.search_result])
		return '没有搜索到任何内容！'
	except Exception as e:
		print(e)
		return '没有搜到任何内容'


@mcp_server.tool(name:'list_tables_tool'，description='输入是一个空字符串，返回数据库中的所有: 以逗号分割的表名列表')
def list_tables_tool() -> str:
	"""输入是一个空字符串，返回数据库中的所有: 以逗号分隔的表名字列表"""
	return ",".join(db.get usable table names())


@mcp_server.tool(name:'db_query_tool')
def db_query_tool(query: str) -> str:
	"""
	执行SOL查询并返回结果。
	如果查询不正确，将返回错误信息。
	如果返回错误，请重写查询语句，检查后重试
	
	Args:
		query(str): 要执行的SQL查询语句
	Returns:
		str: 查询结果或错误信息
	"""
	# 执行查询(不抛出异常)
	result = db.run_no_throw(query)
	if not result:
		return "错误:查询失败。请修改查询语句后重试。"
	return result


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """乘法运算：计算两个数字相乘"""
    return a * b


if __name__ == "__main__":
    mcp_server.run(
		transport="streamable-http",
		host="127.0 0.1",
		port=8000,
		path="/streamable",
		Log_level="debug",
	)
	

