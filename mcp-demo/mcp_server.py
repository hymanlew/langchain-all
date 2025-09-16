# math_server.py
from mcp.server.fastmcp import FastMCP
from zhipuai import ZhipuAI
from utils.env_utils import ZHIPU_API_KEY

mcp = FastMCP("Math")
zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY, base_url='https://open.bigmodel.cn/api/paas/v4/')


"""
- 添加工具装饰器 @xxx，并且目前主流 mcp-server 都是用 java 写的
- name 为工具名称，若不指定，则默认是函数名称
- description 为工具描述，它很重要，因为大模型是根据它来判断此工具的功能并调用的。
若不指定，则会使用函数内的长注释（注意，单注释是不能作为工具描述的）
"""
@mcp.tool(name='my_search_tool', description='搜索互联网上的内容')
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


@mcp.tool()
def add(a: int, b: int) -> int:
    """加法运算: 计算两个数字相加"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """乘法运算：计算两个数字相乘"""
    return a * b


if __name__ == "__main__":
    mcp.run(
		transport="streamable-http",
		host="127.0 0.1",
		port=8000,
		path="/streamable",
		Log_level="debug",
	)
	

