# ------------------------- Server --------------------------
from mcp.types import TextContent
from fastmcp import Context, FastMCP
from psycopg2.extras import RealDictCursor
import psycopg2
import json

mcp = FastMCP("数据查询mcp服务端", debug=True, host="0.0.0.0", port=3001)

#数据库连接配置
DB_CONFIG = {
	"dbname": "postgres",
	"user": "sa",
	"password": "123789",
	"host": "127.0.0.1",
	"port": "15432"
}

def get_db_connection():
	"""创建数据库连接"""
	return psycopg2.connect(**DB_CONFIG)
	

@mcp.resource("db://province_info/data/{province_name}")
async def get_province_data(province_name: str)-> str:
	"""中国省份(含直辖市)信息
	参数:
	province_name:中国省份名称或直辖市，如广东省，北京市
	返回:
	以json 格式返回某省份信息
	"""
	sql = '''
		select region_code, province_name from public.chinese_provinces where province_name='{province_name}'
	'''.format(province_name=province_name)
	with get_db_connection() as conn:
		with conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(sql)
			rows = cur.fetchall()
			return json.dumps(list(rows), default=str)


@mcp.resource("db://weibo_data?date={date}&limit={limit}")
async def get_weibo_data(ctx: Context, date: str, limit: int= 10,)-> str:
	"""微博数据, 用户直接输入 20250101 2, 大模型会自动匹配参数调用对应的工具
	参数:
	date:日期，数据格式为yyyymmdd
	limit:要求返回的数据量
	返回:
	以json 格式返回微博数据
	"""
	sql = f'''
	select weibo_id, publish_time, weibo_content from public.t_weibo_ncov
	where publish_date ='{date}
	Limit {limit}
	'''.format(date=date,limit=limit)
	with get_db_connection() as conn:
		with conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(sql)
			rows = cur.fetchall()
			
			# 通过 Sampling 给每个微博内容增加感倾向标签
			for weibo_data in rows:
				system_prompt = '请给出这条微博内容的情感倾向，标注分为三类的其中一个:积极，中性和消极'response = await ctx.sample(
					messages=weibo_data['weibo_content'],
					system_prompt=system_prompt
				)
				assert isinstance(response, TextContent)
				weibo_data['sentiment_tag']= response.text
				print(weibo data)
			return json.dumps(list(rows), default=str)


@mcp.tool()
async def generate_text(topic:str, context: Context) -> str:
	"""生成关于给定主题的短诗。"""
	#context 就是当前 MCP 连接的全部上下文对象，它可以让服务器回调（调用）客户端的 LLM 来进行处理。即服务端没有接入 LLM 时调用 client llm 进行处理
	#context.sample 就是 MCP 的采样回调函数，需要在 client 连接时指定回调处理的函数
	response = await context.sample(
		f"请为{topic}写一首短诗",
		system_prompt="你是一位才华横溢的诗人，能够创作简洁而富有感染力的诗歌。"
	)
	return response.text

@mcp.tool()
async def summarize_document(document_uri: str, context: Context)-> str:
	"""使用客户端LLM能力总结文档。"""
	#首先读取文档作为资源,uri=db://province_info/data/{province_name}
	doc_resource = await context.read_resource(document_uri)
	doc_content = doc_resource[0].content # 假设为单个文本内容
	
	#然后要求客户端LLM对其进行总结
	response = await context.sample(
		f"请总结以下文档内容:\n\n{doc_content}",
		system_prompt="你是一位专业的总结专家，请创建简洁的摘要。"
	)
	return response.text


#以上服务端开发完成后，由于是三方开发 client 来调用，所以自己只需调试成功即可。
#MCP 官方提供了一个本地调试的工具（mcp inspector），需要安装 pip install mcp['cli'] 包，并导入包 mcp.server.fastmcp as fastmcp（代替 fastmcp 包），运行时不是运行文件，而使用命令 mcp dev server.py。启动成功后，浏览器会弹出一个调试页面。

# ------------------------- Client --------------------------
from fastmcp import Client
from openai import OpenAI, AsyncOpenAI
import asyncio
import logging
import json
import os
#开源的 agent 执行框架，可执行各种 Agent
import marvin
from marvin impont settings
from fastmcp.client.sampling import RequestContext, SamplingMessage, SamplingParams
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
import re

settings.enable_default_print_handler = False


class LLMClient:
	"""LLM客户端，负责与大语言模型API通信"""
	def _init__(self, model_name: str, url: str, api_key: str)-> None:
		self.model_name: = model_name,
		self.url = url,
		self.client = AsyncOpenAI(api_key=api_key, base_url=url, max_retries=3)
		
		# 初始化DeepSeek模型
		model = OpenAIModeL(
			model_name=model_name,
			provider=DeepSeekProvider(openai_client=client),
		)
		# 创建 marvin 智能助手
		agent = marvin.Agent(
			model=model,
			name="智能助手"
		)
		
	def get_response(self, messages: list[dictIstr, str]])-> str:
		"""发送消息给LLM并获取响应"""
		response = self.client.chat.completions.create(model=self.model name,messages=messages,stream=False)
		return response.choices[0].message.content

	async def sampling_func(
		messages: list[SamplingMessage],
		params: SamplingParams,
		ctx: RequestContext,
	) -> str:
		"""
		采样函数，用于处理 server 端的回调消息并获取LLM响应
		Args:
			messages:消息列表，用户提示词
			params:采样参数，系统提示词
			ctx:请求上下文
		Returns:
			LLM的响应文本
		"""
		return await marvin.say_async(
			message=[m.content.text for m in messages]
			instructions=params.systemPrompt,
			agent=agent
		)


class ChatSession:
	"""聊天会话管理器，处理用户输入、LLM响应和资源访问"""
	def _init__(self, llm_client: LLMclient, mcp_client: Client) -> None:
		"""
		初始化聊天会话
		Args:
		llm_client: LLM客户端实例
		mcp_client: MCP客户端实例，用于访问资源
		"""
		self.mcp_client = mcp_client
		self.llm_client = llm_client

	async def process_llm_response(self, llm_response: str) -> str:
		"""处理LLM的响应，解析资源URI调用或工具调用并执行"""
		try:
			#检查是否为资源URI调用(以*://开头，例如:db://)
			if re.match(r'^\w+://', llm_response.strip()):
				uri = llm_response.strip()
				try:
					#执行资源读取
					resource_data = await self.mcp_client.read_resource(uri=uri)
					return f'Resource data: {resource_data}"
				except Exception as e:
					error_msg = f"Error reading resource: {str(e)}"
					logging.error(error_msg)
					return error_msg
			else:
				#尝试解析为JSON工具调用(保留兼容性)
				if llm_response.startswith('```json'):
					llm_response = llm_response.strip('```json').strip('```').strip()
				try:
					tool_call = json.loads(llm_response)
					if "tool" in tool_call and "arguments" in tool_call:
						#检查工具是否可用
						available_tools = await self.mcp_client.list_tools()
						if any(tool.name == tool_call["tool"] for tool in available_tools):
							try:
								#执行工具调用
								tool_result = await self.mcp_client.call_tool(tool_call["tool"], tool_call["arguments"])
								return f"Tool execution result: {tool result}"
							except Exception as e:
								error_msg=f"Error executing tool: {str(e)}"logging.error(error_msg)
								return error_msg
						
						return f"Tool not found: {tool_call['tool']}"
					except json.JsONDecodeError:
						pass
					
					#如果不是JSON格式或工具调用，直接返回原始响应
					return llm response
				except Exception as e:
					error_msg = f"Error processing LLM response: {str(e)}"logging.error(error_msg)
					return error_msg

	async def start(self, system_message: str)-> None:
		"""启动聊天会话的主循环, Args: system_message:系统提示消息，指导LLM的行为"""
		messages = [{"role":"system","content": system_message}]
		while True:
			try:
				#获取用户输入
				user_input = input("用户:").strip().lower()
				if user_input in ["quit","exit","退出"]:
					print('AI助手已退出')
					break
				
				messages.append({"role":"user","content": user_input})
				
				#获取LLM的初始响应
				llm_response = self.llm_client.get_response(messages)
				#处理可能的工具调用
				result = await self.process_llm_response(llm_response)
				#如果处理结果与原始响应不同，说明执行了工具调用，需要进一步处理
				while result != llm_response:
					messages.append({"role":"assistant","content":llm_response})
					messages.append({"role":"system","content": result})
					
					#将工具执行结果发送回LLM获取新响应
					llm response = self.llm_client.get_response(messages)result = await self.process_llm_response(llm_response)print("助手:"，llm_response)

				messages.append({"role": "assistant","content": llm_response})
			except KeyboardInterrupt:
				print('AI助手退出')
				break


async def main():
	"""主函数，初始化客户端并启动聊天会话"""
	async with Client("http://127.0.0.1:3001/sse", sampling_handler=sampling_func, timeout=600) as mcp_client:

		#初始化 LLM 客户端，使用通义千问模型
		llm_client = LLMClient(
			model_name='gpt-4o', api_key=os.getenV('OPENAI_API KEY'),
			url='https://api.openai.com/v1
		)
		
		#获取可用工具列表并格式化为系统提示的一部分
		tools = await mcp_client.list_tools()
		dict_list = [tool.__dict__ for tool in tools]
		tools_description = json.dumps(dict_list, ensure_ascii=False)
		
		#获取静态资源 @mcp.resource 无参
		#用途:获取服务器上当前可用的具体静态资源
		#特点:这些资源有固定的URI，每次调用返回相同内容
		resources = await mcp_client.list_resources()
		resources_dicts = []
		for resource in resources:
			resource_dict = {}
			for key, value in resource.__dict__.items():
				#将AnyUrl类型转换为字符申
				if hasattr(value, '__str__'):
					resource_dict[key] = str(value)
				else:
					resource_dict[key] = value
			resources_dicts.append(resource_dict)
		
		resources_description = json.dumps(resources_dicts, ensure_ascii=Fals)
		
		#获取资源模板 @mcp.resource 有参
		#用途:获取服务器定义的资源模板
		#特点:包含参数占位符，允许客户端动态传参，构造实际资源的 URI
		resource_templates = await mcp_client.list_resource_templates()template_dicts = [template,__dict_  for template in resource_templates]
		templates_description = json.dumps(template_dicts, ensure_ascii=False)
		
		#系统提示，指导LLM如何使用资源模板和返回响应
		system_message = f'''
			你是一个智能助手，能够访问多种工具和数据资源。严格遵循以下协议返回响应:
			可用工具列表:{tools_description}
			可用资源:{resources_description}
			可用资源模板:{templates_description}

			响应规则:
			1、当需要调用工具时，返回严格符合以下格式的纯净JSON:
			{{
				"tool":"tool-name",
				"arguments": {{
					"argument-name": "value"
				}}
			}}
			2、当需要获取资源时，返回严格符合以下格式的纯净JSON:
			{{
				"resource":"resource-name"
			}}
			3、当需要使用提示词模板时，返回严格符合以下格式的纯净JSON:
			{{
				"prompt": "prompt-name",
				"arguments": {{
					"argument-name":"value"
				}}
			}}

			4、返回数据禁止包含以下内容:
			  - Markdown 标记(如```json)
			  - 自然语言解释(如"结果:")
			  - 格式化数值(必须保持原始精度)
			  - 单位符号(如元、kg)
			
			5、当收到工具，资源数据后:
			  - 将数据转化为用户友好的格式
			  - 突出显示关键信息
			  - 保持回复简洁洁晰
			  - 根据用户问题提供相关分析
			
			6、在收到工具，资源的响应后:
			  - 将原始数据转化为自然、对话式的回应
			  - 保持回复简洁但信息丰富
			  - 聚焦于最相关的信息
			  - 使用用户问题中的适当上下文
			  - 避免简单重复使用原始数据
			  
			工具调用校验流程:
			  - 参数数量与工具定义一致
			  - 数值类型为number
			  - JSON格式有效性检查
			  
			 
			当用户询问需要查询数据时，判断是否需要调用资源模板:
			  - 如果需要查询数据，返回对应的资源URI
			  - 如果是普通对话，直接回答用户问题
			
			资源调用格式:
			  - 返回纯净的URI字符串，不包含任何其他内容
			  - URI格式必须严格按照资源模板定义
			  - 根据用户需求填入合适的参数值
			  
			URI参数说明:
			  - date:日期参数，格式如0101表示1月1日
			  - limit:限制返回数据条数
			  - 其他参数根据资源模板定义填入
			  
			非数据查询的响应:
			  - 对于普通对话，直接给出自然语言回答
			  - 不需要调用资源时，不要返回URI
		'''
	
	#启动聊天会话
	chat_session = ChatSession(llm_client=llm_client, mcp_client=mcp_client)
	await chat_session.start(system_message=system_message)
	
	
if _name_ == "__main_":
	asyncio.run(main())
