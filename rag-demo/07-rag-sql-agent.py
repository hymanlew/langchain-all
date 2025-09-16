import os

from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

model = ChatOpenAI(model='gpt-4-turbo')

# 使用 sqlalchemy 库 + mysqlclient 驱动，连接 mysql
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'test_db8'
USERNAME = 'root'
PASSWORD = '123123'
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)

# 创建执行 sql 相关操作的工具
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 使用 agent 完成整个数据库的整合
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""
# 创建代理
system_message = SystemMessage(content=system_prompt)
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools, system_message)

# 执行逻辑就是 RAG 的逻辑，如果模型有答案，就直接返回模型答案而不执行 sql。否则就执行 sql 查询
# resp = agent_executor.invoke({'messages': [HumanMessage(content='请问：员工表中有多少条数据？')]})
# resp = agent_executor.invoke({'messages': [HumanMessage(content='那种性别的员工人数最多？')]})
resp = agent_executor.invoke({'messages': [HumanMessage(content='哪个部门下面的员工人数最多？')]})

result = resp['messages']
print(result)
print(len(result))

# 最后一个才是真正的答案
print(result[len(result)-1])