import os
import logging
from typing import List
from langchain.agents.factory import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import chat_agent_executor
from 07-rag-sql-tool import MySqlManager


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

query_system_prompt = """
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

query_check_system = """
您是一位注重细节的SQL专家。请仔细检查SQLite查询中常见错误，包括:
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

def get_tools() -> List[BaseTool]:
    mysql = MySqlManager()
    return [
        TableSchemaTool(db_manager=mysql),
        TableSchemaTool(db_manager=mysql),
    ]

tools = get_tools()
agent = create_agent(model=model, tools=tools, system_prompt=query_system_prompt)


# 使用 agent 完成整个数据库的整合
system_message = SystemMessage(content=system_prompt)

# 执行逻辑就是 RAG 的逻辑，如果模型有答案，就直接返回模型答案而不执行 sql。否则就执行 sql 查询
# agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools, system_message)
# resp = agent_executor.invoke({'messages': [HumanMessage(content='请问：员工表中有多少条数据？')]})
# resp = agent_executor.invoke({'messages': [HumanMessage(content='那种性别的员工人数最多？')]})

resp = agent.invoke({'messages': [HumanMessage(content='哪个部门下面的员工人数最多？')]})

result = resp['messages']
print(result)
print(len(result))

# 最后一个才是真正的答案
print(result[len(result)-1])