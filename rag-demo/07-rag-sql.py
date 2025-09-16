# pip install mysqlclient
import os
from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

# model = ChatOpenAI(model='gpt-3.5-turbo')
model = ChatOpenAI(
    model='glm-4-0520',
    temperature=0,
    api_key='0884a4262379e6b9e98d08be606f2192.TOaCwXTLNYo1GlRM',
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)

# 使用 sqlalchemy 库 + mysqlclient 驱动，连接 mysql
# 初始化 MySQL 数据库的连接
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'test_db8'
USERNAME = 'root'
PASSWORD = '123123'
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)
# 测试连接是否成功
# print(db.get_usable_table_names())
# print(db.run('select * from t_emp limit 10;'))

# 直接使用大模型和数据库整合
# 1, 初始化生成 SQL 的chain, 生成 sql, 此时只能根据你的问题生成 SQL
sql_chain = create_sql_query_chain(model, db)
# resp = sql_chain.invoke({'question': '请问：员工表中有多少条数据？'})
# print(resp)
# sql = resp.replace('```sql', '').replace('```', '')
# print('提取之后的SQL：' + sql)
# print(db.run(sql))

answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)
# 2, 创建一个执行 sql 语句的工具, 执行 sql
execute_sql_tool = QuerySQLDataBaseTool(db=db)

# 创建一个 chain 链去执行
# chain = sql_chain | (lambda x: x.replace('```sql', '').replace('```', '')) | execute_sql_tool
# resp = chain.invoke({'question': '请问：一共有多少个员工？'})
# print(resp)
sql_chain = sql_chain | (lambda x: x.replace('```sql', '').replace('```', ''))

# 创建一个 chain 链去执行
# assign 是断言-判断，query/result 是模板中的参数，itemgetter 是获取指定 sql 执行后的结果
# RunnablePassthrough 是代表接收用户的问题，然后再传递给 prompt 和 model。
chain = (RunnablePassthrough.assign(query=sql_chain)
         .assign(result=itemgetter('query')
         | execute_sql_tool)
         | answer_prompt
         | model
         | StrOutputParser()
         )

rep = chain.invoke({'question': '请问：员工表中有多少条数据？'})
print(rep)
