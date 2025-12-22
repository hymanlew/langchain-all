# pip install mysqlclient
import os
from operator import itemgetter
from typing import Any, Optional, List

from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import BaseTool
from pydantic import create_model, Field

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'


model = ChatOpenAI(model='gpt-4-turbo')

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

# 创建执行 sql 相关操作的工具（DEMO）
toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()
#  query_sql_database_tool, query
#  info_sql_database_tool, schema
#  list_sql_database_tool, tables
#  query_sql_checker_tool, check


class MySqlManager:
    def __init__(self):
        pass

    def get_tables_name(self) -> list[str]:
        # return list(db.get_usable_table_names())
        return tools[2].invoke({})

    def get_tables_schema(self) -> list[dict]:
        return tools[1].invoke({})

    def execute_query(self, query) -> str:
        # 安全检查, out data by json
        return tools[0].invoke({})

    def validate_query(self, query) -> bool:
        # return tools[3].invoke({})
        # 使用 explain {query} 执行
        return True if self.execute_query(f'explain {query}') else False


class ListTablesTool(BaseTool):
    """列出数据库中的所有表及其描述信息"""

    name:str ="sql_db_list_tables"
    description:str="列出MySQL数据库中的所有表名及其描述信息。当需要"
    db_manager:MySqlManager

    def _run(self)-> str:
        try:
            return ','.join(self.db_manager.get_tables_name())
        except Exception as e:
            return '没有搜索到任何内容!'

    async def _arun(self) -> str:
        return self._run()


class TableSchemaTool(BaseTool):
    """列出数据库中的表模式信息"""

    name: str = "sql_db_schema"
    description: str = "获取MYSQL数据库中指定表的详细模式信息，包括列定义、主键、外键等。输入应为逗号分隔的表名列表，或留空获"

    def __init__(self, db_manager: MySqlManager, **kwargs):
        super().__init__(**kwargs)
        self.db_manager = db_manager
        self.args_schema = create_model('TableSchema', name=(Optional[List[str]], Field(..., description='表名')))

    def _run(self, name: Optional[List[str]] = None)-> str:
        try:
            lists = self.db_manager.get_tables_schema()
            return ','.join([key for key, value in lists])
        except Exception as e:
            return '没有搜索到任何内容!'

    async def _arun(self, name: Optional[List[str]] = None) -> str:
        return self._run(name)


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
# assign 是拼接参数，query/result 是模板中的参数，itemgetter 是获取指定 sql 执行后的结果
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

if __name__ == '__main__':
    mysql = MySqlManager()
    mysql.execute_query()

    # tool = ListTablesTool(db_manager=mysql)
    tool = TableSchemaTool(db_manager=mysql)
    print(tool.invoke({}))
