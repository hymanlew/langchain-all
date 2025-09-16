"""
langchain 中内置的提示词模板：
from langchain.prompts import AIMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import chatMessagromptTemplate

自定义提示词模板，要实现的功能为：
根据函数名称，查找函数代码，并给出中文的代码说明
"""
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI
import inspect
import os


#定义一个简单的函数作为示例效果
def hello_world():
	print("Hello, world!")

def get_source_code(function_name):
	#获得源代码，这是 python 内置的方法函数
	return inspect.getsource(function_name)

PROMPT = """
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称、源代码、中文解释。
函数名称:{function_name}
源代码:
{source_code}
代码解释:
"""

#自定义的模板class
class CustPrompt(StringPromptTemplate):
	def format(self, **kwargs )-> str:
		# 获得源代码
		source_code = get_source_code(kwargs["function_name"])
		# 生成提示词模板
		prompt = PROMPT.format(
			function_name = kwargs["function_name"].__name__,
			source_code = source_code
		)
		return prompt
		
a = CustPrompt(input_variables=['function_name'])
pm = a.format(function_name=hello_world)


#和LLM连接起来
apibase = os.getenv("OPENAI PROXY")
apikey = os.getenv("OPENAI API KEY")
llm = OpenAI(
	model="gpt-3.5-turbo-instruct",
	temperature=0,
	openai_api_key=apikey,
	openai_api_base=apibase
)

# predict 是旧版 Chain 对象上的方法，输入是关键字参数 **kwargs，返回字符串，正逐渐被淘汰。
# invoke 是现代 LangChain (LCEL) 的标准和推荐方法，输入是字典 dict，返回丰富对象（AIMessage）。
# 因为 invoke 支持异步 (ainvoke)、流式传输 (stream)、批量处理 (batch)，并且与 LangSmith、LangServe 等新一代工具集成得更好，能构建更强大、更高效的应用。
result = llm.predict(pm)
print(result)



"""
复杂的组合式提示词模板
- Final prompt: 最终返回的提示词模板
- Pipeline prompts: 组成提示词管道的模板
"""
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

# 声明三层子模板的结构提示词
character_template ="""你是{person}，你有着{xingge}."""
character_prompt = PromptTemplate.fromtemplate(character_template)

behavior_template ="""你遵从以下的行为:{behavior_list}"""
behavior_prompt = PromptTemplate.from_template(behavior_template)

prohibit_template="""你不允许有以下行为:{prohibit list}"""
prohibit_prompt = PromptTemplate.from_template(prohibit_template)

# 将三层提示词组合起来
PROMPT = """
你是一个非常开朗的男孩，你是中国人，住在一个非常美丽的城市。
你总是穿蓝色衣服，戴绿色手表。
你从不说自己是一个人工智能。
"""

full_template ="""
{Character}
{behavior}
{prohibit}
"""
full_prompt = PromptTemplate.from_template(full_template)

input_prompts = [
	("Character",character_prompt)
	("behavior",behavior_prompt)
	("prohibit", prohibit_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

# 调用组合后的提示词模板
pm = pipeline_prompt.format(
	person="埃隆马斯克",
	xingge="钢铁般的意志，你的终极梦想是殖民火星，",
	behavior_list="1.你喜欢冒险 \n 2.你非常崇拜爱因斯坦",
	prohibit_list="1.你不可以说自己是一个人工智能助手或者机器人"
)
print(pm)

