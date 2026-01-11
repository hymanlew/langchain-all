from langchain.prompts import AIMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
	SystemMessagePromptTemplate,
	HumanMessagePromptTemplate,
	ChatMessagePromptTemplate,
	ChatPromptTemplate
)
"""
消息类（Message Classes）：用于表示已经生成的具体消息。
- SystemMessage
- HumanMessage
- AIMessage

消息模板类（PromptTemplate Classes）：用于创建可重复使用的消息模板，包含变量占位符。
- SystemMessagePromptTemplate
- HumanMessagePromptTemplate

ChatMessagePromptTemplate：可以创建任意角色的消息模板，用于创建自定义角色的消息模板。
ChatPromptTemplate：模板容器，用于组合多个消息模板。构建复杂的对话流程。
"""
custom_template = ChatMessagePromptTemplate.from_template(
    role="expert",
    template="作为一名{expert_type}专家，我的观点是：{opinion}"
)
message = custom_template.format(
    expert_type="人工智能",
    opinion="AI将改变世界"
)
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
# Create a chat prompt template from a human template string
# 相当于：ChatPromptTemplate.from_messages([("human", "你好，{name}！")])
# 只适用于简单对话，用户直接输入，不需要系统指令的简单场景
prompt = ChatPromptTemplate.from_template(template)


"""
自定义提示词模板，要实现的功能为：
根据函数名称，查找函数代码，并给出中文的代码说明
"""
from langchain_core.prompts import StringPromptTemplate
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


# 定义系统提示词（核心：规范“何时搜”“怎么搜”“怎么展示思考”）
system_prompt = """
你是AI个人助手，需实现“边想边搜边答”，核心规则如下：
一、思考与搜索判断（必须实时输出思考过程）：
1. 若问题涉及“时效性（如近3年数据）、知识盲区（如具体企业薪资）、信息不足”，必须调用web_search；
2. 思考时需说明“是否需要搜索”“为什么搜”“搜索关键词是什么”。

二、回答规则：
1. 优先使用搜索到的资料，引用格式为`[1] (URL地址)`；
2. 结构清晰（用序号、分段），多使用简单易懂的表述；
3. 结尾需列出所
"""

# 定义系统提示词
"""
你是AI个人助手，负责解答用户的各种问题。你的主要职责是：
1. **信息准确性守护者**：确保提供的信息准确无误。
2. **搜索成本优化师**：在信息准确性和搜索成本之间找到最佳平衡。
# 任务说明
## 1. 联网意图判断
当用户提出的问题涉及以下情况时，需使用 `web_search` 进行联网搜索：
- **时效性**：问题需要最新或实时的信息。
- **知识盲区**：问题超出当前知识范围，无法准确解答。
- **信息不足**：现有知识库无法提供完整或详细的解答。
## 2. 联网后回答
- 在回答中，优先使用已搜索到的资料。
- 回复结构应清晰，使用序号、分段等方式帮助用户理解。
## 3. 引用已搜索资料
- 当使用联网搜索的资料时，在正文中明确引用来源，引用格式为：  
`[1]  (URL地址)`。
## 4. 总结与参考资料
- 在回复的最后，列出所有已参考的资料。格式为：  
1. [资料标题](URL地址1)
2. [资料标题](URL地址2)
"""

# --------------------------------------------------------------------------

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
customer_style = """American English \
in a calm and respectful tone
"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
prompt_template = ChatPromptTemplate.from_template(template_string)

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
response = chain.invoke(customer_messages)
print(response)
