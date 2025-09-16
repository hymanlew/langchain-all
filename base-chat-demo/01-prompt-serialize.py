"""
序列化提示词模板：使用文件来管理提示词模板。
- 便于存储，共享，版本管理
- 支持常见格式(json/yaml/txt)
"""
from langchain.prompts import load_prompt
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 加载yaml格式的prompt模版
prompt = load_prompt("simple-prompt.yaml")
print(prompt.format(name="小黑", what="恐怖的"))

'''
{
	"input variables": [
		"guestion'",
		"student answer'
	],
	"output_parser": {
		"regex": "(.*?)\\nScore:(.*)",
		"output_keys": [
			"answer",
			"score"
		],
		"default_output": null,
		"type": "regex_parser"
	},
	"partial_variables": {},
	"template": "Given the following guestion and student answer, provide a correct answer and score thestudent answer.\n
			Question: {question}\nStudent Answer: {student answer} Correct Answer:",
		"template_fonmat": "f-string",
	"validate template": true,
	"type": "prompt"
}
'''
#支持加载文件格式的模版，并且对 prompt 的最终解析结果进行自定义格式化
prompt = load_prompt("parser.ison")
prompt.output_parser.parse(
	"George washington was born in 1732 and died in 1799.\nscore: 1/2"
)

'''
使用示例选择器，从大批量示例中提取一些示例，提供给 ExampleSelector
它不会将示例直接提供给 FewShotPromptTemplate 对象（一是有些示例对于当前上下文没有参考意义，二是 LLM 也有 max-len token 限制）。
所以是将它们提供给一个 ExampleSelector 对象，插入部分示例。
这里使用 SemanticSimilarityExampleSelector 类。该类根据与输入的相似性选择小样本示例。它使用嵌入模型计算输入和小样本示例之间的相似性，
然后使用向量数据库执行相似搜索，获取跟输入相似的示例。
'''
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

examples = [
	{
		"question":"问题",
		"answer":"回答"
	}
]
example_selector = SemanticSimilarityExampleSelector.from_examples(
	#这是可供选择的示例列表
	examples,
	#这是用于生成嵌入的嵌入类，该嵌入用于衡量语义相似性
	OpenAIEmbeddings(),
	#这是用于存储嵌入和执行相似性搜索的 VectorStore 类
	Chroma,
	#这是要生成的示例数
	k=3
)

#选择与输入最相似的示例
question ="乔治·华盛顿的父亲是谁?"
selected_examples = example_selector.select_examples({"question": question})
print(f"最相似的示例:{question}")

for example in selected_examples:
	print("\\n")
	for k,v in example.items():
		print(f"{k}: {v}")

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题: {question} \n{answer}")

prompt = FewShotPromptTemplate(
	example_selector=example_selector,
	example_prompt=example_prompt,
	# 后缀，执行模板时，接收参数并拼接后的值
	suffix="问题:{input}",
	# 执行模板时，接收的参数
	input_variables=["input"]
)
print(prompt.format(input="乔治·华盛顿的父亲是谁?"))

'''
问题:乔治·华盛顿的祖父母中的母亲是谁?\n
这里需要跟进问题吗:是的。
跟进:乔治·华盛顿的母亲是谁?
中间答案:乔治·华盛顿的母亲是Mary Ball Washington。
跟进:Mary Ball Washington 的父亲是谁?
中间答案:Mary Ball Washington的父亲是Joseph Ball。
所以最终答案是:Joseph Ball

问题:乔治·华盛顿的父亲是谁?
'''
