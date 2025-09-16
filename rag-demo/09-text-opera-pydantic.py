import os
from typing import Optional, List

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

model = ChatOpenAI(model='gpt-4-turbo', temperature=0)

# pydantic: 处理数据，验证数据，定义数据的格式，虚拟化和反虚拟化，类型转换等等。
class Person(BaseModel):
    name: Optional[str] = Field(default=None, description='表示人的名字')
    hair_color: Optional[str] = Field(default=None, description="人的头发颜色")
    height_in_meters: Optional[str] = Field(default=None, description="以米为单位的高度")


# 数据模型类： 代表多个人
class ManyPerson(BaseModel):
    people: List[Person]


# 自定义提示词，以提供指令和任何其他上下文。
# 1) 你可以在提示模板中添加示例以提高提取质量
# 2) 引入额外的参数以考虑上下文（例如，包括有关提取文本的文档的元数据。）
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的提取算法。只从未结构化文本中提取相关信息。如果你不知道要提取的属性的值，返回该属性的值为null。",
        ),
        # 请参阅有关如何使用参考记录消息历史的案例
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

# with_structured_output 模型的输出是一个结构化的数据
chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=ManyPerson)

# text = '马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右，'
# text = "马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右。走在她旁边的是她的男朋友，叫：刘海；比她高10厘米。"
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
resp = chain.invoke(text)
print(resp)


# ------------------------------------

"""
文本分类是自然语言处理领域中的一个重要任务，旨在将文本数据自动归类到预定义的类别中。 它是实现信息检索、智能推荐、情感分析等应用的基础技术之一
文本分类的主流应用场景有:
情感分析:sentiment analysis(SA)
话题标记:topic labeling(TL)
新闻分类:news classification(NC)
对话行为分类:dialog act classification(DAC)
自然语言推理:natural language inference(NLD)
关系分类:relation classification(RC)。
事件预测:event prediction(EP)
"""
# 定义一个 Pydantic 的数据模型，未来需要根据该类型，完成文本的分类
class Classification(BaseModel):
    # 文本的情感倾向，预期为字符串类型
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"], description="文本的情感")
    # 文本的攻击性，预期为1到5的整数
    aggressiveness: int = Field(..., enum=[1, 2, 3, 4, 5], description="描述文本的攻击性，数字越大表示越攻击性")
    # 文本使用的语言，预期为字符串类型
    language: str = Field(..., enum=["spanish", "english", "french", "中文", "italian"], description="文本使用的语言")


# 创建一个用于提取信息的提示模板
tagging_prompt = ChatPromptTemplate.from_template(
    """
    从以下段落中提取所需信息。
    只提取'Classification'类中提到的属性。
    段落：
    {input}
    """
)

chain = tagging_prompt | model.with_structured_output(Classification)

input_text = "中国人民大学的王教授：师德败坏，做出的事情实在让我生气！"
input_text = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
result: Classification = chain.invoke({'input': input_text})
print(result)
