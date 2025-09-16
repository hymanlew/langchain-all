import os

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel

"""
合成数据是指用 AI 或人工生成的数据（AI 生成的，也包括它自己的训练数据也是 AI 生成的），
而不是从现实世界事件中收集的数据。它用于模拟真实数据，而不会泄露隐私或遇到现实世界的限制。
安装依赖:
pip install langchain_experimental -i https://pypi.org/simple

合成数据的优势:
1.隐私和安全:没有真实的个人数据面临泄露风险。
2.数据增强:扩展机器学习的数据集。
3.灵活性:创建特定或罕见的场景。
4.成本效益:通常比现实世界数据收集更便宜。
5.快速原型设计:无需真实数据即可快速测试
6.数据访问:当真实数据不可用时的替代方案
"""
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8)

# 创建生成数据的 chain 链
chain = create_data_generation_chain(model)

# 生成数据，给定一些关键词， 随机生成一句话
# result = chain(
#     {
#         # 指定关键词
#         "fields": ['蓝色', '黄色'],
#         # 选项或参数
#         "preferences": {}
#     }
# )
result = chain(
    {
        "fields": {"颜色": ['蓝色', '黄色']},
        "preferences": {"style": "让它像诗歌一样。"}
    }
)
print(result)


# 生成一些结构化的数据：5个步骤
# 1、定义数据模型
class MedicalBilling(BaseModel):
    patient_id: int  # 患者ID，整数类型
    patient_name: str  # 患者姓名，字符串类型
    diagnosis_code: str  # 诊断代码，字符串类型
    procedure_code: str  # 程序代码，字符串类型
    total_charge: float  # 总费用，浮点数类型
    insurance_claim_amount: float  # 保险索赔金额，浮点数类型


# 2、提供一些样例数据，给AI
examples = [
    {
        "example": "Patient ID: 123456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"
    },
    {
        "example": "Patient ID: 789012, Patient Name: 王兴鹏, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"
    },
    {
        "example": "Patient ID: 345678, Patient Name: 刘晓辉, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"
    },
]

# 3、创建一个提示模板，用来指导AI生成符合规定的数据
# 接收名为 example 的变量作为模板，将 example 变量内容替换为生成的字符串
data_template = PromptTemplate(input_variables=['example'], template="{example}")

# 根据提供的模板示例，生成少许提示示例（与 zero-shot 对应），用于大模型参考
prompt_template = FewShotPromptTemplate(
    # 前缀，设置任务的上下文或说明。比如，告诉模型要生成医疗账单数据。
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    # 后缀，包含具体的指令。比如让模型根据主题和额外信息生成数据。
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    examples=examples,
    example_prompt=data_template,
    # 用于替换 prefix 或 suffix 中的占位符，列表中的参数名是固定的
    input_variables=['subject', 'extra']
)

# 4、创建一个结构化数据的生成器
generator = create_openai_data_generator(
    output_schema=MedicalBilling,  # 指定输出数据的格式，指定为自定义的结构
    llm=model,
    prompt=prompt_template
)

# 5、调用生成器
result = generator.generate(
    subject='医疗账单',  # 指定生成数据的主题
    extra='医疗总费用呈现正态分布，最小的总费用为1000',  # 额外的一些指导信息
    runs=10  # 指定生成数据的数量
)
print(result)
