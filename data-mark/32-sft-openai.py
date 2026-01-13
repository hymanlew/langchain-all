from openai import OpenAI

client = OpenAI()

"""
OpenAI 官方推荐原生的全参 SFT方法，封装了所有底层细节，API调用简单。根据提供的问题和标准答案进行学习，模仿学习特定知识、格式或风格。
- 数据要求：需要大量成对的标准的“指令-输出”数据
- 适用场景：风格统一、有明确答案的任务（格式化、内容生成）	
  -	邮件服务：让模型学习公司邮件的特定写作风格
  - 客户服务：基于历史工单数据微调模型，使其能精准回答产品相关问题
  - 报告生成：按照公司的特定模板和语气生成报告、营销文案或邮件。

而 DPO/GRPO/LoRA 等，是让模型的输出更符合人类或特定标准（如逻辑、安全）的偏好。在开放式任务中追求更优解。
- 数据要求：需要复杂、无标准答案的多样性数据
- 适用场景：开放式推理、专业领域判断（法律分析、医学诊断）。学会在任务中进行推理和优化
- 聊天助手、内容创作、复杂推理（需要判断“好坏”的任务）

- 有明确、统一、正确的答案（如格式化文档、特定领域QA），原生 SFT 简单高效。
- 任务没有唯一标准答案，需要权衡质量、安全性、创造性等（如聊天、创意写作、复杂推理），SFT/PEFT 高效。

Creates a fine-tuning job which begins the process of creating a new model from a given dataset.
Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.

Your dataset must be formatted as a JSONL file. Additionally, you must upload your file with the purpose `fine-tune`.
See [upload file](https://platform.openai.com/docs/api-reference/files/create) for how to upload a file.           

重要：client.fine_tuning.jobs.create 是 OpenAI 特有的远端作业创建接口，本地服务通常不直接实现这个端点。
是 OpenAI 云端服务的专属API端点。它设计用于发起一个在 OpenAI服务器上 运行、由其管理和调度的异步微调任务，因此通常不适用于本地模型。

因此仅限于模型推理调用，不包含微调、Assistants等高级管理功能。
"""
# 其他参数如 batch_size、learning_rate_multiplier（学习率倍数），可以使用“auto”选项
n_epochs = 2
job = client.fine_tuning.jobs.create(
    training_file="file-ID8oHTZDz5jp4VdzOnJyvgfs-id",
    model="gpt-3.5-turbo",
    hyperparameters={"n_epochs": n_epochs}
)
print(job.id)

job = client.fine_tuning.jobs.retrieve('ftjob-b9S1AK4BHhBCYJv0I4LBZauM')

job_id = job.id
status = job.status

print(f"微调作业已创建,作业ID: {job_id}")

# 轮询作业状态,直至完成
import time

while status not in ["succeeded", "failed", "cancelled"]:
    print(f"作业状态: {status}, 等待 10 秒...")
    time.sleep(10)

    job = client.fine_tuning.jobs.retrieve(job_id)
    status = job.status

print(f"微调作业已完成,最终状态: {status}")

if status == "succeeded":
    print(f"微调后的模型名称: {job.fine_tuned_model}")

    # 输出微调信息
    response = client.fine_tuning.jobs.list_events(job_id)

    events = response.data
    events.reverse()

    for event in events:
        print(event.message)
else:
    print("微调作业未成功完成,请检查错误信息。")

'''
FineTuningJob(id='ftjob-b9S1AK4BHhBCYJv0I4LBZauM', created_at=1717608923, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(n_epochs=2, batch_size='auto', learning_rate_multiplier='auto'), model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-2MRF2ZhOMmubKIrnf84j3uGi', result_files=[], seed=1597839268, status='validating_files', trained_tokens=None, training_file='file-ID8oHTZDz5jp4VdzOnJyvgfs', validation_file=None, estimated_finish=None, integrations=[], user_provided_suffix=None)
微调作业已创建,作业ID: ftjob-b9S1AK4BHhBCYJv0I4LBZauM
'''
