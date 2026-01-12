"""
AutoModel、AutoModelForSequenceClassification、AutoModelForCausalLM 都是 HuggingFace Transformers 库中的自动模型类，
它们根据预训练模型配置自动选择合适的模型架构。区别在于它们针对不同的任务设计。

AutoModel：
- 通用的自动模型类，根据给定的 checkpoint 返回基础模型，不包含任何特定于任务的头部。通常用于获取模型的原始隐藏状态，或者想要自定义任务头部时使用。
- 使用场景：只需要模型的原始输出，或者进行特征提取、研究模型的中间表示、迁移学习中的预训练权重时使用。例如，获取每个标记的上下文嵌入。
- 只提供预训练模型的基础架构（比如Transformer的编码器或解码器），而没有针对具体任务（如分类、生成、问答等）的最后一层（称为“头部”或“head”）。
因此，使用AutoModel加载的模型只能输出原始的隐藏状态（hidden states tensor），需要自己添加合适的头部来完成特定任务。

AutoModelForSequenceClassification：
- 用于序列分类任务（例如，情感分析、文本分类）。它在基础模型之上添加了一个分类头部，将序列的汇总信息（通常是[CLS]标记的隐藏状态）映射到类别标签。
- 使用场景：当执行文本分类任务是时使用。例如，判断一段文本的情感是正面还是负面。

AutoModelForCausalLM：
- 用于因果语言建模（Causal Language Modeling），即自回归生成任务（例如，文本生成）。常用于生成模型，模型根据前面的标记预测下一个标记。
- 使用场景：当执行文本生成任务是时使用。例如，给定一个开头，让模型继续写下去。

注意：这些类会自动根据模型配置文件（config.json）自动实例化正确的模型架构。因此，即使使用 AutoModelForSequenceClassification 加载
一个没有在文本分类上微调的模型，它也会实例化一个带有分类头部的模型，不过这个分类头部的权重是随机初始化的。所以，通常我们使用与任务对应的
自动类来加载在特定任务上微调过的模型。

@see 02-rag-retrieve
"""
# AutoModel 通用基础模型，不包含任何任务
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # 输出隐藏状态

# 取[CLS]标记的表示作为整个句子的向量
# 或者取所有token的平均
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
mean_embedding = outputs.last_hidden_state.mean(dim=1)
final_embed = cls_embedding.numpy()


# AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 不使用 pipeline（@see 02-rag-retrieve），需要手动处理每个步骤
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 手动预处理
inputs = tokenizer("I love this movie!", return_tensors="pt")

# 手动推理
outputs = model(**inputs)
# with torch.no_grad():
#     outputs = model(**inputs)

# 手动后处理
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)  # 分类概率


# AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = './Qwen1.5-1.8B-Chat-LoRA'
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, device_map='auto')

# 模型设为评估状态，推理生成状态
# 训练状态（model.train()），微调训练阶段
# - model.eval()：关闭Dropout、BatchNorm使用运行统计量
# - model.train()：开启Dropout、BatchNorm使用批次统计量
model.eval()

# 定义测试示例
test_examples = [
    {
        "instruction": "使用中医知识正确回答适合这个病例的中成药。",
        "input": "肛门疼痛，痔疮，肛裂。"
    },
    {
        "instruction": "使用中医知识正确回答适合这个病例的中成药。",
        "input": "有没有能够滋养肝肾、清热明目的中药。"
    }
]

# 生成回答
for example in test_examples:
    context = f"Instruction: {example['instruction']}\nInput: {example['input']}\nAnswer: "
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(inputs.input_ids.to(model.device), max_length=512, num_return_sequences=1, no_repeat_ngram_size=2)


# 在生成过程中，通常不需要计算梯度，因此使用torch.no_grad()上下文管理器来加快计算并减少内存使用
with torch.no_grad():
    outputs = model.generate(
        # inputs["input_ids"],
        inputs.input_ids.to(model.device), # 输入的token ID
        max_length=100,
        num_return_sequences=1,  # 只生成一个回答
        no_repeat_ngram_size=2,  # 避免重复生成n-gram
        use_cache=False,  # 不使用缓存加速生成
        temperature=0.7,
        top_p=0.9,
        do_sample=True,  # 设置为True进行随机采样
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# 或使用以下 禁用梯度计算，节省内存和加速
torch.set_grad_enabled(False)
model.generate

# 解码生成的token ID，得到回答文本
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input: {example['input']}")
print(f"Output: {answer}\n")


"""
model.generate()：
- 是 transformers库提供的推理生成方法，内部实现了完整的生成循环（自回归生成），直到满足停止条件（如达到最大长度或生成结束符）。
返回的是生成的 token ids。并支持多种生成策略（如beam search、sampling等）。
- 适用于需要生成连续文本的任务，比如文本生成、对话、翻译等。

model(inputs)：
- 这种方式是直接调用模型，获取模型在给定输入下的输出（通常是logits）。是仅一次前向传播从中获取模型对当前输入序列的预测（每个位置的下一个token的logits）。
- 适用于非生成任务，比如分类、序列标注，或者只需要模型对当前输入的处理结果时。
- 在生成任务中，如果想要自己实现生成循环（例如使用特殊的解码策略），则会使用这种方式，但需要自己处理后续的token生成。
"""

