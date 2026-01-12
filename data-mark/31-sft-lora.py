from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

import torch
import json


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, device):
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        formatted_example = self.format_example(example)
        inputs = self.tokenizer(
            formatted_example["context"],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        labels = self.tokenizer(
            formatted_example["target"],
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        inputs['labels'] = labels['input_ids']
        # 确保所有张量在同一个设备上
        result = {key: val.squeeze().to(self.device) for key, val in inputs.items()}

        # 打印输入数据和形状
        print(f"Sample {idx}:")
        print(f"Context: {formatted_example['context']}")
        print(f"Target: {formatted_example['target']}")
        print({key: val.shape for key, val in result.items()})

        return result

    def format_example(self, example: dict) -> dict:
        context = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            context += f"Input: {example['input']}\n"
        context += "Answer: "
        target = example["output"]
        return {"context": context, "target": target}


# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器和模型
MODEL = 'Qwen/Qwen1.5-1.8B-Chat'
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)

# # 使用 BitsAndBytesConfig 进行量化配置
# quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0)
# model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, device_map="auto",
#                                              quantization_config=quantization_config,
#                                              torch_dtype=torch.float16)

model = model.to(device) # 把模型移到设备上

# 节省内存的一些配置
model.supports_gradient_checkpointing = True  # 支持梯度检查点功能，减少显存使用
model.gradient_checkpointing_enable()  # 启用梯度检查点功能，减少训练时的显存占用
model.enable_input_require_grads()  # 允许模型输入的张量需要梯度，支持更灵活的梯度计算
model.is_parallelizable = True  # 指定模型可以并行化处理
model.model_parallel = True  # 启用模型并行化，在多设备（如多GPU）上分布计算

# 配置并应用 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型：因果语言模型 (Causal LM)
    inference_mode=False, # 模式：训练模式 (inference_mode=False)
    r=8, # 低秩分解的秩：8
    lora_alpha=32, # 缩放因子：32
    lora_dropout=0.1, # dropout 概率：0.1
    target_modules=["q_proj", "v_proj"]  # q投影和 v投影模块，确保这些模块名与加载的模型兼容
)
model = get_peft_model(model, peft_config) # 应用LoRA配置

# 准备数据集
train_dataset = CustomDataset("data/cpmi.json", tokenizer, device)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',  # 训练结果保存的目录
    num_train_epochs=50,  # 训练的总轮数
    per_device_train_batch_size=4,  # 每个设备上的训练批次大小
    gradient_accumulation_steps=8,  # 梯度累积步数，在进行反向传播前累积多少步
    evaluation_strategy="no",  # 评估策略，这里设置为不评估
    save_strategy="epoch",  # 保存策略，每个 epoch 保存一次模型
    learning_rate=5e-5,  # 学习率
    fp16=True,  # 启用 16 位浮点数训练，提高训练速度并减少显存使用
    logging_dir='./logs',  # 日志保存目录
    dataloader_pin_memory=False,  # 禁用 pin_memory 以节省内存
)

# 自定义 Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # 从输入中取出标签
        print({key: val.shape for key, val in inputs.items()})  # 打印输入形状

        outputs = model(**inputs)  # 获取模型输出
        logits = outputs.logits  # 获取模型输出的logits
        shift_logits = logits[..., :-1, :].contiguous()  # 对logits进行偏移，准备计算交叉熵损失
        shift_labels = labels[..., 1:].contiguous()  # 对标签进行偏移，准备计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # 计算损失
        return (loss, outputs) if return_outputs else loss  # 根据参数返回损失和输出

# 定义 Trainer
trainer = CustomTrainer(
    model=model,  # 训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
)

# 开始训练
trainer.train()

# 创建保存模型的目录
import os
save_directory = 'Qwen1.5-1.8B-Chat'
os.makedirs(save_directory, exist_ok=True)

# 保存训练后的模型和配置文件
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# 将配置文件下载到模型目录中
config = model.config.to_dict()
config_path = os.path.join(save_directory, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"Model and configuration saved to {save_directory}")


# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "lora-adapter-dir")

