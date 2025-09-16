# 数据标注，基于 PyTorch, Transformers + Trainer (HuggingFace)
import os

import torch
import json
from tqdm import tqdm
from datetime import datetime
from transformers import (
    BertForTokenClassification,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Union

"""
**业务实现流程：**
- 纯人工标注，只适合小规模数据集、标注规则简单明确
- 初始用 1000 条人工标注数据训练基础模型
- 用该模型预测 10000 条新数据生成"预标注"
- 人工只修正其中30%明显错误（相比全人工标注节省70%工作量）
- 用新数据迭代训练更准的模型
- 保存模型
- 加载模型，进行标注任务
"""
# ====================== 初始化模型 ======================
"""
多任务学习（Multi-Task Learning, MTL）框架是一种让单个模型同时学习多个相关任务的机器学习范式。在训练时同时优化多个任务的损失函数。适合需要同时获取实体识别、分类、情感分析等结果的NLP流水线。
"""
class MultiTaskAnnotationSystem:
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        # 初始化所有任务模型，多个模型对应多个任务
        self.models = {
            # 命名实体识别
            "ner": BertForTokenClassification.from_pretrained(
                "bert-base-chinese",
                num_labels=len(ner_tags)  # 需预先定义标签
            ).to(device),
            # 情感分析
            "sentiment": BertForSequenceClassification.from_pretrained(
                "bert-base-chinese",
                num_labels=3  # 如正面/中性/负面
            ).to(device),
            # 文本分类
            "text_class": BertForSequenceClassification.from_pretrained(
                "bert-base-chinese",
                num_labels=len(text_classes)  # 如新闻分类标签
            ).to(device),
            # 任务分类
            "ticket_class": BertForSequenceClassification.from_pretrained(
                "bert-base-chinese",
                num_labels=len(ticket_categories)  # 工单类别
            ).to(device)
        }
        
        # 初始化pipeline（仅用于推理）
        self.pipelines = {
            "ner": pipeline(
                "token-classification",
                model=self.models["ner"],
                tokenizer=self.tokenizer,
                device=device,
                aggregation_strategy="simple"  # 实体合并策略
            ),
            "sentiment": pipeline(
                "text-classification",
                model=self.models["sentiment"],
                tokenizer=self.tokenizer,
                device=device
            ),
            "text_class": pipeline(
                "text-classification",
                model=self.models["text_class"],
                tokenizer=self.tokenizer,
                device=device
            ),
            "ticket_class": pipeline(
                "text-classification",
                model=self.models["ticket_class"],
                tokenizer=self.tokenizer,
                device=device
            )
        }

    # ====================== 训练逻辑 ======================
    def train(
        self,
        task: str,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        epochs: int = 3,
        batch_size: int = 16
    ):
        """训练指定任务的模型"""
        assert task in self.models, f"不支持的任务类型: {task}"
        
        training_args = TrainingArguments(
            output_dir=f"./results/{task}",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_dir=f"./logs/{task}",
            report_to="none"
        )
        
        # 兼容自定义多任务模型
        trainer = Trainer(
            model=self.models[task],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        self._save_model(task)

    def _save_model(self, task: str):
        """保存模型和tokenizer（必须！否则无法正确解码）"""
        self.models[task].save_pretrained(f"./models/{task}")
        self.tokenizer.save_pretrained(f"./models/{task}")


    def incremental_train(
        self,
        task: str,
        trained_model_path: str,  # 已训练好的模型路径（非checkpoint）
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        additional_epochs: int = 1,
        batch_size: int = 16
    ):
        """在已训练模型基础上增量训练

        Args:
            trained_model_path: 已训练模型的目录路径（包含pytorch_model.bin）
            additional_epochs: 需要追加训练的轮次
        """
        # 加载已有模型（而非checkpoint）
        self.models[task] = type(self.models[task]).from_pretrained(
            trained_model_path
        ).to(self.device)

        # 设置增量训练参数（关键区别：不恢复优化器状态）
        training_args = TrainingArguments(
            output_dir=f"./incremental_results/{task}",
            per_device_train_batch_size=batch_size,
            num_train_epochs=additional_epochs,
            resume_from_checkpoint=False,  # 与continue_train的关键区别
            overwrite_output_dir=True
        )

        trainer = Trainer(
            model=self.models[task],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        trainer.train()
        self._save_model(task)  # 保存增量训练后的模型

    # 版本控制增量训练
    def versioned_incremental_train(self, task: str, base_version: str, new_data: Dataset):
        """带版本管理的增量训练"""
        # 加载指定版本模型
        model_path = f"./model_repo/{task}/{base_version}"
        self.models[task] = type(self.models[task]).from_pretrained(model_path)

        # 训练并保存新版本
        self.incremental_train(task, model_path, new_data, additional_epochs=1)

        # 生成新版本号
        new_version = f"v{datetime.now().strftime('%Y%m%d')}"
        self._save_to_registry(task, new_version)


    # ====================== 标注逻辑 ======================
    def _format_entities(self, raw_entities):
        """将huggingface输出转为标准格式"""
        return [{
            "entity": ent["entity"],
            "word": ent["word"],
            "start": ent["start"],
            "end": ent["end"],
            "entity_group": ent["entity_group"]
        } for ent in raw_entities]


    def annotate_ner(self, text: str) -> List[Dict]:
        """命名实体识别"""
        entities = self.pipelines["ner"](text)
        return self._format_entities(entities)
    
    def annotate_sentiment(self, text: str) -> Dict:
        """情感分析"""
        result = self.pipelines["sentiment"](text[:512])  # 截断长文本
        return {"label": result[0]["label"], "score": result[0]["score"]}
    
    def annotate_text_class(self, text: str) -> Dict:
        """文本分类"""
        result = self.pipelines["text_class"](text[:512])
        return {"class": result[0]["label"], "score": result[0]["score"]}
    
    def annotate_ticket(self, text: str) -> Dict:
        """工单分类"""
        result = self.pipelines["ticket_class"](text[:512])
        return {"category": result[0]["label"], "score": result[0]["score"]}
    
    def extract_info(self, text: str, entities: List[str]) -> Dict:
        """信息抽取（基于NER的增强版）"""
        ner_results = self.annotate_ner(text)
        return {
            ent: [e for e in ner_results if e["entity_group"] == ent]
            for ent in entities
        }


    # ====================== 批量处理（处理百万级数据） ======================
    def batch_annotate_old(texts, batch_size=32):
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            results.extend([annotator.annotate(text) for text in batch])
        return results

    def batch_annotate(
        self,
        texts: List[str],
        tasks: List[str] = ["ner", "sentiment"]
    ) -> List[Dict]:
        """批量标注多任务"""
        results = []
        for text in tqdm(texts):
            result = {"text": text}
            if "ner" in tasks:
                result["entities"] = self.annotate_ner(text)
            if "sentiment" in tasks:
                result.update(self.annotate_sentiment(text))
            if "text_class" in tasks:
                result.update(self.annotate_text_class(text))
            if "ticket_class" in tasks:
                result.update(self.annotate_ticket(text))
            results.append(result)
        return results

    # ====================== 质量控制 - 置信度过滤 ======================
    def filter_low_confidence(
        self,
        results: List[Dict],
        min_score: float = 0.7
    ) -> Dict[str, List]:
        """过滤低置信度结果"""
        approved = []
        needs_review = []
        
        for res in results:
            if any(v.get("score", 1.0) < min_score for v in res.values() if isinstance(v, dict)):
                needs_review.append(res)
            else:
                approved.append(res)
        
        return {"approved": approved, "needs_review": needs_review}

    # ====================== 数据导出（CSV/JSONL格式） ======================
    def export_results(
        self,
        results: List[Dict],
        format: str = "jsonl",
        output_dir: str = "./outputs"
    ) -> str:
        """导出标注结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        path = f"{output_dir}/annotations_{timestamp}.{format}"
        
        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
        elif format == "csv":
            pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8-sig")
        
        return path


# ====================== 人工处理流程实现 ==================
class HumanReviewSystem:
    def __init__(self, annotator: MultiTaskAnnotationSystem):
        self.annotator = annotator
        self.review_queue = []
    
    def add_to_review(self, results: List[Dict]):
        """将需要审核的结果加入队列"""
        filtered = self.annotator.filter_low_confidence(results)
        self.review_queue.extend(filtered["needs_review"])
        
    def review_interface(self):
        """人工审核界面模拟"""
        print(f"\n待审核数量: {len(self.review_queue)}")
        for i, item in enumerate(self.review_queue[:5]):  # 示例显示前5条
            print(f"\n#{i+1} 文本: {item['text']}")
            for k, v in item.items():
                if k != "text":
                    print(f"- {k}: {v}")
            
            # 模拟人工修正
            correction = input("修正标注? (y/n): ")
            if correction.lower() == 'y':
                self._manual_correct(item)
    
    def _manual_correct(self, item: Dict):
        """人工修正逻辑"""
        print("可用操作:")
        print("1. 修正实体标签")
        print("2. 调整情感极性")
        choice = input("选择操作: ")
        
        if choice == "1":
            for i, ent in enumerate(item["entities"]):
                print(f"{i}. {ent['word']} → {ent['entity_group']}")
            idx = int(input("选择要修改的实体索引: "))
            new_label = input(f"新标签 (可选{ner_tags}): ")
            item["entities"][idx]["entity_group"] = new_label
        elif choice == "2":
            new_sentiment = input("新情感 (positive/negative): ")
            item["label"] = new_sentiment
            item["score"] = 1.0  # 人工标注设为最高置信度


# ====================== 使用示例 ======================
if __name__ == "__main__":

    # 0. 初始化系统（需预先定义标签）
    ner_tags = ["PER", "LOC", "ORG"]  # 实体标签
    text_classes = ["体育", "财经", "科技"]  # 文本分类标签
    ticket_categories = ["硬件", "软件", "网络"]  # 工单分类标签
    
    annotator = MultiTaskAnnotationSystem()

    # 1. 训练示例（需准备数据集）
    train_dataset = Dataset.from_pandas(pd.read_csv("ner_train.csv"))
    # 首次训练（完整训练）
    annotator.train("ner", train_dataset, epochs=3)  # 模型自动保存到./models/ner

    # 增量训练（在已训练模型上继续）
    annotator.incremental_train(
        task="ner",
        trained_model_path="./models/ner",  # 加载最终模型（非checkpoint）
        train_dataset=train_dataset,       # 可以是用新数据
        additional_epochs=2
    )

    # 2. 标注示例
    sample_text = "张三认为苹果公司的iPhone 14在北京市很受欢迎"
    print("实体识别:", annotator.annotate_ner(sample_text))
    print("情感分析:", annotator.annotate_sentiment(sample_text))
    print("信息抽取:", annotator.extract_info(sample_text, ["PER", "ORG"]))

    # 3. 批量处理
    batch_results = annotator.batch_annotate(
        texts=[sample_text, "李四在上海投诉路由器故障"],
        tasks=["ner", "sentiment", "ticket_class"]
    )
    print("批量结果:", batch_results)

    # 质量检测，人工复核
    filtered = annotator.filter_low_confidence(batch_results)
    reviewer = HumanReviewSystem(annotator)

    # 将需要人工审核的加入队列
    reviewer.add_to_review(batch_results)

    # 人工处理（实际应用时可集成到Flask/Django）
    reviewer.review_interface()

    # 获取最终结果
    approved = filtered["approved"]
    corrected = reviewer.review_queue  # 人工审核后的结果
    final_results = approved + corrected

    # 4. 导出结果
    annotator.export_results(batch_results, format="csv")



# ---------------- 示例2 ----------

# 1. 定义标注规范
annotation_template = {
    "project": "电商评论分析-v2",
    "fields": {
        "产品型号": {"type": "entity", "color": "#4285F4"},
        "功能点": {"type": "entity", "color": "#34A853"},
        "情感倾向": {"type": "classification", "options": ["正面", "中性", "负面"]},
        "价格区间": {"type": "freeform", "regex": r"\d{4,}元"}
    }
}

# 2. 生成标注界面配置（供标注工具使用）
def generate_ui_config():
    return {
        "ui": {
            "entity_types": [
                {"name": k, **v} 
                for k,v in annotation_template["fields"].items() 
                if v["type"] == "entity"
            ],
            "classification_schemas": [
                {"name": k, "options": v["options"]}
                for k,v in annotation_template["fields"].items()
                if v["type"] == "classification"
            ]
        }
    }

# 3. 实际标注处理
class EcommerceAnnotator:
    def __init__(self):
        self.ner_model = load_model("product_ner")
        self.sentiment_model = load_model("sentiment_v2")
    
    def annotate(self, text):
        # 基础标注
        entities = self.ner_model.predict(text)
        sentiment = self.sentiment_model.predict(text)
        
        # 自定义业务逻辑
        price_ranges = self._extract_price_ranges(text)
        
        return {
            "text": text,
            "annotations": {
                "产品型号": [e["text"] for e in entities if e["type"] == "PRODUCT"],
                "功能点": [e["text"] for e in entities if e["type"] == "FEATURE"],
                "情感倾向": sentiment,
                "价格区间": price_ranges,
                "metadata": {
                    "标注时间": datetime.now().isoformat(),
                    "标注人": "auto_system"
                }
            }
        }

