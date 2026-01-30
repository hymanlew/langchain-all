from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. 加载预训练的 BERT 模型（用于句对相关性计算）
model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # 适用于检索任务
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. 查询和山西文旅相关的文档
query = "山西有哪些著名的旅游景点？"
documents = [
    "五台山是中国四大佛教名山之一，以文殊菩萨道场闻名。",
    "云冈石窟是中国三大石窟之一，以精美的佛教雕塑著称。",
    "平遥古城是中国保存最完整的古代县城之一，被列为世界文化遗产。",
]

# 3. 计算相关性分数
def encode_and_score(query, docs):
    scores = []
    for doc in docs:
        inputs = tokenizer(query, doc, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits[0][0].item()
            scores.append(score)
    return scores

# 4. 获取排序结果
scores = encode_and_score(query, documents)
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# 5. 输出结果
print("查询:", query)
print("\n排序结果:")
for rank, (doc, score) in enumerate(ranked_docs, start=1):
    print(f"{rank}. 相关性分数: {score:.4f} | 文档: {doc}")
