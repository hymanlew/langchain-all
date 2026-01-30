from transformers import AutoTokenizer, AutoModel
import torch

# 加载ColBERT模型和分词器
model_name = "bert-base-uncased"  # 可以替换为经过ColBERT微调的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# 查询和文档集合
query = "山西有哪些著名的旅游景点？"
documents = [
    "五台山是中国四大佛教名山之一，以文殊菩萨道场闻名。",
    "云冈石窟是中国三大石窟之一，以精美的佛教雕塑著称。", 
    "平遥古城是中国保存最完整的古代县城之一，被列为世界文化遗产。"]
# 编码函数
def encode_text(texts, max_length=128):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state  # 返回[CLS]和其他token的嵌入
# 编码查询和文档
query_embeddings = encode_text([query])  # 查询向量
doc_embeddings = encode_text(documents)  # 文档向量
# 计算余弦相似度
def calculate_similarity(query_emb, doc_embs):
    # ColBERT使用后期交互方法（即逐token比较方法），这里简化为采用余弦相似度进行比较
    query_emb = query_emb.mean(dim=1)  # 平均池化查询向量
    doc_embs = doc_embs.mean(dim=1)    # 平均池化文档向量
    query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)  # 单位化
    doc_embs = doc_embs / doc_embs.norm(dim=1, keepdim=True)
    scores = torch.mm(query_emb, doc_embs.t())  # 计算余弦相似度
    return scores.squeeze().tolist()
# 排序文档
scores = calculate_similarity(query_embeddings, doc_embeddings)
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
# 输出排序结果
print("Query:", query)
print("\nRanked Results:")
for rank, (doc, score) in enumerate(ranked_docs, start=1):
    print(f"{rank}. Score: {score:.4f} | Document: {doc}")

