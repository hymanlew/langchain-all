import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ragas.embeddings import HuggingfaceEmbeddings
from ragas import evaluate

# 准备评估用的LLM（使用GPT-3.5）
llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-3.5-turbo"))

# 准备数据集
data = {
    "question": [
        "Who is the main character in Black Myth: Wukong?",
        "What are the special features of the combat system in Black Myth: Wukong?",
        "How is the visual quality of Black Myth: Wukong?",
    ],
    "answer": [
        "The main character in Black Myth: Wukong is Sun Wukong, based on the Chinese classic 'Journey to the West' but with a new interpretation. This version of Sun Wukong is more mature and brooding, showing a different personality from the traditional character.",
        "Black Myth: Wukong's combat system combines Chinese martial arts with Soulslike game features, including light and heavy attack combinations, technique transformations, and magic systems. Notably, Wukong can transform between different weapon forms during combat, such as his iconic staff and nunchucks, and use various mystical abilities.",
        "Black Myth: Wukong is developed using Unreal Engine 5, showcasing stunning visual quality. The game's scene modeling, lighting effects, and character details are all top-tier, particularly in its detailed recreation of traditional Chinese architecture and mythological settings.",
    ],
    "contexts": [
        [
            "Black Myth: Wukong is an action RPG developed by Game Science, featuring Sun Wukong as the protagonist based on 'Journey to the West' but with innovative interpretations. In the game, Wukong has a more composed personality and carries a special mission.",
            "The game is set in a mythological world, telling a new story that presents a different take on the traditional Sun Wukong character."
        ],
        [
            "The game's combat system is heavily influenced by Soulslike games while incorporating traditional Chinese martial arts elements. Players can utilize different weapon forms, including the iconic staff and other transforming weapons.",
            "During combat, players can unleash various mystical abilities, combined with light and heavy attacks and combo systems, creating a fluid and distinctive combat experience. The game also features a unique transformation system."
        ],
        [
            "Black Myth: Wukong demonstrates exceptional visual quality, built with Unreal Engine 5, achieving extremely high graphical fidelity. The game's environments and character models are meticulously crafted.",
            "The lighting effects, material rendering, and environmental details all reach AAA-level standards, perfectly capturing the atmosphere of an Eastern mythological world."
        ]
    ]
}

dataset = Dataset.from_dict(data)

print("\n=== Ragas评估指标说明 ===")
print("\n1. Faithfulness（忠实度）")
print("- 评估生成的答案是否忠实于上下文内容")
print("- 通过将答案分解为简单陈述，然后验证每个陈述是否可以从上下文中推断得出")
print("- 该指标仅依赖LLM，不需要embedding模型")

# 评估Faithfulness
faithfulness_metric = [Faithfulness(llm=llm)]
print("\n正在评估忠实度...")
faithfulness_result = evaluate(dataset, faithfulness_metric)
scores = faithfulness_result['faithfulness']
mean_score = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"忠实度评分: {mean_score:.4f}")

print("\n2. AnswerRelevancy（答案相关性）")
print("- 评估生成的答案与问题的相关程度")
print("- 使用embedding模型计算语义相似度")
print("- 我们将比较开源embedding模型和OpenAI的embedding模型")

# 设置两种embedding模型
opensource_embedding = LangchainEmbeddingsWrapper(
    HuggingfaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
openai_embedding = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-ada-002"))

# 创建答案相关性评估指标
opensource_relevancy = [AnswerRelevancy(llm=llm, embeddings=opensource_embedding)]
openai_relevancy = [AnswerRelevancy(llm=llm, embeddings=openai_embedding)]

print("\n正在评估答案相关性...")
print("\n使用开源Embedding模型评估:")
opensource_result = evaluate(dataset, opensource_relevancy)
scores = opensource_result['answer_relevancy']
opensource_mean = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {opensource_mean:.4f}")

print("\n使用OpenAI Embedding模型评估:")
openai_result = evaluate(dataset, openai_relevancy)
scores = openai_result['answer_relevancy']
openai_mean = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {openai_mean:.4f}")

# 比较两种embedding模型的结果
print("\n=== Embedding模型比较 ===")
diff = openai_mean - opensource_mean
print(f"开源模型评分: {opensource_mean:.4f}")
print(f"OpenAI模型评分: {openai_mean:.4f}")
print(f"差异: {diff:.4f} ({'OpenAI更好' if diff > 0 else '开源模型更好' if diff < 0 else '相当'})")
