import os
import json
import argparse
from datetime import datetime
import time
import yaml
import pandas as pd

# Ragas 提供了修改或替换默认提示为自定义提示词，用于评估的方式。但一般使用默认即可
from ragas.prompt import PydanticPrompt
from typing import Dict, List, Optional, Any, Union, Tuple
from datasets import Dataset, DatasetDict
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma, FAISS, Milvus
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# 导入评估相关模块
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from ragas.run_config import RunConfig
from ragas.metrics.base import Metric
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.dataset_schema import MultiTurnSample


class ConfigManager:
    """配置管理类，负责加载和管理配置"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                return {}
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 从环境变量覆盖配置
            for key, value in os.environ.items():
                if key.startswith("RAG_"):
                    config_key = key[4:].lower()
                    if config_key in config:
                        # 尝试转换为正确的类型
                        try:
                            config[config_key] = type(config[config_key])(value)
                        except (ValueError, TypeError):
                            config[config_key] = value
                        print(f"从环境变量覆盖配置: {config_key}={value}")
            
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def _validate_config(self) -> None:
        """验证配置的有效性"""
        required_sections = ["llm", "embeddings", "vector_store", "evaluation", "data", "output"]
        for section in required_sections:
            if section not in self.config:
                print(f"配置中缺少必要的部分: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

class DocumentProcessor:
    """文档处理类，负责加载、清洗和分割文档"""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.chunk_size = config.get("vector_store.chunk_size", 256)
        self.chunk_overlap = config.get("vector_store.chunk_overlap", 50)
        
    def load_documents(self, sources: List[str], source_types: List[str]) -> List[Document]:
        """从多种数据源加载文档"""
        documents = []
        max_docs = self.config.get("data.max_documents", 1000)
        
        for source, source_type in zip(sources, source_types):
            try:
                if source_type.lower() == "web":
                    loader = WebBaseLoader(source)
                elif source_type.lower() == "pdf":
                    loader = PyPDFLoader(source)
                elif source_type.lower() == "text":
                    loader = TextLoader(source, encoding="utf-8")
                elif source_type.lower() == "directory":
                    loader = DirectoryLoader(source, show_progress=True, recursive=True)
                else:
                    loader = UnstructuredFileLoader(source)
                
                docs = loader.load()
                # 清洗文档，同时补充文档的元数据
                for doc in docs:
                    doc.metadata['filename'] = doc.metadata['source']
                    if isinstance(doc, dict):
                        doc["page_content"] = self._clean_text(doc.get("page_content", ""))
                    else:
                        doc.page_content = self._clean_text(doc.page_content)
                    
                documents.extend(docs)
                print(f"成功加载 {len(docs)} 个文档片段，来源: {source}")
                
                # 检查是否达到最大文档数
                if len(documents) >= max_docs:
                    print(f"已达到最大文档数 {max_docs}，停止加载更多文档")
                    documents = documents[:max_docs]
                    break
                
            except Exception as e:
                print(f"加载文档失败，来源: {source}, 类型: {source_type}, 错误: {e}")
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """清洗文本内容"""
        # 移除多余的空白字符
        text = ' '.join(text.split())
        # 移除可能导致JSON解析错误的特殊字符
        text = text.replace('\u0000', '')
        # 处理其他特殊字符
        # ...
        return text
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成块"""
        try:
            # 根据文档类型选择合适的文本分割器
            if any('\n\n' in (doc.page_content if hasattr(doc, 'page_content') else doc.get('page_content', '')) for doc in documents):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
            else:
                splitter = CharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separator=" "
                )
            
            chunks = splitter.split_documents(documents)
            print(f"成功将 {len(documents)} 个文档分割成 {len(chunks)} 个块")
            return chunks
        except Exception as e:
            print(f"分割文档失败: {e}")
            return documents

class VectorStoreManager:
    """向量存储管理类，负责创建和管理向量数据库"""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.vector_store_type = config.get("vector_store.type", "chroma")
        self.persist_directory = config.get("vector_store.persist_directory", "./vector_store")
        self.embedding_model = self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> Any:
        """初始化嵌入模型"""
        try:
            model_name = self.config.get("embeddings.model_name", "BAAI/bge-small-zh-v1.5")
            cache_folder = self.config.get("embeddings.cache_folder", "./embedding_cache")
            normalize_embeddings = self.config.get("embeddings.normalize_embeddings", True)
            
            if model_name.startswith("BAAI/"):
                embed = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    cache_folder=cache_folder,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": normalize_embeddings}
                )
            else:
                embed = SentenceTransformerEmbeddings(
                    model_name=model_name,
                    cache_folder=cache_folder,
                    model_kwargs={"device": "cpu"}
                )
            return LangchainEmbeddingsWrapper(embed)
        except Exception as e:
            print(f"初始化嵌入模型失败: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> Tuple[Any, Any]:
        """创建向量存储"""
        try:
            if self.vector_store_type.lower() == "chroma":
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_directory
                )
                vector_store.as_retriever()
            elif self.vector_store_type.lower() == "faiss":
                vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embedding_model
                )
                # 保存 FAISS 索引
                if not os.path.exists(self.persist_directory):
                    os.makedirs(self.persist_directory)
                vector_store.save_local(self.persist_directory)
            elif self.vector_store_type.lower() == "milvus":
                # 从配置中获取 Milvus 连接参数，并初始化 MilvusManager（单例模式）
                milvus_config = self.config.get("vector_store.milvus_config", {})
                collection_name = milvus_config.get("collection_name", "langchain_demo")
                alias = milvus_config.get("alias", "prod")

                # 使用 LangChain 的 Milvus 类，但传入正确的连接参数
                # 先检查集合是否存在，如果不存在则创建
                vector_store = Milvus(
                    embedding_function=self.embedding_model,
                    connection_args={
                        # alias = alias,
                        "host": milvus_config.get("host", "localhost"),
                        "port": milvus_config.get("port", 19530),
                        "user": milvus_config.get("user", ""),
                        "password": milvus_config.get("password", ""),
                        "db_name": milvus_config.get("db_name", "default")
                    },
                    collection_name = collection_name
                )
                # 如果需要添加文档（在首次创建时）
                if documents:
                    vector_store.add_documents(documents)
            else:
                raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}")
            
            retriever = vector_store.as_retriever()
            print(f"成功创建 {self.vector_store_type} 向量存储，包含 {len(documents)} 个文档块")
            return vector_store, retriever
        except Exception as e:
            print(f"创建向量存储失败: {e}")
            raise
    
    def load_existing_vector_store(self) -> Tuple[Any, Any]:
        """加载已有的向量存储"""
        try:
            if self.vector_store_type.lower() == "chroma":
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model
                )
            elif self.vector_store_type.lower() == "faiss":
                vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            elif self.vector_store_type.lower() == "milvus":
                # 从配置中获取 Milvus 连接参数
                milvus_config = self.config.get("vector_store.milvus_config", {})
                
                # 使用 LangChain 的 Milvus 类，但传入正确的连接参数
                vector_store = Milvus(
                    embedding_function=self.embedding_model,
                    connection_args={
                        "host": milvus_config.get("host", "localhost"),
                        "port": milvus_config.get("port", 19530),
                        "user": milvus_config.get("user", ""),
                        "password": milvus_config.get("password", ""),
                        "db_name": milvus_config.get("db_name", "default")
                    },
                    collection_name=milvus_config.get("collection_name", "langchain_demo")
                )
            else:
                raise ValueError(f"不支持的向量存储类型: {self.vector_store_type}")
            
            retriever = vector_store.as_retriever()
            print(f"成功加载已有的 {self.vector_store_type} 向量存储")
            return vector_store, retriever
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return None, None

class RAGEvaluator:
    """RAG 评估器类，负责构建 RAG 链和执行评估"""
    def __init__(self, config: ConfigManager, retriever: Any, embedding_model: Any):
        self.config = config
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.llm = self._initialize_llm()
        self.rag_chain = self._build_rag_chain()
    
    def _initialize_llm(self) -> Any:
        """初始化语言模型"""
        try:
            provider = self.config.get("llm.provider", "ollama").lower()
            model = self.config.get("llm.model", "gwen2:7b-instruct-g4_0")
            temperature = self.config.get("llm.temperature", 0.1)
            
            if provider == "ollama":
                from langchain_ollama.llms import OllamaLLM
                llm = OllamaLLM(
                    model=model,
                    temperature=temperature
                )
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature
                )
            else:
                raise ValueError(f"不支持的 LLM 提供商: {provider}")
            return LangchainLLMWrapper(llm)
        except Exception as e:
            print(f"初始化语言模型失败: {e}")
            raise
    
    def _build_rag_chain(self) -> Any:
        """构建 RAG 链"""
        try:
            template = """
            您是问答任务的助理。使用以下检索到的上下文来回答问题。
            如果你不知道答案，就说你不知道。
            最多使用三句话，不超过100字，保持答案简洁。
            Question: {question}
            Context: {context}
            Answer: """
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            print("成功构建 RAG 链")
            return rag_chain
        except Exception as e:
            print(f"构建 RAG 链失败: {e}")
            raise
    
    def prepare_one_dataset(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> EvaluationDataset:
        """准备评估数据集，使用ragas的SingleTurnSample类构建现代格式的评估数据"""
        try:
            # 如果没有提供测试用例，使用默认的测试用例
            if test_cases is None:
                test_cases = [
                    {
                        "question": "艾伦图灵的论文叫什么?",
                        "ground_truth": ["计算机器与智能(Computing Machinery and Intelligence)"]
                    },
                    {
                        "question": "人工智能生成的画作在佳士得拍卖行卖了什么价格?",
                        "ground_truth": ["43.25万美元"]
                    },
                    {
                        "question": "目前企业在使用相关 AIGC 能力时，主要有哪五种方式?",
                        "ground_truth": ["直接使用、Prompt、LoRA、Finetune、Train"]
                    }
                ]

            samples = []

            for test_case in test_cases:
                question = test_case.get("question", "")
                ground_truth = test_case.get("ground_truth", [""])
                if not question:
                    continue
                
                # 生成答案和相关上下文，评估自己RAG系统的性能
                try:
                    answer = self.rag_chain.invoke(question)
                    retrieved_docs = self.retriever.get_relevant_documents(question)
                    contexts = [doc.page_content for doc in retrieved_docs]

                    # SingleTurnSample和MultiTurnSample是Ragas为了更清晰、更结构化地表示评估数据而引入的现代方式。
                    # 它们与传统的Dataset对象服务于同一个核心目的——为evaluate函数提供评估数据，但在数据组织和适用场景上有所不同。
                    # 它们主要区别在于：
                    # - SingleTurnSample 只包含一个用户输入、一个参考答案和一个生成的答案，
                    # - MultiTurnSample 包含多个用户输入、多个参考答案和生成的答案。代表人类、AI、工具和用于评估的预期结果之间的多轮交互。适用于在更复杂的交互中表示对话式代理以便进行评估。

                    # Rubrics 机制是让评估者根据多个评分标准进行多维度评分，最终取平均值作为综合得分
                    # 评估标准，用于指导评估器如何判断RAG系统的输出质量。在实际评估过程中，这些标准会被用于对生成的回答进行多维度评分。
                    rubrics = {
                        "accuracy": "Correct",  # 准确性要求为"正确"
                        "completeness": "High",  # 完整性要求为"高"
                        "fluency": "Excellent"  # 流畅度要求为"优秀"
                    }
                    sample = SingleTurnSample(
                        user_input=question,
                        retrieved_contexts=contexts,
                        response=answer,
                        # reference=ground_truth[0] if ground_truth else "",  # 将列表转为字符串
                        reference=", ".join(ground_truth) if ground_truth else "",
                        rubrics=rubrics
                    )
                    samples.append(sample)
                    print(f"处理测试问题: {question[:50]}...")
                except Exception as e:
                    print(f"处理测试问题失败: {question}, 错误: {e}")
                    # 即使出错也创建样本，以便记录错误信息
                    sample = SingleTurnSample(
                        user_input=question,
                        response="错误: 无法生成答案",
                        retrieved_contexts=["错误: 无法检索相关文档"],
                        reference=ground_truth[0] if ground_truth else ""
                    )
                    samples.append(sample)

            print(f"成功准备评估数据集，包含 {len(samples)} 个测试用例")
            return EvaluationDataset(samples=samples)
        except Exception as e:
            print(f"准备评估数据集失败: {e}")
            raise

    def prepare_multi_dataset(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> EvaluationDataset:
        """准备评估数据集，使用ragas的MultiTurnSample类构建现代格式的评估数据"""
        user_message = HumanMessage(content="What's the weather like in New York City today?")
        ai_initial_response = AIMessage(
            content="Let me check the current weather in New York City for you.",
            tool_calls=[ToolCall(name="WeatherAPI", args={"location": "New York City"})]
        )
        tool_response = ToolMessage(content="It's sunny with a temperature of 75°F in New York City.")
        ai_final_response = AIMessage(content="It's sunny and 75 degrees Fahrenheit in New York City today.")
        reference_response = "the current weather in real New York City"

        conversation = [
            user_message,
            ai_initial_response,
            tool_response,
            ai_final_response
        ]
        sample = MultiTurnSample(
            user_input=conversation,
            reference=reference_response,
        )
        return EvaluationDataset(samples=[sample])

    def prepare_old_dataset(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> EvaluationDataset:
        from ragas import EvaluationDataset
        dataset = []
        dataset.append(
            {
                "user_input": "query",
                "retrieved_contexts": [],
                "response": "response",
                "reference": "reference"
            }
        )
        return EvaluationDataset.from_list(dataset)

    def get_metrics_by_name(self, metric_names: List[str]) -> List[Metric]:
        """根据名称获取评估指标"""
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity
        }
        
        metrics = []
        for name in metric_names:
            if name in metric_map:
                metrics.append(metric_map[name])
            else:
                print(f"未知的评估指标: {name}")

        return metrics
    
    def evaluate(self, dataset: Dataset) -> Any:
        """执行评估"""
        try:
            # 获取评估配置
            metric_names = self.config.get("evaluation.metrics", ["context_precision", "context_recall", "faithfulness", "answer_relevancy"])
            max_retries = self.config.get("evaluation.max_retries", 10)
            max_wait = self.config.get("evaluation.max_wait", 60)
            log_tenacity = self.config.get("evaluation.log_tenacity", True)
            
            # 创建运行配置
            run_config = RunConfig(
                max_retries=max_retries,
                max_wait=max_wait,
                log_tenacity=log_tenacity
            )
            
            # 获取评估指标
            metrics = self.get_metrics_by_name(metric_names)
            
            # 执行评估
            start_time = time.time()
            print(f"开始执行 RAGAS 评估，使用指标: {', '.join(metric_names)}")
            result = evaluate(
                dataset=dataset,
                llm=self.llm,
                embeddings=self.embedding_model,
                run_config=run_config,
                metrics=metrics
            )
            
            end_time = time.time()
            print(f"评估完成，耗时: {end_time - start_time:.2f} 秒")
            return result
        except Exception as e:
            print(f"执行评估失败: {e}")
            raise

class ResultManager:
    """评估结果管理类，负责处理和导出评估结果"""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.output_dir = config.get("output.output_dir", "./evaluation_results")
        self.filename_prefix = config.get("output.filename_prefix", "ragas_eval_")
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
    
    def save_results(self, result: Any, dataset: Dataset) -> Dict[str, str]:
        """保存评估结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_files = {}
            
            # 转换结果为 DataFrame
            df = result.to_pandas()
            
            # 添加原始数据信息
            for i in range(len(df)):
                df.at[i, "question"] = dataset["question"][i]
                df.at[i, "answer"] = dataset["answer"][i]
            
            # 输出格式处理
            output_formats = self.config.get("output.format", ["console", "csv"])
            
            # 控制台输出
            if "console" in output_formats:
                print("评估结果摘要:")
                print(result)
                print("详细评估结果:")
                print(df)
            
            # CSV 格式输出
            if "csv" in output_formats:
                csv_file = os.path.join(self.output_dir, f"{self.filename_prefix}{timestamp}.csv")
                df.to_csv(csv_file, index=False, encoding="utf-8-sig")
                print(f"评估结果已保存到 CSV 文件: {csv_file}")
                output_files["csv"] = csv_file
            
            # JSON 格式输出
            if "json" in output_formats:
                json_file = os.path.join(self.output_dir, f"{self.filename_prefix}{timestamp}.json")
                # 转换 DataFrame 为 JSON 兼容格式
                result_dict = {
                    "summary": {k: float(v) for k, v in result.items()},
                    "details": df.to_dict(orient="records")
                }
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
                print(f"评估结果已保存到 JSON 文件: {json_file}")
                output_files["json"] = json_file
            
            # Excel 格式输出
            if "excel" in output_formats:
                excel_file = os.path.join(self.output_dir, f"{self.filename_prefix}{timestamp}.xlsx")
                df.to_excel(excel_file, index=False)
                print(f"评估结果已保存到 Excel 文件: {excel_file}")
                output_files["excel"] = excel_file
            
            return output_files
        except Exception as e:
            print(f"保存评估结果失败: {e}")
            raise

    # 自动生成 Markdown 格式的评估报告
    def generate_report(self, result: Any, dataset: Dataset, output_files: Dict[str, str]) -> None:
        """生成评估报告"""
        try:
            report_file = os.path.join(self.output_dir, f"{self.filename_prefix}report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("# RAG 系统评估报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## 评估摘要\n\n")

                # 写入评估指标摘要
                for metric, score in result.items():
                    f.write(f"- **{metric}**: {float(score):.4f}\n")
                
                f.write("\n## 评估详情\n\n")
                
                # 写入详细结果
                df = result.to_pandas()
                f.write("| 问题 | 答案 | 忠实度 | 相关性 | 上下文精度 | 上下文召回率 |\n")
                f.write("|------|------|--------|--------|------------|--------------|\n")
                
                for i in range(len(df)):
                    question = dataset["question"][i][:50] + "..." if len(dataset["question"][i]) > 50 else dataset["question"][i]
                    answer = dataset["answer"][i][:50] + "..." if len(dataset["answer"][i]) > 50 else dataset["answer"][i]
                    
                    # 填充各列数据
                    row_data = [question, answer]
                    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        if metric in df.columns:
                            row_data.append(f"{float(df.at[i, metric]):.4f}")
                        else:
                            row_data.append("-")
                    
                    f.write(f"| {' | '.join(row_data)} |\n")
                
                # 写入保存的文件信息
                f.write("\n## 保存的文件\n\n")
                for file_type, file_path in output_files.items():
                    f.write(f"- **{file_type.upper()}**: {file_path}\n")
                
            print(f"评估报告已生成: {report_file}")
        except Exception as e:
            print(f"生成评估报告失败: {e}")
            # 不抛出异常，继续执行

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="企业级 RAG 系统 RAGAS 评估工具")
    parser.add_argument("--config", type=str, default="rag-file/config.yaml", help="配置文件路径")
    parser.add_argument("--skip-data-prep", action="store_true", help="跳过数据准备，使用已有的向量存储")
    parser.add_argument("--test-cases", type=str, default="rag-file/test-cases", help="测试用例文件路径（JSON 格式）")
    parser.add_argument("--output-format", type=str, nargs="+", help="输出格式，可选值: console, csv, json, excel")
    parser.add_argument("--metrics", type=str, nargs="+", help="评估指标，可选值: faithfulness, answer_relevancy, context_recall, context_precision, context_relevancy, answer_correctness, answer_similarity")
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()

        # 初始化配置管理器
        config = ConfigManager(args.config)

        # 应用命令行参数覆盖配置
        if args.output_format:
            config.config["output"]["format"] = args.output_format
        if args.metrics:
            config.config["evaluation"]["metrics"] = args.metrics
        
        # 记录配置信息
        print("===== 配置信息 =====")
        print(f"LLM 提供商: {config.get('llm.provider')}")
        print(f"LLM 模型: {config.get('llm.model')}")
        print(f"嵌入模型: {config.get('embeddings.model_name')}")
        print(f"向量存储类型: {config.get('vector_store.type')}")
        print(f"向量存储路径: {config.get('vector_store.persist_directory')}")
        print(f"评估指标: {', '.join(config.get('evaluation.metrics', []))}")
        print(f"数据来源: {', '.join(config.get('data.sources', []))}")
        print(f"输出格式: {', '.join(config.get('output.format', []))}")
        print("===================")
        
        # 初始化向量存储管理器
        vector_store_manager = VectorStoreManager(config)
        
        if args.skip_data_prep:
            # 尝试加载已有的向量存储
            vector_store, retriever = vector_store_manager.load_existing_vector_store()
            if vector_store is None or retriever is None:
                print("无法加载已有的向量存储，将重新准备数据")
                args.skip_data_prep = False
        
        if not args.skip_data_prep:
            # 准备数据
            document_processor = DocumentProcessor(config)
            sources = config.get("data.sources", ["https://baike.baidu.com/item/AIGC-box"])
            source_types = config.get("data.source_types", ["web"] * len(sources))
            
            # 确保 sources 和 source_types 长度一致
            if len(sources) != len(source_types):
                print(f"sources 和 source_types 长度不一致，将默认所有来源为 web")
                source_types = ["web"] * len(sources)
            
            # 加载文档
            documents = document_processor.load_documents(sources, source_types)
            if not documents:
                print("没有加载到任何文档，无法继续执行")
                return
            
            # 分割文档
            chunks = document_processor.split_documents(documents)
            # 创建向量存储
            vector_store, retriever = vector_store_manager.create_vector_store(chunks)
        
        # 初始化 RAG 评估器
        evaluator = RAGEvaluator(config, retriever, vector_store_manager.embedding_model)
        
        # 加载本地备好的测试用例，或直接从本地文档中生成测试数据集（但需要人工审核）@see 20-rag-ragas-base.py
        test_cases = None
        if args.test_cases and os.path.exists(args.test_cases):
            try:
                with open(args.test_cases, "r", encoding="utf-8") as f:
                    test_cases = json.load(f)
                print(f"成功加载测试用例文件: {args.test_cases}")
            except Exception as e:
                print(f"加载测试用例文件失败: {e}")
        
        # 准备评估数据集
        dataset = evaluator.prepare_one_dataset(test_cases)
        
        # 执行评估
        result = evaluator.evaluate(dataset)
        
        # 管理评估结果
        result_manager = ResultManager(config)
        output_files = result_manager.save_results(result, dataset)
        
        # 自动生成 Markdown 格式的评估报告
        result_manager.generate_report(result, dataset, output_files)
        print("RAG 系统评估完成！")
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序执行失败: {e}")
        raise

"""
常见问题:
Q: 评估过程中出现 `Failed to parse output` 错误怎么办？
A: 这可能是由于知识内容中包含特殊字符导致 JSON 解析错误。请确保您的文档内容已经过清洗，移除了可能导致 JSON 解析错误的特殊字符。

Q: 评估速度很慢怎么办？
A: 评估速度受多种因素影响，包括模型大小、网络速度、计算机性能等。您可以尝试减少评估指标数量、减少测试用例数量或使用更轻量级的模型。

Q: 如何选择合适的评估指标？
A: 评估指标应根据业务需求选择：
- 智能客服：关注上下文召回率、精度、上下文相关性、答案相关性、忠实度
- 情感对话：关注上下文召回率、精度、上下文相关性
- 知识问答：关注忠实度、答案相关性、上下文精度
"""
if __name__ == "__main__":
    # python 20-rag-ragas-chain-prod.py --config config.yaml
    main()
