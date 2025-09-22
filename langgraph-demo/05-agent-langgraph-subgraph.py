# 示例: 使用子图实现日志分析功能 sub graph.py
from typing import List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated
from operator import add


# 定义日志的结构, 接收来自系统生成的日志文件
class Logs(TypedDict):
    id: str  # 日志的唯一标识符
    question: str  # 问题文本
    docs: Optional[List]  # 可选的相关文档列表
    answer: str  # 回答文本
    grade: Optional[int]  # 可选的评分
    grader: Optional[str]  # 可选的评分者
    feedback: Optional[str]  # 可选的反馈信息


# 定义故障分析状态的结构
class FailureAnalysisState(TypedDict):
    docs: List[Logs]  # 日志列表
    failures: List[Logs]  # 失败的日志列表
    fa_summary: str  # 故障分析总结


# 获取失败的日志
def get_failures(state):
    docs = state["docs"]  # 从状态中获取日志
    failures = [doc for doc in docs if "grade" in doc]  # 筛选出包含评分的日志
    return {"failures": failures}  # 返回包含失败日志的字典


# 生成故障分析总结
def generate_summary(state):
    failures = state["failures"]  # 从状态中获取失败的日志
    # 添加函数，内容可调用大模型生成总结。这里先给个固定的总结内容
    # fa_summary=summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary}  # 返回包含总结的字典


# 创建故障分析的状态图
fa_builder = StateGraph(FailureAnalysisState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

# 定义问题总结状态的结构
class QuestionSummarizationState(TypedDict):
    docs: List[Logs]  # 日志列表
    qs_summary: str  # 问题总结
    report: str  # 报告


# 生成问题总结
def generate_summary(state):
    docs = state["docs"]  # 从状态中获取日志
    # 添加函数:summary=summarize(docs)
    summary = "questions focused on usage of ChatOllama and chroma vector store."
    return {"qs_summary": summary}  # 返回包含总结的字典


# 发送总结到Slack
def send_to_slack(state):
    qs_summary = state["qs_summary"]  # 从状态中获取问题总结
    # 添加所数: report=report_generation(qs_summary)
    report = "foo bar baz"  # 固定的报告内容
    return {"report": report}  # 返回包含报告的字典


# 格式化报告以便在Slack中发送
def format_report_for_slack(state):
    report = state["report"]  # 从状态中获取报告
    # 添加所数: formatted report=report_format(report)
    formatted_report = "foo bar"  # 固定的格式化报告内容
    return {"report": formatted_report}  # 返回包含格式化报告的字典


# 创建问题总结的状态图
qs_builder = StateGraph(QuestionSummarizationState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_node("format_report_for_slack", format_report_for_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", "format_report_for_slack")
qs_builder.add_edge("format_report_for_slack", END)  # 添加边:从格式化报告到结束

# Entry Graph，总流程的入口


class EntryGraphState(TypedDict):
    source_logs: Annotated[List[dict], add]  # 原始的日志内容是 dict 格式的。第一步
    docs: Annotated[List[Logs], add]  # 将原始系统日志内容转换成 Logs 对象格式。第二步
    fa_summary: str  # 故障分析总结。第三步
    report: str  # 问题总结报告。第三步


# demo logs，模拟将原始系统日志文件转换成了文档
question_answer = Logs(
    id="1",
    question="如何导入ChatOpenAI?",
    answer="要导入ChatOpenAI,使用:'from langchain_openai import ChatOpenAI.'",
)

question_answer_feedback = Logs(
    id="2",
    question="如何使用Chroma向量存储?",
    answer="要使用Chroma, 请定义: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="文档相似性回顾",
    feedback="检索到的文档一般讨论了向量存储，但没有专门讨论Chroma",
)


def convert_logs_to_docs(state):
    # Get source logs
    source_logs = state["source_logs"]
    # 模拟实际转换的动作，已经生成了文档
    docs = [question_answer, question_answer_feedback]
    return {"docs": docs}


# 添加子图到总流程中
entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("convert_logs_to_docs", convert_logs_to_docs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "convert_logs_to_docs")
entry_builder.add_edge("convert_logs_to_docs", "failure_analysis")
entry_builder.add_edge("convert_logs_to_docs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

# 编译图
raw_logs = [{"foo": "bar"}, {"foo": "baz"}]
app = entry_builder.compile()
print(app.invoke({"source_logs": raw_logs}, debug=False))

# 将生成的图片保存到文件
graph_png = app.get_graph(xray=1).draw_mermaid_png()
with open("sub_graph.png", "wb") as f:
    f.write(graph_png)
