import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, wait_exponential, stop_after_attempt
from config import ONE_API_BASE_URL, ONE_API_MASTER_KEY

logger = logging.getLogger(__name__)

# 定义工作流状态
class WorkflowState(TypedDict):
    user_input: str
    model: str
    answer: str

# 创建指向 One‑API 的 LLM 客户端
# 管理页面中启用负载均衡，会自动分配请求（默认轮询），还支持设置渠道的优先级和权重，故障转移
llm_client = ChatOpenAI(
    base_url=ONE_API_BASE_URL,
    api_key=ONE_API_MASTER_KEY,  # 使用 One‑API 的主密钥
    temperature=0.7,
)

# 带重试的 LLM 调用，模型在每次调用时传入
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def safe_llm_invoke(messages, model: str):
    bound_client = llm_client.bind(model=model)
    return bound_client.invoke(messages)

# 工作流节点函数
def answer_node(state: WorkflowState) -> WorkflowState:
    logger.info(f"处理输入: {state['user_input']}, 使用模型: {state['model']}")
    try:
        messages = [
            SystemMessage(content="你是一个专业的助手。"),
            HumanMessage(content=state["user_input"])
        ]
        response = safe_llm_invoke(messages, state["model"])
        return {"answer": response.content}
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return {"answer": f"请求失败，请重试。错误信息: {str(e)}"}

# 构建图
workflow = StateGraph(WorkflowState)
workflow.add_node("answer", answer_node)
workflow.add_edge(START, "answer")
workflow.add_edge("answer", END)
graph = workflow.compile()