from typing import Optional, Literal, List
from datetime import date, datetime
from langchain.agents import create_agent
from pydantic import BaseModel, Field, field_validator
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import pytz  # 需要安装 pytz
from pydantic.v1 import create_model


# --- 1. 使用 Pydantic BaseModel 定义参数 Schema ---
# 生产推荐，这是最强大、最推荐的方式
class OrderQueryInput(BaseModel):
    """
    订单查询工具的输入参数模型。
    企业级提示：每个字段都用 Field 提供详细描述，便于LLM理解。
    """
    order_id: str = Field(
        ...,
        description="订单的唯一标识符，格式为 'ORD-YYYYMMDD-XXXXX'。",
        min_length=10,
        max_length=30
    )

    query_type: Literal["status", "details", "timeline"] = Field(
        default="status",
        description="查询类型：'status'(仅状态)，'details'(详情)，'timeline'(时间线)。"
    )

    customer_id: Optional[str] = Field(
        default=None,
        description="内部客户ID，用于增强验证和权限检查（如VIP客户可查更多信息）。",
        pattern=r'^CUST-\d+$'  # 正则验证格式
    )

    timezone: Optional[str] = Field(
        default="Asia/Shanghai",
        description="用于显示时间的IANA时区标识符，例如 'America/New_York'。"
    )
    date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="查询的日期，格式必须为 YYYY-MM-DD。"
    )

    # --- 企业级增强：自定义验证器 ---
    @field_validator('order_id')
    def validate_order_id_format(cls, v):
        """验证订单ID符合公司内部格式。"""
        if not v.startswith('ORD-'):
            raise ValueError('订单ID必须以 "ORD-" 开头')
        # 可在此添加更复杂的逻辑，如校验日期部分
        return v.upper()  # 统一转为大写

    @field_validator('timezone')
    def validate_timezone(cls, v):
        """验证时区标识符是否有效。"""
        if v and v not in pytz.all_timezones:
            raise ValueError(f"'{v}' 不是有效的IANA时区标识符")
        return v

 class OrderQueryOutput(BaseModel):
    city: str = Field(..., description='查询的城市')
    temperature: float = Field(..., description='摄氏度温度值', ge = -50, le = 60)
    forecast: List[str] = Field(..., description='未来两天的天气预测')
    is_reliable: bool = Field(default=True, description='数据是否可靠')

    @field_validator('city')
    def validate_condition(cls, v):
        pass


# --- 2. 使用 @tool 装饰器，并指定 args_schema ---
# name：指定工具的名称
# description：描述工具的功能，至关重要！Agent的LLM通过它理解何时调用该工具。描述应清晰说明工具的用途、适用场景和输入含义
# return_direct：控制是否直接返回结果，默认False。为True时，工具结果将绕过Agent的思考，直接作为最终答案返回用户。适用于无需后续处理的简单查询。
# args_schema：定义输入参数的结构化模式。用于严格定义工具接受的参数名称、类型、描述和验证规则。确保LLM生成正确的参数格式，并提供运行时验证。
# parse_docstring：是否解析函数内部的注释字符串，默认False。为True时，会尝试从函数的docstring中提取description和参数信息。

# parse_docstring 在生产场景下不可用，因为它会自动收集函数中的参数，并拼接成 function-call 格式字符串。所以这种不精准，很少用
# response_format：强制工具返回一个固定结构，比如JSON，而不是自由文本。好处是让输出变成机器可读、可预测的数据，方便后续步骤处理。
# 若不指定格式，则默认返回文本，也可能每次格式可能不同。参数主要接受两种值：字符串字面量（json）或 Pydantic BaseModel 类。

# 通过字典定义 args_schema，简单场景
args_schema_dict = {
    "city_name": {
        "type": "string",  # 参数类型
        "description": "要查询天气的城市名称，必须是中国的城市，例如：北京、上海。"
    },
    "date": {
        "type": "string",
        "description": "查询的日期，格式为 YYYY-MM-DD。默认为今天。",
        "default": None  # 可以指定默认值
    }
}
@tool(
    name_or_callable="query_tool",
    description="查询详细订单信息",
    args_schema=OrderQueryInput,
    return_direct=False
)
def query_order_tool(order_id: str, query_type: str = "status",
                     customer_id: Optional[str] = None,
                     timezone: str = "Asia/Shanghai") -> str:
    """
    根据订单ID和查询类型，从企业ERP系统获取订单信息。

    严格使用提供的参数，特别是遵循 order_id 的格式要求。
    """
    # 模拟企业级工具：带权限和逻辑校验
    print(f"[TOOL LOG] 查询订单: ID={order_id}, 类型={query_type}, 客户={customer_id}")

    # 模拟根据不同类型返回
    base_info = f"订单 {order_id} 状态：已发货。客户：{customer_id or '未知'}。"
    if query_type == "status":
        return f"{base_info} 查询类型：状态概要。"
    elif query_type == "details":
        return f"{base_info} 查询类型：详细信息。产品列表：[...] 金额：$999。"
    elif query_type == "timeline":
        return f"{base_info} 查询类型：时间线。创建：2024-01-01 -> 发货：2024-01-05。"
    else:
        # 理论上 args_schema 已限制，此为防御性代码
        return f"错误：不支持的查询类型 '{query_type}'。"



"""
企业级开发建议：
继承 BaseTool 类（企业级推荐）

首选 Pydantic BaseModel：
类型安全：mypy/pyright 等工具可静态检查。
自文档化：Field 的 description 直接帮助 LLM 理解参数。
验证丰富：可使用 @field_validator、@model_validator 实现复杂业务规则。
"""
from typing import Type, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from abc import ABC, abstractmethod


# 1. 抽象配置接口（支持依赖注入）
class RiskServiceConfig:
    def __init__(self, api_base_url: str, api_key: str, timeout: int = 30):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.timeout = timeout


# 2. 抽象缓存层（便于替换实现）
class RiskCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600): pass


class MemoryRiskCache(RiskCache):
    def __init__(self):
        self._store = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600):
        self._store[key] = value


# 3. 参数Schema作为内部类（封装性更好）
class CustomerRiskAssessmentTool(BaseTool):
    name: str = "assess_customer_risk"
    description: str = """
    调用企业风控引擎，评估指定客户的风险等级和信用评分。
    适用于贷前审批、交易监控等场景。
    """
    return_direct = False
    args_schema: Type[BaseModel] = None
    response_format = 'json' # 或 basemodel

    # 依赖通过构造器注入，而非全局变量
    def __init__(self,
                 config: RiskServiceConfig,
                 cache: Optional[RiskCache] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.cache = cache or MemoryRiskCache()
        self.logger = logging.getLogger(self.__class__.__name__)

        # 如果参数比较少时，没必要单独定义一个类，可直接使用 create-model 动态生成一个类
        self.args_schema = create_model("OrderQueryInput", order_id=(str, Field(..., description='<UNK>ID<UNK>')))
        # 动态创建参数Schema类，可基于配置变化
        self.args_schema = OrderQueryInput
        self.response_format = OrderQueryOutput

    # 4. 核心执行逻辑（专注业务，基础设施被分离）
    def _run(self, customer_id: str, force_refresh: bool = False, **kwargs: Any) -> str:
        """同步执行"""
        return self._assess_risk(customer_id, force_refresh)

    async def _arun(self, customer_id: str, force_refresh: bool = False, **kwargs: Any) -> str:
        """异步执行 - 生产环境推荐"""
        return await self._assess_risk_async(customer_id, force_refresh)

    # 5. 私有方法，实现具体业务逻辑（可测试）
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.exceptions.Timeout),
        reraise=True
    )
    def _call_risk_api(self, customer_id: str) -> dict:
        """调用风控API，自带重试机制"""
        self.logger.info(f"调用风控API评估客户 {customer_id}")
        # 或直接使用异步调用的 httpx
        response = requests.post(
            f"{self.config.api_base_url}/risk/v1/assess",
            json={"customerId": customer_id},
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()

    def _assess_risk(self, customer_id: str, force_refresh: bool) -> str:
        """风险评估主逻辑"""
        # 检查缓存
        cache_key = f"risk_{customer_id}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                self.logger.debug(f"命中缓存 for {customer_id}")
                return f"[缓存结果] {cached}"

        try:
            # 调用API
            data = self._call_risk_api(customer_id)
            risk_level = data.get("riskLevel", "UNKNOWN")
            score = data.get("score", 0)

            # 业务规则：根据分数分类
            if score >= 80:
                category = "高风险"
            elif score >= 50:
                category = "中等风险"
            else:
                category = "低风险"

            result = f"客户 {customer_id}: {category} ({risk_level}, 评分{score})"

            # 更新缓存
            self.cache.set(cache_key, result, ttl=300)  # 缓存5分钟
            return OrderQueryOutput()

        except requests.exceptions.HTTPError as e:
            self.logger.error(f"风控API HTTP错误: {e}")
            if e.response.status_code == 404:
                return f"客户 {customer_id} 不存在于风控系统中"
            elif e.response.status_code == 403:
                return "权限不足，无法访问风控系统"
            else:
                return f"风控系统错误: HTTP {e.response.status_code}"
        except requests.exceptions.Timeout:
            self.logger.error("风控API请求超时")
            return "风控系统响应超时，请稍后重试"
        except Exception as e:
            self.logger.exception(f"评估客户 {customer_id} 风险时发生未知错误")
            return "系统暂时不可用，请联系管理员"

    async def _assess_risk_async(self, customer_id: str, force_refresh: bool) -> str:
        """异步版本 - 在实际项目中可能调用异步HTTP客户端"""
        # 这里为了示例，我们简单调用同步版本
        # 生产环境中可使用 aiohttp 或 httpx 实现真正的异步
        return self._assess_risk(customer_id, force_refresh)


# 6. 使用工厂函数创建工具实例（便于依赖管理）
def create_risk_assessment_tool(env: str = "production") -> CustomerRiskAssessmentTool:
    """工具工厂：根据环境创建配置好的工具实例"""
    if env == "production":
        config = RiskServiceConfig(
            api_base_url="https://risk-api.prod.company.com",
            api_key=os.getenv("RISK_API_KEY_PROD"),
            timeout=30
        )
        # 生产环境使用Redis缓存
        cache = RedisRiskCache(host=os.getenv("REDIS_HOST"))
    else:
        config = RiskServiceConfig(
            api_base_url="https://risk-api.staging.company.com",
            api_key=os.getenv("RISK_API_KEY_STAGING"),
            timeout=10
        )
        cache = MemoryRiskCache()  # 测试环境用内存缓存

    return CustomerRiskAssessmentTool(config=config, cache=cache)


# 7. 在Agent或LangGraph中使用
def setup_agent_with_tools():
    # 从工厂获取工具实例
    risk_tool = create_risk_assessment_tool(env=os.getenv("APP_ENV", "development"))

    # 可以轻松组合多个工具
    other_tool = AnotherBaseToolSubclass(...)

    llm = ChatOpenAI(model="gpt-4")
    agent = create_agent(
        model=llm,
        tools=[risk_tool, other_tool],  # 工具实例列表
        system_prompt=...
    )
    return agent


# --- 测试工具调用 (模拟LLM绑定后的调用) ---
def test_base_model_tool():
    print("=== 测试装饰器 @tool 调用 Schema ===")

    print("=== 测试 Pydantic BaseModel 参数 Schema ===")

    # 模拟一个来自LLM的、结构良好的工具调用请求
    test_request = {
        "name": "query_order_tool",
        "args": {
            "order_id": "ORD-20240415-12345",  # 符合格式
            "query_type": "details",
            "customer_id": "CUST-VIP-001",
            "timezone": "America/New_York"
        }
    }

    try:
        # 这里模拟了 LangChain 内部如何调用工具
        # 实际在 Agent 中，这个过程是自动的
        result = query_order_tool.invoke(test_request["args"])
        print(f"✅ 工具调用成功:\n{result}\n")
    except Exception as e:
        print(f"❌ 工具调用失败: {type(e).__name__}: {e}\n")

    # 测试验证失败的情况
    print("--- 测试参数验证失败 ---")
    bad_request = {
        "name": "query_order_tool",
        "args": {
            "order_id": "INVALID-123",  # 不以 'ORD-' 开头
            "query_type": "unknown_type"  # 不在 Literal 中
        }
    }

    try:
        # Pydantic 会在 invoke 时自动验证
        result = query_order_tool.invoke(bad_request["args"])
        print(f"结果: {result}")
    except Exception as e:
        print(f"预期内的验证错误: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_base_model_tool()



# # 创建并添加 ToolNode，配置错误处理策略：出现异常时，向LLM发送自定义错误信息
# tool_node = ToolNode(
#     tools=TOOLS,
#     handle_tool_errors=lambda e: f"工具执行失败。系统错误信息: {str(e)}。请根据现有信息继续或建议用户联系客服。"
# )