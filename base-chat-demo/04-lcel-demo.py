"""
LCEL，全称为 LangChain Expression Language，是一种专为 LangChain 框架设计的表达语言。它通过一种链式组合的方式，允许开发者使用清晰、
声明式的语法来构建语言模型驱动的应用流程。它是一种“函数式管道风格”的组件组合机制，用于连接各种可执行单元（Runnable）。这些单元包括提示模板、
语言模型、输出解析器、工具函数等。

LCEL 的核心组成有如下三点:
- Runnable 接口：LCEL 的一切基础单元都是 Runnable 对象（prompt, llm, OutputParser），它是一种统一的可调用接口，支持如下形式：
  - 所有 LCEL 组件都实现了 .invoke()、.stream()、.batch() 等标准方法，便于在同步、异步流式或批处理环境下调用。
  - 或使用 RunnableLambda/RunnableGenerator 封装的，非流式/流式输出的函数。

- 管道运算符 |：这是 LCEL 的语法符号。多个 Runnable 对象可以通过 | 串联起来，形成清晰的数据处理链。
  - 表示数据将依次传入提示模板、模型和输出解析器，最终输出结构化结果。

- PromptTemplate 与 OutputParser
  - LCEL 强调组件之间的职责明确，Prompt 只负责模板化输入，Parser 只负责格式化输出，Model 只负责推理。

"""