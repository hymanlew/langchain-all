
具备以下功能：
根据对话类型将请求路由到适当的处理节点。
支持联网搜索，获取实时信息。
根据问题和对话历史生成优化的搜索提示词。
支持文件上传与处理。
利用编程专用的 LLM 解决代码相关问题。
基于提供的文档内容，总结生成答案。


项目结构如下：
.
├── .streamlit  # Streamlit 配置
│   └── config.toml
├── chains  # 智能体
│   ├── generate.py
│   ├── models.py
│   └── summary.py
├── graph   # 图结构
│   ├── graph.py
│   └── graph_state.py
├── upload_files    # 上传的文件
│   └── .keep
├── .env   # 环境变量配置
├── app.py  # Streamlit 应用
├── main.py # 命令行程序
└── requirements.txt    # 依赖

