# AI 多智能体旅游助手

## 项目简介

AI 多智能体旅游助手是一个基于 LangGraph、LangChain、Streamlit 和 PostgreSQL 构建的智能旅游规划系统。

项目将一次完整的旅游规划任务拆分为多个 Agent 协同完成，包括景点推荐、酒店推荐、交通规划、行程生成、AI 审核和人工审核。系统集成 Tavily Search 和高德地图 API 获取外部实时信息，并使用 PostgreSQL 对旅行任务、方案版本、审核记录和 Agent 执行日志进行持久化存储。

相比简单的大模型问答式旅游规划，本项目更强调多智能体协作、工具调用、审核闭环、方案版本管理和运行过程可观测性。

## 核心功能

- 多智能体旅游规划
- 景点推荐
- 酒店推荐
- 城际交通查询
- 市内交通规划
- 最终行程生成
- AI Reviewer 自动审核
- Human-in-the-loop 人工审核
- 方案打回与重新生成
- 历史版本管理
- PostgreSQL 数据持久化
- Agent 执行日志记录
- Harness Dashboard 运行监控

## 技术栈

- Python
- LangGraph
- LangChain
- Streamlit
- PostgreSQL
- SQLAlchemy
- Pydantic
- Tavily Search API
- 高德地图 API
- python-dotenv

## 系统架构

系统整体流程如下：

```text
用户输入旅游需求
        ↓
Supervisor Agent
        ↓
Attraction Agent
        ↓
Hotel Agent
        ↓
Transport Agent
        ↓
Planner Agent
        ↓
AI Reviewer Agent
        ↓
Human Reviewer
        ↓
PostgreSQL 持久化存储
```

各 Agent 职责说明：

| Agent | 职责 |
| --- | --- |
| Supervisor Agent | 控制整体流程，决定下一个执行节点 |
| Attraction Agent | 根据目的地和用户偏好推荐景点 |
| Hotel Agent | 基于目的地、景点和预算推荐住宿 |
| Transport Agent | 查询城际交通和市内路线，生成交通建议 |
| Planner Agent | 汇总景点、酒店、交通信息，生成完整旅游方案 |
| AI Reviewer Agent | 自动审核方案是否合理、可执行 |
| Human Reviewer | 允许用户人工确认、打回并提出修改意见 |

## 项目结构

```text
py_project05_旅游小助手/
├── agents/
│   ├── attraction.py
│   ├── hotel.py
│   ├── planner.py
│   ├── reviewer.py
│   ├── supervisor.py
│   └── transport.py
│
├── config/
│   ├── database.py
│   ├── graph.py
│   ├── human_reviewer.py
│   ├── model.py
│   ├── route.py
│   └── state.py
│
├── database/
│   ├── _init_db.py
│   ├── harness_stats.py
│   ├── models.py
│   └── repository.py
│
├── harness/
│   ├── runtime.py
│   └── trace.py
│
├── tools/
│   ├── attraction.py
│   ├── hotel.py
│   ├── intercity_transport.py
│   └── transport.py
│
├── app.py
├── harness_dashboard.py
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## 核心流程

### 1. 用户输入旅游需求

用户在 Streamlit 页面输入出发地、目的地、出发日期、返程日期、出行人数、总预算、旅游偏好和其他限制条件。系统会自动计算旅游天数，并构造初始 TravelState。

### 2. 多 Agent 协同规划

系统通过 LangGraph 编排多个 Agent。Attraction Agent 负责景点推荐，Hotel Agent 负责住宿推荐，Transport Agent 负责交通方案，Planner Agent 负责汇总生成完整行程，AI Reviewer Agent 负责自动审核，Human Reviewer 负责人工确认和反馈。

### 3. AI 审核与自动修改

AI Reviewer 会检查旅游方案是否存在行程过满、景点安排不合理、交通路线不合理、酒店位置不合适、预算可能超支、不符合用户偏好或违反用户限制等问题。

如果 AI 审核不通过，系统会将审核意见返回给 Planner，由 Planner 根据审核意见重新生成方案。

### 4. 人工审核与版本迭代

当方案进入人工审核环节后，用户可以选择同意方案，也可以打回方案并输入修改意见。如果用户打回方案，系统会基于用户修改意见重新生成方案，并保存为新的版本。

### 5. 历史版本管理

系统支持查看历史方案版本，包括方案内容、AI 审核结果、人工审核意见和是否为最终版本。用户可以在页面中将某个版本设为最终版，也可以删除不需要的版本。

## 数据库设计

项目使用 PostgreSQL 存储核心数据。

| 表名 | 说明 |
| --- | --- |
| travel_sessions | 存储一次旅游规划任务 |
| travel_plans | 存储不同版本的旅游方案 |
| ai_reviews | 存储 AI Reviewer 审核结果 |
| human_reviews | 存储人工审核结果 |
| agent_runs | 存储每次 Agent 执行记录 |
| execution_steps | 存储 Agent 内部 Tool / LLM 调用步骤 |

### travel_sessions

用于记录用户的一次完整旅游需求，包括出发地、目的地、日期、人数、预算、偏好和限制条件。

### travel_plans

用于保存不同版本的旅游方案。每次用户打回后重新生成的方案都会作为一个新的版本保存。

### ai_reviews

用于保存 AI Reviewer 对每个方案版本的审核结果。

### human_reviews

用于保存用户对方案的人工审核意见。

### agent_runs

用于记录每个 Agent 的运行情况，例如 Agent 名称、触发原因、开始时间、结束时间、执行耗时、是否成功、重试次数和错误信息。

### execution_steps

用于记录 Agent 内部的工具调用和 LLM 调用情况，例如 Tavily 查询、高德地图查询、LLM 生成耗时等。

## 运行方式

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd py_project05_旅游小助手
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 并创建 `.env` 文件：

```bash
cp .env.example .env
```

在 `.env` 中填写：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
AMAP_KEY=your_amap_key

DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=travel_agent
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. 初始化数据库

```bash
python database/_init_db.py
```

### 5. 启动主应用

```bash
streamlit run app.py
```

### 6. 启动运行监控 Dashboard

```bash
streamlit run harness_dashboard.py
```

## 环境变量说明

| 变量名 | 说明 |
| --- | --- |
| DEEPSEEK_API_KEY | DeepSeek 模型 API Key |
| TAVILY_API_KEY | Tavily Search API Key |
| AMAP_KEY | 高德地图 API Key |
| DB_HOST | PostgreSQL 数据库地址 |
| DB_PORT | PostgreSQL 数据库端口 |
| DB_NAME | PostgreSQL 数据库名称 |
| DB_USER | PostgreSQL 用户名 |
| DB_PASSWORD | PostgreSQL 密码 |

## 项目亮点

### 多智能体任务拆解

项目没有直接使用单个大模型生成完整旅游方案，而是将复杂任务拆分为景点、酒店、交通、规划和审核多个 Agent，使每个 Agent 的职责更加清晰。

### LangGraph 工作流编排

使用 LangGraph 构建可控的多节点执行流程，支持条件路由、自动打回和人工中断恢复。

### 外部工具增强

通过 Tavily Search 和高德地图 API 获取真实外部信息，减少大模型凭空生成景点、酒店和交通信息的问题。

### AI Reviewer 审核机制

引入 AI Reviewer 对 Planner 生成的方案进行自动审核，并在发现明显问题时将审核意见反馈给 Planner 重新生成。

### Human-in-the-loop

用户可以对方案进行人工审核。如果不满意，可以输入修改意见，系统会根据反馈重新生成方案。

### 方案版本管理

每次生成的新方案都会保存为一个独立版本，用户可以查看历史版本、设为最终版或删除版本。

### 可观测性设计

系统记录 Agent 执行过程，包括执行耗时、是否成功、失败原因、重试次数、Tool 调用和 LLM 调用情况，并通过 Dashboard 展示运行状态。

## 当前不足

- 外部 API 查询结果依赖网络稳定性
- Planner 最终输出仍以文本为主，结构化程度可以继续提升
- 缺少自动化测试
- 暂未支持用户登录和多用户隔离
- 暂未支持 Docker 一键部署
- 局部重新生成能力还可以继续增强，例如只重新推荐酒店或只重新规划交通

## 后续优化方向

- 增加结构化 Planner 输出
- 增加自动化测试
- 增加 API 查询缓存
- 优化外部工具失败时的降级策略
- 支持多用户登录
- 支持 Docker 一键部署
- 增加方案导出功能，例如导出为 Markdown、PDF 或 Word
- 增加更详细的预算拆分
- 增加地图可视化展示

## 适用场景

本项目适合作为：

- AI 应用开发学习项目
- LangGraph 多智能体实践项目
- Python + Streamlit 项目
- 大模型工具调用项目
- 研究生课程项目
- AI 应用方向简历项目

## License

This project is for learning and demonstration purposes.
