# AI 多智能体旅游助手

## 项目简介

AI 多智能体旅游助手是一个基于 LangGraph、LangChain、Streamlit 和 MySQL 构建的智能旅游规划系统。

项目将一次完整的旅游规划任务拆分为多个 Agent 协同完成，包括景点推荐、酒店推荐、交通规划、行程生成、AI 审核和人工审核。系统集成 Tavily Search 和高德地图 API 获取外部查询证据，并使用 MySQL 对旅行任务、方案版本、审核记录、长期记忆和 Agent 执行日志进行持久化存储。

相比简单的大模型问答式旅游规划，本项目更强调多智能体协作、工具调用、审核闭环、方案版本管理和运行过程可观测性。

## 本次更新（2026-10-13）

- 默认数据库改为 MySQL 8.4，通过 SQLAlchemy + PyMySQL 连接并使用 `utf8mb4`。
- 酒店节点使用受控 ReAct 工具循环，限制业务工具和模型调用次数，复用本轮重复查询结果。
- 一日游跳过酒店查询，仅查询城际去返程和景点之间的市内交通。
- 公交接口成功但无可用路线时，对该段补查真实步行路线，按距离、耗时和累计限制决定是否采用。
- 原始需求与最终结果分开保存：相同首次需求直接复用人工修改后确认的最终方案，不重新调用 Agent。
- 只在人工同意后保存最终版；方案、审核、长期记忆和任务完成状态在同一个事务中提交，失败可幂等重试。
- 保留历史版本查询和删除；删除版本同时删除关联审核、记忆，但不删除旅行任务记录和执行日志。
- 149 项离线回归测试通过，覆盖节点调度、人工修改、交通兜底、记忆匹配、最终版事务保存和历史删除。

项目说明统一维护在本文件。本地学习文档、旧代码副本、验证截图和运行日志不作为本次 GitHub 更新的内容。

## 核心功能

- 多智能体旅游规划
- 景点推荐
- 酒店推荐
- 城际交通查询
- 市内交通规划
- 一日游跳过住宿，公共交通空方案时按距离和耗时限制逐段补查步行
- 完整条件相同时直接复用已确认的历史方案快照
- 最终行程生成
- AI Reviewer 自动审核
- Human-in-the-loop 人工审核
- 方案打回与重新生成
- 按修改类型局部重新生成，并更新依赖节点
- 历史版本管理
- MySQL 数据持久化
- Agent 执行日志记录
- Harness Dashboard 运行监控

## 技术栈

- Python
- LangGraph
- LangChain
- Streamlit
- MySQL 8.4 + PyMySQL
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
Hotel Agent（仅多日游）
        ↓
Transport Agent
        ↓
Planner Agent
        ↓
AI Reviewer Agent
        ↓
Human Reviewer
        ↓
MySQL 持久化存储
```

各 Agent 职责说明：

进入上述生成流程之前，系统会按完整请求条件查询长期记忆；命中后直接展示已确认的历史方案，不运行 Agent。一日游从景点查询直接进入交通，不查询酒店。

| Agent | 职责 |
| --- | --- |
| Supervisor Agent | 根据节点状态、结果结构和依赖规则调度下一节点，不调用大模型决定路由 |
| Attraction Agent | 根据目的地和用户偏好推荐景点 |
| Hotel Agent | 基于目的地、景点和预算推荐住宿 |
| Transport Agent | 查询城际交通和市内路线，生成交通建议 |
| Planner Agent | 汇总景点、酒店、交通信息，生成完整旅游方案 |
| AI Reviewer Agent | 自动审核方案是否合理、可执行 |
| Human Reviewer | 允许用户人工确认、打回并提出修改意见 |

## 项目结构

```text
ai-travel-assistant/
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
│   ├── memory.py
│   ├── revision.py
│   ├── route.py
│   ├── state.py
│   └── trip_policy.py
│
├── database/
│   ├── _init_db.py
│   ├── _migrate_plan_memory.py
│   ├── harness_stats.py
│   ├── memory_migration.py
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
├── tests/                  # 模拟模型和外部接口，不连接正式数据库的离线测试
├── harness_dashboard.py
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## 核心流程

### 1. 用户输入旅游需求

用户在 Streamlit 页面输入出发地、目的地、出发日期、返程日期、出行人数、总预算、旅游偏好和其他限制条件。系统会自动计算旅游天数，并构造初始 TravelState。

侧栏可设置市内、城际交通方式以及是否允许公交无方案时补查步行。默认单段步行兜底最多 2 公里且 30 分钟，累计最多 4 公里且 60 分钟；这不是公交接驳和景点内活动的总步行限制。出发和返程日期相同即为一日游，住宿状态为 `skipped`。

### 2. 多 Agent 协同规划

系统通过 LangGraph 编排多个 Agent。Attraction Agent 负责景点推荐，Hotel Agent 负责住宿推荐，Transport Agent 负责交通方案，Planner Agent 负责汇总生成完整行程，AI Reviewer Agent 负责自动审核，Human Reviewer 负责人工确认和反馈。

Hotel 节点内部使用 `create_agent` 运行受控 ReAct：模型选择酒店搜索关键词，观察候选后决定是否补查到景点的步行路线，最终输出 `HotelDecision`。目的地、品牌与排除项由程序锁定，选定酒店仍必须属于本轮真实候选。每次节点执行最多 3 次业务工具调用、5 次模型调用；重复请求复用本轮结果。外层 Harness 仍最多额外重试一次，因此两次尝试合计最多 6 次业务工具调用、10 次模型调用，不含 SDK 内部请求重试和用户主动重试。

### 3. AI 审核与自动修改

AI Reviewer 会检查旅游方案是否存在行程过满、景点安排不合理、交通路线不合理、酒店位置不合适、预算可能超支、不符合用户偏好或违反用户限制等问题。

如果 AI 审核不通过，系统会将审核意见返回给 Planner，由 Planner 根据审核意见重新生成方案。达到当前配置的审核计数上限后转入人工审核，不意味着 AI 已通过。人工明确同意可以覆盖 AI 判断，但数据库仍保存实际的 AI 审核结果。

### 4. 人工审核与版本迭代

当方案进入人工审核环节后，用户可以选择同意方案，也可以打回方案并输入修改意见。生成中、AI 未通过和人工修改中的方案只保留在当前工作流，不保存为数据库方案版本。点击“同意方案”后，只保存最终确认的结果；新任务通常只有一个 Version 1。

修改意见区域提供明确的修改类型，不依赖模型猜测需要重跑哪个节点：

| 修改类型 | 重跑范围 | 复用结果 |
| --- | --- | --- |
| 仅调整行程 | Planner、Reviewer | 景点、酒店、交通 |
| 更换景点 | 景点、酒店、交通、Planner、Reviewer | 原始需求及各类已保存的修改要求 |
| 更换酒店 | 酒店、交通、Planner、Reviewer | 景点 |
| 更换交通方式 | 交通、Planner、Reviewer | 景点、酒店 |

景点修改支持从页面选择排除名单；酒店修改支持品牌/名称关键词并排除当前酒店。市内交通支持公共交通、自驾、步行，城际交通支持铁路或航班搜索（同时作用于去程、返程）。

共享状态使用 `pending / dirty / completed` 表示节点结果是否需要重新生成。人工反馈通过 `Command(resume=...)` 更新需求和失效状态，再返回规则式 Supervisor 调度。各类修改要求独立保存，后续 AI 打回不会丢失先前的酒店、景点或交通要求。

一日游不提供酒店修改选项，更换景点时也不会重新激活酒店。公交接口成功但方案列表为空时，交通节点仅对该段补查步行，使用真实距离和耗时检查单段及累计限制，并保存 `requested_mode` 和 `actual_mode`；超时、鉴权失败、响应错误、无步行路线或超限仍然中断生成。

模型输出必须通过基本非空校验；选定酒店必须属于当前真实 POI 候选。酒店搜索正常但候选为空时，模型可在预算内换关键词；最终仍无有效候选、工具异常或交通节点路线不可用时停止生成。酒店可选的步行核查无路线时只返回“无法核查”，不伪造距离，也不替代交通节点。页面支持在同一进程、同一浏览器会话中从失败位置重试；失败的修改也可取消并返回上一个待确认方案。人工确认时，最终方案、AI 审核、人工同意记录、记忆和任务完成状态在同一事务提交。保存失败可单独重试，不重新生成；确认指纹和任务行锁防止重复保存。

这不等于所有事实均已核实：景点名称的别名和真实性仍需更强的 POI 核验，住宿 POI 不提供实时房价/库存，Tavily 城际检索不提供实时余票或订票。当前工作流检查点保存在内存中，应用重启或新浏览器会话不能自动恢复原任务。

### 5. 历史版本管理

侧栏“历史方案”从 MySQL 读取已保存的旅行任务，支持按出发地、目的地或需求搜索，再选择任务查看各版本的方案内容、AI 审核结果、人工审核意见和最终版标记。重启应用或重新打开页面后也可以查看历史方案，无需恢复工作流检查点。用户可以将某个版本设为最终版，也可以删除不需要的版本；删除版本会同时删除关联审核记录和长期记忆。

“删除此版本”不是删除整个旅行任务。对应的 `travel_plans`、审核和记忆会真实删除，但 `travel_sessions` 中的原始需求及 `agent_runs / execution_steps` 日志保留。没有方案的任务会被历史列表过滤，因此页面不显示该任务，不代表任务表的记录也被删除。新任务通常仅一个最终版本，删除该版本后也遵循这条规则。

为避免人工审核附着到错误版本，当前任务等待审核或重生成失败时，该任务历史版本的“设为最终版”和删除操作暂时禁用；完成确认后可操作。查看或管理其他历史任务不会切换当前生成任务，也不会调用模型或重新查询路线。

### 6. 长期记忆与精确复用

生成前规范化首次提交的完整请求条件，生成 SHA-256 指纹并核对条件 JSON。除了日期、地点、预算、人数和偏好，还匹配其他要求、最初的交通方式和步行策略。首次需求保存为独立快照，后续换酒店、排除景点、调整交通和行程不会改变匹配键；相同初始需求复用人工修改后确认的最终结果。只复用已完成任务中最新人工审核通过的最终版本；命中不新建旅行任务、不执行 Agent，页面注明保存时间和历史快照性质。勾选“重新查询，不复用历史方案”可跳过复用。

升级前没有完整匹配元数据的旧方案仍可查看，但不会自动猜测条件并复用。对于具有可验证原始条件的旧记忆，使用 `python -m database._migrate_plan_memory` 预览，再用 `--apply` 升级匹配元数据；不改变方案正文、审核记录和历史版本。最终结果快照保留实际交通方式、排除项和人工修改意见。数据库记忆可跨应用重启读取，但不改变内存工作流检查点的恢复限制；当前没有用户身份隔离。

例如首次输入“广州到香港、两天、1 人、3000 元、著名景点、拍照漂亮”，生成后人工要求更换酒店、排除某个景点。确认后保存修改后的方案，但匹配键仍来自最初需求。再次提交相同初始条件，直接展示修改后的最终结果。最初的交通方式、步行策略或其他要求不同则不自动命中；偏好列表先规范化、去重并排序，其他要求列表保留顺序。这里是精确条件匹配，不是向量检索或语义相似搜索。

记忆命中是历史结果复用，不能据此保证今天的票价、余票、营业时间或路线未变化。当前没有自动过期机制，需要实时信息时应勾选“重新查询，不复用历史方案”。

### 7. Harness 执行管理

主要业务节点经统一 Harness 执行，记录 Agent 运行耗时、成功状态、错误和重试次数；工具及模型步骤另外记录到 `execution_steps`。节点返回值还会经过结构校验，Supervisor 不仅检查结果是否非空，也检查完成状态和可用结构。

Harness 对能识别的超时、连接失败、限流等错误最多额外重试一次，明显程序错误或不可恢复的业务错误停止。当前分类依据异常类型和消息关键词，仍不是完善的错误码体系；重试范围是整个节点，可能重复其中已完成的查询。AI 审核拒绝是正常业务反馈，不属于 Harness 异常重试。

## 数据库设计

项目当前使用 MySQL 存储核心数据；SQLAlchemy JSON 类型用于保存精确匹配条件和结果快照。

| 表名 | 说明 |
| --- | --- |
| travel_sessions | 存储一次旅游规划任务 |
| travel_plans | 存储不同版本的旅游方案 |
| ai_reviews | 存储 AI Reviewer 审核结果 |
| human_reviews | 存储人工审核结果 |
| agent_runs | 存储每次 Agent 执行记录 |
| execution_steps | 存储 Agent 内部 Tool / LLM 调用步骤 |
| travel_plan_memories | 保存方案版本的请求指纹、完整匹配条件和结构化结果快照 |

### travel_sessions

用于记录用户的一次完整旅游需求，包括出发地、目的地、日期、人数、预算、偏好和限制条件。

### travel_plans

用于保存人工确认的最终旅游方案。新任务不保存修改过程中的版本；旧版本仍可查询和管理，不会自动删除。

### ai_reviews

用于保存 AI Reviewer 对每个方案版本的审核结果。

### human_reviews

用于保存最终方案的人工同意记录；本轮打回意见随最终记忆快照保存，不新增未确认的方案版本。

### agent_runs

用于记录每个 Agent 的运行情况，例如 Agent 名称、触发原因、开始时间、结束时间、执行耗时、是否成功、重试次数和错误信息。

### execution_steps

用于记录 Agent 内部的工具调用和 LLM 调用情况，例如 Tavily 查询、高德地图查询、LLM 生成耗时等。

### travel_plan_memories

每个已保存方案最多一条记忆，通过唯一 `plan_id` 关联方案。最终方案、AI 审核、人工审核、记忆元数据和任务完成状态在同一个事务保存。请求键来自最初需求，结果快照来自最终确认状态。方案删除时级联删除记忆，查询只接受符合确认状态的记录。

## 运行方式

### 1. 克隆项目

```bash
git clone https://github.com/li123456-x/ai-travel-assistant.git
cd ai-travel-assistant
```

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 并创建 `.env` 文件。在 Windows PowerShell 中执行：

```powershell
Copy-Item .env.example .env
```

Linux / macOS 使用 `cp .env.example .env`。

在 `.env` 中填写：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
AMAP_KEY=your_amap_key

DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=travel_agent
DB_USER=travel_app
DB_PASSWORD=your_password
```

MySQL 服务必须先启动，目标数据库和应用用户需要提前创建，数据库字符集建议为 `utf8mb4`。数据库运行在虚拟机或独立容器时，`DB_HOST` 填写运行 Python 的主机能访问到的地址，不直接使用容器内部 IP。这里只修改连接后端，不自动把原 PostgreSQL 数据迁移到 MySQL。

真实 API Key、数据库密码和 `.env` 不提交到 GitHub；`.env.example` 只提供占位配置。

### 4. 初始化数据库

```bash
python -m database._init_db
```

### 5. 启动主应用

```bash
python -m streamlit run app.py
```

### 6. 启动运行监控 Dashboard

```bash
python -m streamlit run harness_dashboard.py
```

## 环境变量说明

| 变量名 | 说明 |
| --- | --- |
| DEEPSEEK_API_KEY | DeepSeek 模型 API Key |
| TAVILY_API_KEY | Tavily Search API Key |
| AMAP_KEY | 高德地图 API Key |
| DB_HOST | MySQL 数据库地址；数据库在虚拟机时填写可达的虚拟机 IP |
| DB_PORT | MySQL 数据库端口，默认 3306 |
| DB_NAME | MySQL 数据库名称 |
| DB_USER | MySQL 应用用户名 |
| DB_PASSWORD | MySQL 应用用户密码 |

## 局部重新生成的升级和测试

本次新增独立的 `travel_plan_memories` 表，不修改原有表结构、不删除已有任务或历史版本。执行 `python -m database._init_db` 创建缺少的表。更新代码后完整重启 Streamlit，并新建任务验证；旧内存检查点不包含新状态字段，不应继续恢复使用。

旧记忆的匹配规则可以先预览再升级：

```bash
python -m database._migrate_plan_memory
python -m database._migrate_plan_memory --apply
```

不加 `--apply` 不写数据库；加上后只升级能够验证原始需求的旧记忆元数据，无法确定的任务跳过，不猜测条件，也不删除旧版本。新建空数据库不需要执行此迁移。

离线测试无需 API Key 或可用的 MySQL 服务，使用模拟模型、接口和临时 SQLite，验证真实 LangGraph 中断/恢复、页面交互、交通兜底、原始需求匹配、仅最终版保存、事务回滚和确认重试幂等。2026-10-13 正式项目 149 项离线测试全部通过；真实 MySQL 已完成新增表创建和历史记忆元数据升级，不代表真实接口端到端验收。

| 测试文件 | 重点验证 |
| --- | --- |
| `test_app_regeneration.py` | 页面生成、人工修改、确认保存、历史操作及缓存复用 |
| `test_regeneration_workflow.py` | 真实 LangGraph 中断恢复和局部重生成顺序 |
| `test_selection_nodes.py` | 景点及酒店选择约束、异常和候选校验 |
| `test_hotel_react.py` | 酒店工具循环、调用预算、重复查询与真实候选选择 |
| `test_transport_nodes.py` | 交通输入校验、真实工具参数和失败拦截 |
| `test_transport_fallback.py` | 公交空路线后的步行兜底、距离耗时和累计限制 |
| `test_trip_policy_memory.py` | 一日游策略、原始需求匹配、最终保存事务、幂等重试及记忆迁移 |
| `test_plan_version_save.py` | 底层方案与 AI 审核保存的事务回滚 |
| `test_history_repository.py` | 历史查询和版本关联删除 |

真实页面已验证相同需求命中已确认历史方案，不执行生成流程。SQLite 离线测试不证明 MySQL 行锁在高并发下的效果；还需要独立的 MySQL 并发集成测试和真实接口、模型内容质量评测。

```bash
python -m unittest discover -s tests -v
```

人工验证时，在同一任务中分别选择更换酒店、景点、交通和仅调整行程，观察 Agent 执行顺序、实际查询参数和新结果。确认前历史列表不新增方案，确认后只出现一个最终版本。修改应保持相同的旅行任务 ID，而不是创建新的 `travel_sessions` 记录；再次提交相同初始需求应直接展示该最终结果。

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

新任务仅保存人工确认的最终结果，避免历史列表混入未通过和修改中的草稿。已有历史版本继续支持查看、设为最终版和删除。

### 可观测性设计

系统记录 Agent 执行过程，包括执行耗时、是否成功、失败原因、重试次数、Tool 调用和 LLM 调用情况，并通过 Dashboard 展示运行状态。

## 当前不足

- 外部 API 查询结果依赖网络稳定性
- Planner 最终输出仍以文本为主，结构化程度可以继续提升
- 已有离线回归测试，仍缺少真实接口端到端测试和方案质量评测集
- 暂未支持用户登录和多用户隔离
- 暂未支持 Docker 一键部署
- 局部重生成按整个节点执行；交通重跑仍包含城际和市内查询，可继续细分缓存和依赖

## 后续优化方向

- 增加结构化 Planner 输出
- 扩充真实接口集成测试和方案质量评测
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
