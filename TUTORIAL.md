# 多智能体小说创作系统 —— 完整教学教程

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈与架构总览](#2-技术栈与架构总览)
3. [环境搭建](#3-环境搭建)
4. [核心技术详解](#4-核心技术详解)
   - [4.1 统一模型接口：init_chat_model](#41-统一模型接口init_chat_model)
   - [4.2 统一提示词管理：ChatPromptTemplate](#42-统一提示词管理chatprompttemplate)
   - [4.3 DeepAgents 框架](#43-deepagents-框架)
   - [4.4 工具工厂模式](#44-工具工厂模式)
   - [4.5 三层记忆系统](#45-三层记忆系统)
5. [代码逐文件解析](#5-代码逐文件解析)
6. [核心设计模式](#6-核心设计模式)
7. [自定义和扩展指南](#7-自定义和扩展指南)
8. [常见问题与调试](#8-常见问题与调试)

---

## 1. 项目概述

### 1.1 项目简介

这是一个基于 **LangChain + LangGraph + DeepAgents** 的多智能体 AI 小说创作系统。它模拟了一个完整的出版团队：

```
总编(Supervisor)
  ├── 大纲设计师(Architect)   — 人物设定、世界观、大纲
  ├── 章节撰写者(Writer)      — 逐章撰写小说内容
  ├── 出版社编辑(Editor)      — 质量审核、评分反馈
  └── 资深读者(Reader)        — 设定一致性检查、逻辑验证
```

### 1.2 核心能力

- **智能任务委派**：Supervisor 自动分析用户需求，将任务分派给最合适的子智能体
- **持久化记忆**：SQLite 存储人物/大纲/章节/操作日志；ChromaDB 提供向量语义检索
- **版本管理**：支持章节草稿多版本，编辑评分回溯
- **交互式 CLI**：自然语言驱动的交互界面

### 1.3 设计哲学

整个系统严格遵循以下三条原则：

1. ** Supervisor 编排模式**：一个主智能体协调多个专业子智能体
2. **工具即接口**：所有记忆操作通过 LangChain Tool 暴露，智能体通过工具调用访问数据
3. **配置与环境分离**：模型/温度等通过 `.env` 配置，提示词通过 YAML 管理

---

## 2. 技术栈与架构总览

### 2.1 技术栈一览

| 层级 | 技术 | 用途 |
|------|------|------|
| **模型层** | `langchain.init_chat_model` | 统一模型初始化接口 |
| **智能体框架** | `deepagents.create_deep_agent` | 内置中间件的 Agent 编译图 |
| **图编排** | `langgraph` | 底层状态图编译引擎 |
| **工具系统** | `langchain_core.tools` | 工具定义与注册 |
| **提示词** | `langchain_core.prompts.ChatPromptTemplate` | 统一提示词模板 |
| **长期记忆** | SQLAlchemy + SQLite | 结构化数据持久化 |
| **短期记忆** | SQLAlchemy + SQLite | 会话级上下文 |
| **向量搜索** | ChromaDB | 语义相似内容检索 |
| **配置** | Pydantic Settings + .env | 环境变量管理 |
| **日志** | Loguru | 结构化日志 |
| **CLI** | 标准 Python stdio | 交互式终端 |

### 2.2 项目目录结构

```
wangwen_creat/
├── app/
│   ├── agent.py                # ★ 核心：Deep Agent 编排
│   ├── main.py                 # ★ 入口：交互式 CLI
│   ├── core/
│   │   ├── config.py           # 配置管理（Pydantic Settings）
│   │   ├── model_client.py     # ★ 模型注册表（多供应商/多型号）
│   │   ├── exceptions.py       # 自定义异常
│   │   └── logging.py          # Loguru 日志配置
│   ├── memory/
│   │   ├── long_term.py        # 长期记忆（人物/大纲/设定）
│   │   ├── short_term.py       # 短期记忆（草稿/子情节）
│   │   ├── vector_store.py     # ChromaDB 向量检索
│   │   └── store.py            # StoreManager 单例
│   ├── models/
│   │   ├── novel.py            # SQLAlchemy ORM 模型
│   │   └── memory.py           # Pydantic 数据传输模型
│   ├── tools/
│   │   └── factory.py          # ★ 工具工厂（闭包模式）
│   ├── prompts/
│   │   └── __init__.py         # ★ 提示词加载与管理
│   ├── mcp/
│   │   ├── server.py           # MCP 服务器（工具暴露）
│   │   ├── tools.py            # 工具注册表单例
│   │   └── adapters.py         # 兼容层
│   └── utils/
│       ├── text_processing.py
│       └── file_io.py
├── conf/
│   ├── prompts.yaml            # ★ 所有智能体提示词配置
│   └── app_config.yaml         # ★ 模型池 + 智能体模型映射 + 应用配置
├── data/                       # 本地数据（SQLite DB、ChromaDB）
├── tests/                      # 测试文件
├── .env.example                # 环境变量模板
├── requirements.txt
└── pyproject.toml
```

### 2.3 数据流架构

```
用户输入
    │
    ▼
┌─────────────────────────────────────────┐
│            NovelCreationCLI              │
│  (交互循环，包装为 HumanMessage)          │
└──────────────┬──────────────────────────┘
               │ graph.invoke({messages})
               ▼
┌─────────────────────────────────────────┐
│          Deep Agent (编译后的图)           │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │       Supervisor (主智能体)        │   │
│  │  - 分析用户意图                    │   │
│  │  - 调用 task() 委派子智能体        │   │
│  │  - 使用记忆工具检索上下文          │   │
│  └──────┬───────────────────────────┘   │
│         │ task() 工具调用                 │
│         ▼                                │
│  ┌─────────────────────────────────┐    │
│  │   SubAgentMiddleware                │    │
│  │  ┌─────────┐ ┌─────────┐          │    │
│  │  │Architect│ │ Writer  │  ...     │    │
│  │  │ t=1.0   │ │ t=0.8   │          │    │
│  │  └─────────┘ └─────────┘          │    │
│  │  ┌─────────┐ ┌─────────┐          │    │
│  │  │ Editor  │ │ Reader  │          │    │
│  │  │ t=0.2   │ │ t=0.5   │          │    │
│  │  └─────────┘ └─────────┘          │    │
│  └─────────────────────────────────┘    │
└──────────────┬──────────────────────────┘
               │ AI 响应
               ▼
        显示给用户
```

---

## 3. 环境搭建

### 3.1 前置条件

- Python >= 3.10
- 一个 DeepSeek API 密钥（[获取地址](https://platform.deepseek.com)）

### 3.2 安装步骤

```bash
# 1. 克隆或进入项目目录
cd wangwen_creat

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
```

### 3.3 环境变量详解

`.env` 文件中每个变量的含义：

```bash
# ── 核心模型配置 ──
DEEPSEEK_API_KEY=sk-your-key-here    # 【必填】DeepSeek API 密钥
DEEPSEEK_MODEL=deepseek-v4-pro       # 模型名称
DEEPSEEK_BASE_URL=https://api.deepseek.com  # API 端点（使用默认即可）

# ── 智能体温度（控制创意性 0~2，越高越有创意） ──
ARCHITECT_TEMPERATURE=1.0   # 大纲设计师：高创意
WRITER_TEMPERATURE=0.8      # 章节撰写者：中高创意
EDITOR_TEMPERATURE=0.2      # 编辑审核：低创意/高一致性
READER_TEMPERATURE=0.5      # 读者检查：平衡
SUPERVISOR_TEMPERATURE=0.3  # 总编决策：低创意/高稳定

# ── 存储路径 ──
SQLITE_DB_PATH=data/novels.db          # SQLite 长期记忆
CHROMA_DB_PATH=data/vector_db/         # ChromaDB 向量存储

# ── 日志级别 ──
LOG_LEVEL=INFO
```

### 3.4 启动系统

```bash
python -m app.main
```

启动后会看到横幅，输入 `帮助` 查看命令，输入自然语言描述创作需求即可。

---

## 4. 核心技术详解

### 4.1 模型池：init_chat_model + 多供应商/多型号切换

#### 问题

旧版 LangChain 中，使用不同的模型提供商需要导入不同的类：

```python
# 旧方式 —— 每个供应商一个类
from langchain_openai import ChatOpenAI          # OpenAI
from langchain_anthropic import ChatAnthropic      # Anthropic
from langchain_deepseek import ChatDeepSeek        # DeepSeek

model = ChatOpenAI(model="gpt-4", api_key="...", base_url="...")
```

这种方式的问题是：
- 依赖具体供应商的实现类
- 切换模型需要改代码
- 不同类的初始化参数不统一
- **无法让不同智能体使用不同供应商/不同型号的模型**

#### 解决方案：模型注册表（ModelRegistry）

系统通过 `app/core/model_client.py` 中的 `ModelRegistry` 实现了「模型池 + 智能体映射」的配置驱动架构：

```
conf/app_config.yaml
├── models:                          # 模型池（可定义任意多个槽位）
│   ├── deepseek_pro:    {provider: deepseek, model: deepseek-v4-pro}
│   ├── deepseek_flash:  {provider: deepseek, model: deepseek-v4-flash}
│   ├── qwen_plus:       {provider: openai,  model: qwen-plus}       # 另一个供应商
│   └── ollama_qwen:     {provider: openai,  model: qwen2.5:72b}     # 本地
│
└── agents:                          # 智能体 → 模型槽位映射
    ├── supervisor: {model: deepseek_pro}     # 总编用 pro
    ├── architect:  {model: deepseek_pro}     # 设计师用 pro
    ├── writer:     {model: deepseek_pro}     # 撰写者用 pro
    ├── editor:     {model: deepseek_flash}   # 编辑用 flash（省成本）
    └── reader:     {model: deepseek_flash}   # 读者用 flash
```

**核心能力**：
1. **同一供应商不同型号**：pro 和 flash 可以并存，按需分配给不同智能体
2. **跨供应商**：一个项目里 supervisor 用 DeepSeek、editor 用 Qwen、reader 用 Claude
3. **配置驱动**：切换模型只需改 YAML，零代码改动
4. **密钥与代码分离**：每个槽位通过 `api_key_env` 指定密钥来源环境变量

#### 模型池配置（conf/app_config.yaml）

```yaml
models:
  deepseek_pro:
    provider: deepseek                   # 供应商
    model: deepseek-v4-pro               # 型号
    base_url: https://api.deepseek.com   # API 端点
    api_key_env: DEEPSEEK_API_KEY        # 密钥来源环境变量

  deepseek_flash:
    provider: deepseek
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY

  # 跨供应商示例
  qwen_plus:
    provider: openai                     # OpenAI 兼容 API
    model: qwen-plus
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key_env: DASHSCOPE_API_KEY
```

#### 智能体映射配置（conf/app_config.yaml）

```yaml
agents:
  supervisor:
    model: deepseek_pro      # 引用上面的模型槽位名
  editor:
    model: deepseek_flash    # 不同的智能体用不同的型号
```

#### 底层实现：init_chat_model（分离式参数）

参考 `model_client/model.py` 的写法，本项目使用 `init_chat_model` 的**分离式参数**形式（而非前缀式）：

```python
# app/core/model_client.py 中的 _build_model 方法
model = init_chat_model(
    model=slot.model,              # "deepseek-v4-pro"
    model_provider=slot.provider,  # "deepseek"
    api_key=slot.api_key,          # 从环境变量读取
    base_url=slot.base_url or None,
    temperature=temperature,       # 每个智能体不同
    max_tokens=self._settings.max_tokens,
)
```

**工作原理**：`model_provider` 参数告诉 `init_chat_model` 使用哪个底层实现类（`deepseek` → `ChatDeepSeek`，`openai` → `ChatOpenAI`），`base_url` 支持 OpenAI 兼容端点（Ollama、OpenRouter、阿里百炼等）。

#### 每个智能体的模型绑定（实测结果）

| 智能体 | 模型槽位 | 实际型号 | Temperature |
|--------|---------|---------|------------|
| Supervisor | deepseek_pro | deepseek-v4-pro | 0.3 |
| Architect | deepseek_pro | deepseek-v4-pro | 1.0 |
| Writer | deepseek_pro | deepseek-v4-pro | 0.8 |
| Editor | deepseek_flash | deepseek-v4-flash | 0.2 |
| Reader | deepseek_flash | deepseek-v4-flash | 0.5 |

#### Temperature 的含义

Temperature 控制模型输出的**随机性/创意性**：

```
t=0.0  → 完全确定性，每次都输出相同内容（适合代码生成）
t=0.2  → 低创意，输出稳定一致（适合 Editor 审核评分）
t=0.5  → 平衡（适合 Reader 检查评价）
t=0.8  → 中高创意（适合 Writer 创作章节）
t=1.0  → 高创意，输出多样（适合 Architect 天马行空设计）
t=2.0  → 最高创意，可能产生随机输出
```

---

### 4.2 统一提示词管理：ChatPromptTemplate

#### 提示词存储

所有智能体的系统提示词集中存储在 `conf/prompts.yaml` 中：

```yaml
architect:
  system_prompt: |
    你是一位资深的小说大纲设计师和世界观构建师。你的职责包括：
    1. **人物设定**：设计主要人物...
    2. **世界观构建**：构建小说的世界观设定...
    3. **大纲制定**：制定完整的小说大纲...
```

#### 提示词加载 (`app/prompts/__init__.py`)

```python
# 1. 从 YAML 加载纯文本（供 deepagents SubAgent 直接使用）
def load_system_prompt(agent_name: str) -> str:
    prompts_data = _load_prompts_yaml()
    return prompts_data[agent_name]["system_prompt"].strip()

# 2. 创建 ChatPromptTemplate（供需要动态变量的场景使用）
def create_chat_template(agent_name: str, user_template: str = "{input}"):
    system_prompt = load_system_prompt(agent_name)
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),      # 系统消息（角色设定）
        ("user", user_template),        # 用户消息（带变量占位符）
    ])

# 3. 预加载为模块级常量（便于直接引用）
SUPERVISOR_PROMPT = load_system_prompt("supervisor")
ARCHITECT_PROMPT = load_system_prompt("architect")
```

#### ChatPromptTemplate 的两种使用场景

**场景 A：纯文本提示词（本项目主要场景）**

`create_deep_agent()` 的 `system_prompt` 参数接受 `str` 类型。直接传入预加载的文本：

```python
agent = create_deep_agent(
    system_prompt=SUPERVISOR_PROMPT,   # str 类型，直接传入
    ...
)
```

**场景 B：动态模板（需要注入变量时）**

```python
# 创建一个带变量的模板
tmpl = create_chat_template("architect", "请为类型为 {genre} 的小说设计人物")
# tmpl.invoke({"genre": "玄幻"})
# → [SystemMessage("你是..."), HumanMessage("请为类型为 玄幻 的小说设计人物")]
```

---

### 4.3 DeepAgents 框架

这是整个系统最核心的部分。理解它需要先理解三个概念。

#### 概念 1：中间件栈

`create_deep_agent()` 内部构建了一个**中间件栈**，每个中间件在模型调用前后执行特定逻辑：

```
用户消息
  │
  ▼
[SkillsMiddleware]          ← 可选：注入自定义技能
[FilesystemMiddleware]      ← 内置：文件读写工具
[SubAgentMiddleware]        ← ★ 关键：task() 工具，管理子智能体
[SummarizationMiddleware]   ← 内置：自动摘要长对话
[PatchToolCallsMiddleware]  ← 内置：修复格式错误的工具调用
[MemoryMiddleware]          ← 可选：长期记忆中间件
[HumanInTheLoopMiddleware]  ← 可选：人工审批中断
  │
  ▼
模型调用
  │
  ▼
[AnthropicPromptCachingMiddleware] ← 内置：缓存优化
  │
  ▼
返回结果
```

#### 概念 2：子智能体（SubAgent）

SubAgent 是一个 TypedDict，定义了子智能体的完整规格：

```python
SubAgent = {
    "name": "architect",              # 名称（task 工具用此名称调用）
    "description": "设计人物和世界观",  # 描述（Supervisor 据此决定何时调用）
    "system_prompt": "你是...",        # 系统提示词
    "model": ChatDeepSeek(...),        # 模型实例（可有自己的 temperature）
    "tools": [tool1, tool2, ...],      # 该子智能体可用的工具
}
```

**工作流程**：

1. Supervisor 收到用户消息 "创建一个玄幻小说"
2. Supervisor 分析后决定需要设计工作，调用 `task("architect", "请设计人物和世界观")`
3. `SubAgentMiddleware` 拦截 `task()` 调用，启动 architect 子智能体
4. Architect 使用自己的系统提示词、模型和工具，执行设计任务
5. Architect 的结果返回给 Supervisor
6. Supervisor 汇总后展示给用户

#### 概念 3：compile 后的图

```python
agent = create_deep_agent(...)  # 返回 CompiledStateGraph

# 图中有固定节点：
# __start__ → model → tools → PatchToolCallsMiddleware.before_agent → __end__
#
# 当模型决定调用工具时： model → tools → model → ...
# 当模型输出最终回复时： model → __end__
```

#### 本项目的 SubAgent 配置 (`app/agent.py` 第 67-127 行)

每个子智能体有不同的工具分配：

| SubAgent | 工具 | 设计原因 |
|----------|------|---------|
| architect | `search_long_term_memory`, `save_to_long_term`, `get_novel_outline`, `get_character_profile`, `get_world_building`, `list_novels` | 设计阶段需要读写长期记忆 |
| writer | `get_novel_outline`, `get_character_profile`, `get_world_building`, `get_short_term_context`, `search_similar_content`, `save_chapter` | 写作时需要读取设定和上下文，保存草稿 |
| editor | `get_novel_outline`, `get_character_profile`, `get_short_term_context`, `save_chapter` | 审核时读取内容，保存评分反馈 |
| reader | `get_character_profile`, `get_world_building`, `search_similar_content` | 检查时只需读取验证 |

**最小权限原则**：每个子智能体只能访问完成任务所需的工具，不能越权。

---

### 4.4 工具工厂模式

#### 问题：全局变量反模式

旧的方式使用模块级全局变量：

```python
# ❌ 反模式：模块级全局变量
_ltm: Optional[LongTermMemory] = None

def init_adapters(ltm, stm, vs):
    global _ltm        # 全局注入
    _ltm = ltm

@tool
def search_long_term_memory(query, novel_id):
    results = _ltm.search(...)  # 依赖全局状态
```

问题：
- **测试困难**：必须手动设置全局变量才能测试工具
- **线程不安全**：多线程环境下全局状态可能被覆盖
- **隐式依赖**：调用者不知道工具依赖了哪些后端

#### 解决方案：闭包工厂模式

```python
# ✅ 闭包工厂模式 (app/tools/factory.py)
class NovelMemoryTools:
    def __init__(self, ltm: LongTermMemory, stm: ShortTermMemory, vs: VectorStore):
        self._ltm = ltm    # 通过构造函数显式注入
        self._stm = stm
        self._vs = vs

    def _search_long_term_memory(self, query: str, novel_id: str) -> str:
        results = self._ltm.search_semantic(...)  # 使用实例变量（闭包捕获）
        return format_results(results)

    def get_tools(self) -> list:
        # 将实例方法包装为工具，并重命名为不带下划线前缀的名字
        t1 = tool(self._search_long_term_memory)
        t1.name = "search_long_term_memory"
        ...
        return [t1, t2, ..., t10]
```

**设计模式**：
- **构造函数注入**：依赖通过 `__init__` 显式传入
- **闭包捕获**：方法通过 `self` 访问后端，不需要全局变量
- **工厂方法**：`get_tools()` 是唯一的外部接口，返回包装好的工具列表

**优势**：
- 测试时可以直接传入 mock 对象
- 每个 NovelMemoryTools 实例独立，线程安全
- 依赖关系显式、可追踪

---

### 4.5 三层记忆系统

系统实现了三层记忆，对应人类创作过程中的不同时间尺度的信息：

#### 第一层：长期记忆（SQLite）

存储**跨创作会话**的持久数据：

| 表名 | 内容 | 关键字段 |
|------|------|---------|
| `novels` | 小说项目 | novel_id, title, genre, synopsis, status |
| `characters` | 人物档案 | char_id, novel_id, name, role_type, personality, background |
| `world_settings` | 世界观设定 | setting_id, novel_id, category, name, description |
| `outlines` | 大纲条目 | outline_id, novel_id, chapter_seq, title, summary, key_events |
| `main_plots` | 主线情节 | plot_id, novel_id, arc_name, description, start_chapter, end_chapter |

**数据模型（`app/models/novel.py`）**：

```python
class Novel(Base):
    __tablename__ = "novels"
    id: Mapped[int] = mapped_column(primary_key=True)
    novel_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(256), comment="小说标题")
    genre: Mapped[str] = mapped_column(String(128), comment="小说类型/流派")
    synopsis: Mapped[str] = mapped_column(Text, comment="小说简介")
    status: Mapped[str] = mapped_column(String(32), default="planning")
    target_chapters: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = ...
    updated_at: Mapped[datetime] = ...
```

#### 第二层：短期记忆（SQLite）

存储**当前创作会话**的临时数据：

| 表名 | 内容 | 关键字段 |
|------|------|---------|
| `sub_plots` | 子情节/支线 | sub_plot_id, novel_id, content, status |
| `chapter_drafts` | 章节草稿（多版本） | draft_id, novel_id, chapter_seq, version, content, quality_score |
| `agent_logs` | 操作日志 | log_id, agent_name, action, output_summary |

草稿版本管理是关键设计：

```python
def save_draft(self, novel_id, chapter_seq, content, feedback="", quality_score=None):
    # 查找当前章节最新版本号
    latest = self._get_latest_version(novel_id, chapter_seq)
    new_version = (latest + 1) if latest else 1
    # 创建新版本（不覆盖旧版本）
    draft = ChapterDraft(
        draft_id=f"draft_{uuid4().hex[:12]}",
        novel_id=novel_id,
        chapter_seq=chapter_seq,
        version=new_version,     # 自动递增版本号
        content=content,
        feedback=feedback,
        quality_score=quality_score,
    )
```

这意味着每次重写都会保留历史版本，可以追溯修改历程。

#### 第三层：向量检索（ChromaDB）

为**语义搜索**提供能力：

| Collection | 内容 | 用途 |
|-----------|------|------|
| `novel_characters` | 人物描述嵌入 | "哪个角色是勇敢正直的？" |
| `novel_settings` | 世界观嵌入 | "有哪些关于魔法体系的设定？" |
| `novel_plots` | 情节片段嵌入 | "有哪些暗线伏笔？" |
| `chapter_content` | 章节内容嵌入 | "找到描述战斗场景的段落" |

**当前状态**：向量搜索通过 ChromaDB 的 built-in embedding 或外部传入的 embedding 向量工作。`search_semantic` 方法还有一个基于关键词匹配的降级方案。

#### 记忆系统的数据流

```
智能体调用工具
    │
    ├── get_novel_outline("novel_xxx")
    │   └── LongTermMemory.get_outlines()  → SQLite outlines 表
    │
    ├── save_to_long_term("novel_xxx", "character", "张三", ...)
    │   └── LongTermMemory.save_character()  → SQLite characters 表
    │
    ├── get_short_term_context("novel_xxx", chapter=5)
    │   └── ShortTermMemory.get_active_sub_plots()  → SQLite sub_plots 表
    │   └── ShortTermMemory.get_latest_draft(chapter=4)  → SQLite chapter_drafts 表
    │   └── ShortTermMemory.get_recent_logs()  → SQLite agent_logs 表
    │
    └── search_similar_content("魔法体系")
        └── VectorStore.search()  → ChromaDB
```

---

## 5. 代码逐文件解析

### 5.1 `app/agent.py` — 核心编排文件

这是整个系统的**心脏**。让我逐段解释：

```python
# 导入部分 (第 1-29 行)
from deepagents import create_deep_agent, SubAgent   # ★ DeepAgents 框架
from langgraph.checkpoint.sqlite import SqliteSaver   # ★ Checkpoint 持久化
from langgraph.graph.state import CompiledStateGraph  # ★ 编译后的图类型

from app.core.config import get_settings             # 配置单例
from app.core.model_client import get_model_registry, ModelRegistry  # ★ 模型注册表
from app.memory import LongTermMemory, ShortTermMemory, VectorStore  # 记忆后端
from app.prompts import (SUPERVISOR_PROMPT, ARCHITECT_PROMPT, ...)  # 提示词
from app.tools import NovelMemoryTools                 # 工具工厂
```

#### 3 个关键函数

**`_build_sub_agents(memory_tools, client)`** (第 32-129 行)

构建 4 个专业化子智能体。核心逻辑：

```python
# 步骤 1：获取工具全集
all_tools = memory_tools.get_tools()

# 步骤 2：按名称筛选，为每个子智能体分配工具子集
architect_tools = [t for t in all_tools if t.name in {
    "search_long_term_memory", "save_to_long_term",
    "get_novel_outline", "get_character_profile",
    "get_world_building", "list_novels",
}]

# 步骤 3：从模型注册表获取每个子智能体的模型（可能是不同供应商/型号）
sub_agents = [
    {
        "name": "architect",
        "description": "资深小说大纲设计师...",
        "system_prompt": ARCHITECT_PROMPT,       # 从 YAML 加载的提示词
        "model": client.get_architect_model(),  # ★ 由 ModelRegistry 按映射分配
        "tools": architect_tools,
    },
    # ... writer, editor, reader 同理
]
```

**`create_novel_agent(...)`** (第 132-208 行)

主入口函数，按特定顺序初始化系统：

```python
def create_novel_agent(ltm=None, stm=None, vs=None, checkpoint_db_path=None):
    client = get_model_registry()   # ★ 获取模型注册表单例

    # 步骤 1：初始化记忆层
    if ltm is None:
        ltm = LongTermMemory()     # SQLite 长期记忆
    if stm is None:
        stm = ShortTermMemory()    # SQLite 短期记忆
    if vs is None:
        vs = VectorStore()         # ChromaDB 向量存储

    # 步骤 2：创建工具集（闭包工厂）
    memory_tools = NovelMemoryTools(ltm, stm, vs)
    tools = memory_tools.get_tools()   # 返回 10 个 LangChain Tool

    # 步骤 3：从注册表获取 Supervisor 模型（绑定到 deepseek_pro 槽位）
    supervisor_model = client.get_supervisor_model()

    # 步骤 4：构建子智能体（各自从注册表获取模型）
    sub_agents = _build_sub_agents(memory_tools, client)

    # 步骤 5：可选 Checkpointer（保存对话历史）
    checkpointer = None
    if checkpoint_db_path:
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    # 步骤 6：创建 Deep Agent（核心调用）
    agent = create_deep_agent(
        model=supervisor_model,        # 主模型
        tools=tools,                   # 主智能体的工具（全部 10 个）
        system_prompt=SUPERVISOR_PROMPT, # Supervisor 系统提示词
        subagents=sub_agents,          # 4 个子智能体定义
        checkpointer=checkpointer,     # 对话持久化
        name="novel_supervisor",       # 图名称
    )

    return agent    # CompiledStateGraph
```

### 5.2 `app/main.py` — 交互式 CLI

关键设计在 `NovelCreationCLI` 类中：

```python
class NovelCreationCLI:
    def __init__(self):
        self.session_id = f"session_{uuid4().hex[:12]}"   # 会话标识
        self.graph = None    # 编译后的图

    def init_workflow(self):
        """初始化：创建 Deep Agent（调用 agent.py）"""
        self.graph = create_novel_agent(
            checkpoint_db_path="data/checkpoints.db"
        )

    def run(self):
        """主循环"""
        config = {"configurable": {"thread_id": self.session_id}}
        # thread_id 是 LangGraph checkpoint 的隔离键
        # 不同会话的对话历史通过 thread_id 区分

        while True:
            user_input = input("📝 > ")
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config,   # ← 传入同一个 config 维持对话连续性
            )
            self._handle_result(result)

    def _handle_result(self, result):
        """从消息列表中提取最后一条 AI 响应"""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(msg.content[:3000])
                break
```

**要点**：
- `config["configurable"]["thread_id"]` 是 LangGraph 的会话隔离机制
- 每次 `invoke` 都传入同一个 `config`，graph 会自动恢复之前的对话历史
- `"messages"` 键包含完整的对话历史（用户消息 + AI 响应 + 工具调用记录）

### 5.3 `app/tools/factory.py` — 工具工厂

```python
class NovelMemoryTools:
    def __init__(self, ltm, stm, vs):
        self._ltm = ltm    # 记忆后端通过构造函数注入
        self._stm = stm
        self._vs = vs

    def _search_long_term_memory(self, query, novel_id, category="all", k=5):
        """实例方法，通过 self._ltm 访问长期记忆"""
        results = self._ltm.search_semantic(novel_id, query, category, k)
        if not results:
            return f"未在长期记忆中找到与 '{query}' 相关的内容。"
        lines = [f"[{i}] [{entry.category}] {entry.content[:500]}"
                 for i, entry in enumerate(results, 1)]
        return "\n\n".join(lines)

    def get_tools(self) -> list:
        """唯一的外部接口：返回包装好的工具列表"""
        t = tool(self._search_long_term_memory)
        t.name = "search_long_term_memory"    # 覆盖默认的下划线前缀名称
        # ... 同理创建其他 9 个工具
        return [t1, t2, ..., t10]
```

**`tool()` 装饰器的两个作用**：
1. 从函数的 docstring 自动提取 `description`（LLM 据此决定是否调用）
2. 从函数的参数签名自动生成 JSON Schema（LLM 据此构造参数）

### 5.4 `app/prompts/__init__.py` — 提示词管理

```python
def load_system_prompt(agent_name: str) -> str:
    """从 conf/prompts.yaml 加载指定智能体的系统提示词"""
    prompts_data = _load_prompts_yaml()
    agent_config = prompts_data.get(agent_name, {})
    return agent_config.get("system_prompt", "").strip()

def create_chat_template(agent_name, user_template="{input}"):
    """创建 ChatPromptTemplate 实例"""
    system_prompt = load_system_prompt(agent_name)
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_template),
    ])

# 预加载为模块常量
SUPERVISOR_PROMPT = load_system_prompt("supervisor")
ARCHITECT_PROMPT = load_system_prompt("architect")
```

### 5.5 `app/core/config.py` 和 `app/core/model_client.py` — 配置与模型管理

**config.py 负责基础配置**（密钥、温度、存储路径）：

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",            # 自动读取 .env 文件
        env_file_encoding="utf-8",
        extra="ignore",             # 忽略未知环境变量
    )

    # 每个字段的 alias 指定对应的环境变量名
    llm_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    llm_model: str = Field(default="deepseek-v4-pro", alias="DEEPSEEK_MODEL")
    max_tokens: int = 200000
    supervisor_temperature: float = Field(default=0.3, alias="SUPERVISOR_TEMPERATURE")
    ...
```

**model_client.py 负责模型池管理**（多供应商、多型号）：

```python
class ModelRegistry:
    def __init__(self):
        self._slots: dict[str, ModelSlot] = {}      # 模型池（来自 YAML models 段）
        self._bindings: dict[str, str] = {}         # 智能体→槽位映射（来自 YAML agents 段）
        self._models: dict[str, BaseChatModel] = {} # 模型实例缓存
        self._load_model_pool()                     # 加载 models 段
        self._load_agent_bindings()                 # 加载 agents 段

    def get_model(self, agent_name: str) -> BaseChatModel:
        """按智能体名获取绑定的模型实例（带缓存）"""
        slot = self._slots[self._bindings[agent_name]]
        return self._build_model(slot, self._temperature_for(agent_name))
```

**设计要点**：
- `BaseSettings` 自动从环境变量和 `.env` 文件加载密钥和温度
- `ModelRegistry` 从 `conf/app_config.yaml` 加载模型池和智能体映射
- **密钥在 `.env`，模型结构在 YAML，代码零耦合**——切换供应商/型号只改配置

---

## 6. 核心设计模式

### 6.1 Supervisor 编排模式

```
                   ┌──────────────┐
                   │  Supervisor  │  ← 主智能体，分析意图、分配任务
                   │   t = 0.3    │
                   └──────┬───────┘
                          │ task() 工具调用
            ┌─────────────┼─────────────┬─────────────┐
            ▼             ▼              ▼             ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
     │Architect │ │ Writer   │ │ Editor   │ │ Reader   │
     │ t = 1.0  │ │ t = 0.8  │ │ t = 0.2  │ │ t = 0.5  │
     └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**为什么 Supervisor 用最低温度 (0.3)？**
- 任务分配需要**一致性**——相同的输入应产生相同的路由决策
- 评价工作质量需要**稳定性**——不能让创意波动影响判断

**为什么 Architect 用最高温度 (1.0)？**
- 创意工作需要**多样性**——相同设定每次应有不同的设计方案
- 人物和世界观的创新需要**跳出固有模式**

### 6.2 闭包工厂模式（工具）

```python
def create_memory_tools(ltm, stm, vs):     # ← 工厂函数接收依赖
    @tool
    def search_memory(query, novel_id):    # ← 内部函数通过闭包访问依赖
        return ltm.search(query, novel_id)
    return [search_memory, ...]            # ← 返回包装好的工具
```

**对比传统方式**：

| 传统全局变量 | 闭包工厂 |
|------------|---------|
| `global _ltm` | `self._ltm` |
| 隐式依赖 | 显式依赖 |
| 线程不安全 | 线程安全 |
| 测试需 monkey-patch | 测试传入 mock 即可 |
| 启动时必须初始化全局状态 | 按需创建实例 |

### 6.3 单例模式（配置）

```python
_settings: Optional[Settings] = None   # 模块级私有变量

def get_settings() -> Settings:
    global _settings
    if _settings is None:              # 延迟初始化
        _settings = Settings()         # 只在第一次调用时创建
    return _settings                   # 之后始终返回同一个实例
```

**为什么用单例？**
- 避免重复读取 `.env` 文件
- 整个应用共享同一份配置
- 测试时可以通过 `reload_settings()` 重置

### 6.4 工具最小权限原则

系统中不同智能体分配到不同的工具子集：

```
Supervisor:  ALL 10 tools      ← 总管，需要全局访问权限
Architect:   6 tools (记忆读写)  ← 设计阶段需要读写长期记忆
Writer:      6 tools (记忆读取)  ← 写作时主要读取设定
Editor:      4 tools (审核相关)  ← 审核只需读内容、写反馈
Reader:      3 tools (检查相关)  ← 检查只需验证一致性
```

这防止了 Writer 意外修改人物设定，或 Editor 意外删除大纲。

---

## 7. 自定义和扩展指南

### 7.1 添加新的子智能体

假设要添加一个 `translator` 子智能体来做多语言翻译：

**步骤 1**：在 `conf/prompts.yaml` 添加提示词：

```yaml
translator:
  system_prompt: |
    你是一位专业的文学翻译家。你的职责包括：
    1. 将小说内容翻译为目标语言
    2. 保留原文的文学风格和修辞手法
    3. 适应目标语言的文化习惯
```

**步骤 2**：在 `app/agent.py` 的 `_build_sub_agents` 中添加：

```python
translator_tools = [t for t in all_tools if t.name in {
    "get_novel_outline", "get_character_profile",
    "get_short_term_context", "save_chapter",
}]

sub_agents.append({
    "name": "translator",
    "description": "专业文学翻译家，将小说翻译为其他语言。在需要多语言版本时调用。",
    "system_prompt": TRANSLATOR_PROMPT,  # 先在 prompts/__init__.py 中加载
    "model": init_chat_model(model_id, temperature=0.3, max_tokens=max_tok),
    "tools": translator_tools,
})
```

**步骤 3**：在 `app/prompts/__init__.py` 中添加：

```python
TRANSLATOR_PROMPT = load_system_prompt("translator")
```

### 7.2 添加新的记忆工具

假设要添加 `delete_character` 工具：

**步骤 1**：在 `app/memory/long_term.py` 添加后端方法：

```python
def delete_character(self, novel_id: str, char_id: str) -> bool:
    with self.get_session() as session:
        char = session.query(Character).filter(
            Character.novel_id == novel_id,
            Character.char_id == char_id,
        ).first()
        if char:
            session.delete(char)
            session.commit()
            return True
        return False
```

**步骤 2**：在 `app/tools/factory.py` 添加工具方法：

```python
def _delete_character(self, novel_id: str, char_id: str) -> str:
    """
    删除指定人物。
    参数:
        novel_id: 小说ID
        char_id: 人物ID
    """
    ok = self._ltm.delete_character(novel_id, char_id)
    return "人物已删除" if ok else "未找到指定人物"

# 在 get_tools() 中添加
t11 = tool(self._delete_character)
t11.name = "delete_character"
# ...
```

### 7.3 调整智能体温度

编辑 `.env` 文件：

```bash
# 提高创意性
ARCHITECT_TEMPERATURE=1.5   # → 更天马行空的设计

# 降低随机性
EDITOR_TEMPERATURE=0.0      # → 完全标准化的评分

# 更谨慎的决策
SUPERVISOR_TEMPERATURE=0.1  # → 更稳定的任务分配
```

### 7.4 更换/切换模型（供应商 + 型号）

模型池和智能体映射都在 `conf/app_config.yaml` 中配置，**切换模型零代码改动**。

#### 场景 1：同一供应商切换型号

在 `models` 段增加新槽位，然后把智能体指过去：

```yaml
models:
  deepseek_pro:
    provider: deepseek
    model: deepseek-v4-pro
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY

  deepseek_reasoner:            # 新增一个型号槽位
    provider: deepseek
    model: deepseek-reasoner    # 推理增强型号
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY

agents:
  supervisor:
    model: deepseek_reasoner    # 让总编改用推理型号
  editor:
    model: deepseek_pro         # 其他不变
```

#### 场景 2：跨供应商（部分智能体用不同供应商）

```yaml
models:
  deepseek_pro:
    provider: deepseek
    model: deepseek-v4-pro
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY

  claude_sonnet:                # 另一个供应商
    provider: anthropic
    model: claude-sonnet-4-5-20250901
    api_key_env: ANTHROPIC_API_KEY

agents:
  writer:
    model: claude_sonnet        # 撰写者用 Claude（文学性更强）
  editor:
    model: deepseek_pro         # 编辑仍用 DeepSeek（省成本）
```

#### 场景 3：本地模型（Ollama）

```yaml
models:
  ollama_qwen:
    provider: openai            # Ollama 走 OpenAI 兼容接口
    model: qwen2.5:72b
    base_url: http://localhost:11434/v1
    api_key_env: OLLAMA_API_KEY

agents:
  reader:
    model: ollama_qwen          # 一致性检查用本地模型，零 API 成本
```

#### 密钥管理

每个槽位的 `api_key_env` 指向 `.env` 中的对应变量名，密钥只存在 `.env`，不进 YAML：

```bash
# .env
DEEPSEEK_API_KEY=sk-your-deepseek-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
OLLAMA_API_KEY=ollama
```

---

## 8. 常见问题与调试

### 8.1 模型密钥未设置

**错误信息**：`If using default api base, DEEPSEEK_API_KEY must be set`

**原因**：模型槽位配置的 `api_key_env` 指向的环境变量在 `.env` 中不存在。

**解决**：
```bash
# 1. 检查 conf/app_config.yaml 中槽位的 api_key_env 字段
#    deepseek_pro: api_key_env: DEEPSEEK_API_KEY

# 2. 确认 .env 中该变量存在且名称完全一致
cat .env | grep DEEPSEEK_API_KEY
# 应为 DEEPSEEK_API_KEY=sk-xxx（注意大小写和前缀）

# 3. 如果用其他供应商，确认对应密钥变量也已配置
```

### 8.2 智能体不按预期调用 task()

**原因**：Supervisor 提示词不够明确。

**解决**：在 `conf/prompts.yaml` 中加强 Supervisor 提示词中的工作流指令。例如，明确写出：

```yaml
当你收到用户说"创建小说"时，你必须：
1. 先用 list_novels 检查是否已有项目
2. 用 task(architect, "设计新小说：{用户描述}")
```

### 8.3 章节内容被截断

**原因**：虽然 `max_tokens=200000`，但输出仍可能受提示词长度影响。

**解决**：
- 减小 `get_short_term_context` 中 `prev.content[:500]` 的截断长度
- 让 Writer 分多次 `task()` 调用分段撰写长章节
- 优化提示词中的上下文注入量

### 8.4 如何查看对话历史

```python
# 在 Python 中查看
from app.agent import create_novel_agent
agent = create_novel_agent(checkpoint_db_path="data/checkpoints.db")

# 获取之前会话的状态
config = {"configurable": {"thread_id": "session_abc123"}}
state = agent.get_state(config)
for msg in state.values.get("messages", []):
    print(f"[{type(msg).__name__}] {msg.content[:100]}")
```

### 8.5 如何重置记忆系统

```bash
# 删除 SQLite 数据库
rm data/novels.db data/short_term.db data/checkpoints.db

# 删除 ChromaDB 向量存储
rm -rf data/vector_db/

# 重新启动会自动创建空数据库
python -m app.main
```

### 8.6 调试工具调用

在 `app/core/logging.py` 中设置日志级别为 DEBUG：

```python
logger.add(sys.stderr, level="DEBUG", ...)
```

或在 `.env` 中：
```bash
LOG_LEVEL=DEBUG
```

这样可以看到每次工具调用的输入参数和返回值。

---

## 小结

本教程覆盖了从环境搭建到核心架构再到自定义扩展的完整流程。系统的核心理念：

1. **模型注册表（ModelRegistry）** 支持多供应商、多型号，每个智能体独立指定模型，切换零代码
2. **`init_chat_model`** 统一模型构建入口，`base_url` 支持 OpenAI 兼容端点
3. **`create_deep_agent` + SubAgent** 实现 Supervisor 编排，中间件自动处理工具调用和子智能体调度
4. **闭包工厂模式** 消除全局状态，依赖注入提高可测试性
5. **三层记忆** 分层管理不同时间尺度的创作上下文
6. **YAML 管理提示词和模型映射** 非技术人员也能调整智能体行为

详细的代码实现请参考项目源文件，每个函数都有中文文档字符串说明。
