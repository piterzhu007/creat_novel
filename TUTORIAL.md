# 多智能体小说创作系统 —— 完整教学教程

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈与架构总览](#2-技术栈与架构总览)
3. [环境搭建](#3-环境搭建)
4. [核心技术详解](#4-核心技术详解)
   - [4.1 模型池：多供应商/多型号切换](#41-模型池多供应商多型号切换)
   - [4.2 统一提示词管理](#42-统一提示词管理)
   - [4.3 DeepAgents 框架与子智能体](#43-deepagents-框架与子智能体)
   - [4.4 MCP 工具暴露与权限边界](#44-mcp-工具暴露与权限边界)
   - [4.5 三层记忆系统](#45-三层记忆系统)
   - [4.6 多项目隔离与角色名防护](#46-多项目隔离与角色名防护)
   - [4.7 上下文控制与 token 节约](#47-上下文控制与-token-节约)
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
  ├── 出版社编辑(Editor)      — 大纲审核 + 章节审核（含视角/文风把关）
  └── 资深读者(Reader)        — 设定一致性检查、读者体验反馈
```

### 1.2 核心能力

- **智能任务委派**：Supervisor 分析用户需求，通过 `task()` 把任务分派给最合适的子智能体
- **持久化记忆**：SQLite 存人物/大纲/章节/源文档；ChromaDB 提供向量语义检索
- **多项目隔离**：每个小说项目拥有自己的源文档、角色名表、易混淆字表，互不污染
- **定稿锁**：只有 Supervisor 拥有定稿/删除/精准修改/进度/导出的权限，定稿后子智能体无法覆盖
- **角色名硬校验**：正文/设定落库前，代码层强制角色名与权威名表逐字一致，杜绝同音错字
- **精准修改**：`patch_chapter` 对非结构性修改做原文→改后文的定点替换，不必全文重写
- **交互式 CLI**：自然语言驱动的交互界面，支持自动推进连续创作

### 1.3 设计哲学

整个系统严格遵循以下原则：

1. **Supervisor 编排模式**：一个主智能体协调多个专业子智能体
2. **工具即接口**：所有记忆操作通过 MCP 暴露为 LangChain Tool，智能体通过工具调用访问数据
3. **权限收归 Supervisor**：只有 Supervisor 能最终落库、定稿、删除；子智能体只写自己的草稿产出
4. **不截断**：子智能体与 Supervisor 都不做中途上下文压缩，token 节约靠「避免重复读取/重复调用」
5. **配置与代码分离**：模型/温度在 `.env` + `conf/app_config.yaml`，提示词在 `conf/prompts.yaml`

---

## 2. 技术栈与架构总览

### 2.1 技术栈一览

| 层级 | 技术 | 用途 |
|------|------|------|
| **模型层** | `langchain.init_chat_model` | 统一模型初始化接口（deepseek → `ChatDeepSeek`） |
| **智能体框架** | `deepagents.create_deep_agent` | 主 Agent + 子智能体编排 |
| **图编排** | `langgraph` | 底层状态图编译引擎 |
| **工具暴露** | MCP (FastMCP) + `langchain_mcp_adapters` | 记忆工具统一经 stdio 子进程暴露 |
| **工具系统** | `langchain_core.tools` | 工具定义与注册 |
| **提示词** | YAML + `ChatPromptTemplate` | 集中管理提示词 |
| **长期记忆** | SQLAlchemy + SQLite | 结构化数据持久化 |
| **短期记忆** | SQLAlchemy + SQLite | 章节草稿 / 子情节 |
| **向量搜索** | ChromaDB + 显式 embedding | 语义相似内容检索 |
| **配置** | Pydantic Settings + .env | 环境变量管理 |
| **日志** | Loguru | 结构化日志 |
| **运行追踪** | LangChain callback (`RunTracer`) | token 消耗 / 工具调用审计 |
| **技能** | `writing-style` + `novel-anti-ai-style` | 网文风格 + 去 AI 味约束 |

### 2.2 项目目录结构

```
wangwen_creat/
├── app/
│   ├── agent.py                # ★ 核心：Deep Agent 编排 + 权限边界
│   ├── main.py                 # ★ 入口：交互式 CLI（含自动推进/单元重置）
│   ├── workflow.py             # 工作流创建兼容入口
│   ├── core/
│   │   ├── config.py           # 配置管理（Pydantic Settings）
│   │   ├── model_client.py     # ★ 模型注册表（多供应商/多型号/温度/输出上限）
│   │   ├── async_runtime.py    # 跨 loop 的 async 运行辅助
│   │   ├── tracing.py          # ★ 运行追踪器（token/工具调用审计）
│   │   ├── exceptions.py       # 自定义异常
│   │   └── logging.py          # Loguru 日志配置
│   ├── memory/
│   │   ├── long_term.py        # 长期记忆（人物/世界观/大纲/源文档/名表）
│   │   ├── short_term.py       # 短期记忆（草稿/子情节）
│   │   ├── vector_store.py     # ChromaDB 向量检索（显式 embedding）
│   │   └── store.py            # StoreManager 单例
│   ├── models/
│   │   ├── novel.py            # SQLAlchemy ORM 模型（含 SourceDoc）
│   │   └── memory.py           # Pydantic 数据传输模型
│   ├── tools/
│   │   └── factory.py          # ★ 工具工厂（25 个工具 + 角色名硬校验）
│   ├── prompts/
│   │   └── __init__.py         # ★ 提示词加载（内联 skills 到 writer/reader）
│   ├── mcp/
│   │   ├── server.py           # ★ MCP 服务器（FastMCP stdio）
│   │   ├── tools.py            # 工具注册表单例
│   │   └── adapters.py         # 兼容层
│   └── utils/
│       ├── text_processing.py
│       └── file_io.py
├── conf/
│   ├── prompts.yaml            # ★ 所有智能体提示词
│   └── app_config.yaml         # ★ 模型池 + 智能体模型映射
├── skills/
│   ├── writing-style/SKILL.md        # 网文写作风格
│   └── novel-anti-ai-style/SKILL.md  # 去 AI 味（反上帝视角/限知第三人称）
├── data/                       # 本地数据（SQLite、ChromaDB、checkpoint、日志）
├── output/                     # 导出产出 + 运行日志
├── tests/                      # 测试文件
├── .env / .env.example         # 环境变量（密钥/温度/输出上限/存储路径）
├── requirements.txt
└── TUTORIAL.md
```

### 2.3 数据流架构

```
用户输入
    │
    ▼
┌─────────────────────────────────────────┐
│            NovelCreationCLI              │
│  (交互循环 / 自动推进，包装为 HumanMessage) │
└──────────────┬──────────────────────────┘
               │ graph.astream({messages}, thread_id)
               ▼
┌─────────────────────────────────────────┐
│          Deep Agent (编译后的图)           │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │       Supervisor (主智能体)        │   │
│  │  - 分析意图、统筹推进              │   │
│  │  - 调用 task() 委派子智能体        │   │
│  │  - 独占定稿/删除/精准修改/进度/导出 │   │
│  └──────┬───────────────────────────┘   │
│         │ task() 工具调用                 │
│         ▼                                │
│  ┌─────────────────────────────────┐    │
│  │    SubAgentMiddleware (子智能体)   │    │
│  │  ┌─────────┐ ┌─────────┐        │    │
│  │  │Architect│ │ Writer  │  ...   │    │
│  │  │ 只写草稿 │ │ 只写正文 │        │    │
│  │  └─────────┘ └─────────┘        │    │
│  └─────────────────────────────────┘    │
└──────────────┬──────────────────────────┘
               │ 工具调用 → MCP 子进程 → SQLite + ChromaDB
               ▼
         AI 响应显示给用户
```

---

## 3. 环境搭建

### 3.1 前置条件

- Python >= 3.10
- 一个 DeepSeek API 密钥（[获取地址](https://platform.deepseek.com)）
- 一个 embedding API 密钥（本项目用 qwen3.7-text-embedding，阿里百炼兼容端点）

### 3.2 安装步骤

```bash
# 1. 进入项目目录
cd wangwen_creat

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 EMBEDDING_API_KEY
```

### 3.3 环境变量详解

`.env` 文件里每个变量的含义（密钥不进 YAML）：

```bash
# ── API 密钥（供 conf/app_config.yaml 的 api_key_env 引用）──
DEEPSEEK_API_KEY=sk-your-key-here       # DeepSeek 对话模型密钥
EMBEDDING_API_KEY=sk-your-embedding     # embedding 密钥（向量检索用）
EMBEDDING_BASE_URL=https://...maas.../compatible-mode/v1  # embedding 端点
EMBEDDING_MODEL=qwen3.7-text-embedding  # embedding 型号

# ── 智能体温度（0~2，越高越有创意、越不稳定）──
ARCHITECT_TEMPERATURE=1.5   # 大纲设计师：高创意
WRITER_TEMPERATURE=0.8      # 章节撰写者：中高创意
EDITOR_TEMPERATURE=0.2      # 编辑审核：低创意/高一致性
READER_TEMPERATURE=0.5      # 读者检查：平衡
SUPERVISOR_TEMPERATURE=0.1  # 总编决策：低创意/高稳定

# ── 智能体单次输出上限（max_tokens）──
ARCHITECT_MAX_TOKENS=200000
WRITER_MAX_TOKENS=200000
EDITOR_MAX_TOKENS=16384
READER_MAX_TOKENS=16384
SUPERVISOR_MAX_TOKENS=16384

# ── 存储路径 ──
SQLITE_DB_PATH=data/novels.db          # 长期记忆
CHROMA_DB_PATH=data/vector_db/         # 向量存储

# ── 日志级别 ──
LOG_LEVEL=INFO
```

> 说明：模型名（`deepseek-v4-pro` / `deepseek-v4-flash`）不在 `.env`，而在 `conf/app_config.yaml` 的 `models` 段里，`.env` 只放密钥和温度/输出上限等运行时参数。

### 3.4 启动系统

```bash
python -m app.main
```

启动后会显示横幅，输入 `帮助` 查看命令，输入自然语言描述创作需求即可。

---

## 4. 核心技术详解

### 4.1 模型池：多供应商/多型号切换

#### 问题

旧版 LangChain 中，不同提供商需要导入不同的类，切换模型要改代码，且**无法让不同智能体用不同供应商/型号**。

#### 解决方案：模型注册表（ModelRegistry）

`app/core/model_client.py` 中的 `ModelRegistry` 实现了「模型池 + 智能体映射」的配置驱动架构：

```
conf/app_config.yaml
├── models:                          # 模型池（可定义任意多个槽位）
│   ├── deepseek_pro:    {provider: deepseek, model: deepseek-v4-pro}
│   ├── deepseek_flash:  {provider: deepseek, model: deepseek-v4-flash}
│   ├── qwen_plus:       {provider: openai,  model: qwen-plus}       # 另一供应商
│   └── ollama_qwen:     {provider: openai,  model: qwen2.5:72b}     # 本地
│
└── agents:                          # 智能体 → 模型槽位映射
    ├── supervisor: {model: deepseek_pro}     # 总编用 pro（强推理）
    ├── architect:  {model: deepseek_pro}     # 设计师用 pro（创意）
    ├── writer:     {model: deepseek_flash}   # 撰写者用 flash（长文本快）
    ├── editor:     {model: deepseek_flash}   # 编辑用 flash（省成本）
    └── reader:     {model: deepseek_flash}   # 读者用 flash
```

**核心能力**：同供应商不同型号并存、跨供应商混用、切换零代码、密钥与代码分离。

#### 底层实现：init_chat_model

```python
# app/core/model_client.py 的 _build_model
model = init_chat_model(
    model=slot.model,              # "deepseek-v4-pro"
    model_provider=slot.provider,  # "deepseek"
    api_key=slot.api_key,          # 从环境变量读取
    base_url=slot.base_url or None,
    temperature=temperature,       # 每个智能体不同
    max_tokens=max_tokens,         # 每个智能体不同（单次输出上限）
    stream_chunk_timeout=None,     # 关闭流式 chunk 超时，容忍长生成中途停顿
)
```

#### 每个智能体的模型绑定与参数（当前默认值）

| 智能体 | 模型槽位 | 实际型号 | Temperature | max_tokens |
|--------|---------|---------|------------|-----------|
| Supervisor | deepseek_pro | deepseek-v4-pro | 0.1 | 16384 |
| Architect | deepseek_pro | deepseek-v4-pro | 1.5 | 200000 |
| Writer | deepseek_flash | deepseek-v4-flash | 0.8 | 200000 |
| Editor | deepseek_flash | deepseek-v4-flash | 0.2 | 16384 |
| Reader | deepseek_flash | deepseek-v4-flash | 0.5 | 16384 |

> 注意：这些值来自 `conf/app_config.yaml`（模型映射）+ `.env`（温度/输出上限）。`.env` 覆盖 `app/core/config.py` 里的默认值。

#### Temperature 的含义

```
t=0.0  → 完全确定性（适合代码/评分）
t=0.1  → 极低创意，稳定遵循指令（适合 Supervisor 统筹）
t=0.2  → 低创意，输出稳定（适合 Editor 审核）
t=0.5  → 平衡（适合 Reader 检查）
t=0.8  → 中高创意（适合 Writer 创作）
t=1.5  → 高创意（适合 Architect 天马行空设计）
t=2.0  → 最高创意，可能产生随机输出
```

---

### 4.2 统一提示词管理

所有智能体的系统提示词集中存储在 `conf/prompts.yaml`，每个智能体一个 `system_prompt` 块。

#### 加载 (`app/prompts/__init__.py`)

```python
def load_system_prompt(agent_name: str) -> str:
    prompts_data = _load_prompts_yaml()
    return prompts_data[agent_name]["system_prompt"].strip()

# 预加载为模块级常量
SUPERVISOR_PROMPT = load_system_prompt("supervisor")
ARCHITECT_PROMPT = load_system_prompt("architect")
WRITER_PROMPT   = load_system_prompt("writer")
EDITOR_PROMPT  = load_system_prompt("editor")
READER_PROMPT  = load_system_prompt("reader")
```

#### 技能内联

`writer` 和 `reader` 的提示词会**内联加载**两个本地技能（把 `skills/*/SKILL.md` 的内容拼进系统提示词）：

- `writing-style`：网文写作风格（画面感、打斗节奏、口语化）
- `novel-anti-ai-style`：去 AI 味（反上帝视角、限知第三人称、去模板句式）

这样 writer 写正文、reader 查文风时，都会强制遵守这两套约束，而无需额外加载机制。

---

### 4.3 DeepAgents 框架与子智能体

#### 主 Agent 与 SubAgent

`create_deep_agent()` 构建主 Agent（Supervisor），`subagents` 参数传入子智能体规格：

```python
SubAgent = {
    "name": "architect",              # 名称（task 工具用此名调用）
    "description": "设计人物和世界观",  # 描述（Supervisor 据此决定何时调用）
    "system_prompt": "你是...",        # 系统提示词
    "model": ChatDeepSeek(...),        # 模型实例（可有自己的 temperature）
    "tools": [tool1, tool2, ...],      # 该子智能体可用的工具
}
```

**工作流程**：Supervisor 收到需求 → 分析后调用 `task("architect", "...")` → `SubAgentMiddleware` 启动子智能体 → 子智能体用自己的提示词/模型/工具执行 → 结果返回 Supervisor。

> 子智能体是**无状态**的：每次 `task()` 委派都从零开始，不保留上次委派的上下文。这就是为什么它们需要靠记忆工具（而非「记得」）来获取上下文——也是「重复读取」问题的根源之一（见 4.7）。

#### 中间件栈（本项目已裁剪）

`create_deep_agent` 内部默认挂一堆中间件，本项目做了裁剪：

- ✅ 保留：`FilesystemMiddleware`（backend，但文件工具已禁用）、`SubAgentMiddleware`（task 调度）
- ❌ **移除：`SummarizationMiddleware`**——硬约束「不截断子智能体的产出和输入」，不做中途上下文压缩
- ✅ 新增：`HarnessProfile` 全局禁用文件系统工具 + general-purpose 子智能体（根治 Windows 下 read_file 死循环）

#### 文件系统工具禁用（重要）

deepagents 会自动给主 Agent 添加文件探索工具（`read_file`/`ls`/`glob`/…），但这些工具在 Windows 上存在硬编码缺陷（拒绝 `D:\` 绝对路径），会导致「读→失败→换路径→再读」死循环。本项目通过 `_register_no_filesystem_profile()` 在 provider 级别全局排除这些工具：

```python
_FS_TOOLS = frozenset({"ls","read_file","write_file","edit_file","delete","glob","grep","execute"})
```

---

### 4.4 MCP 工具暴露与权限边界

#### 为什么用 MCP

所有记忆工具统一通过 **MCP（Model Context Protocol）** 暴露：`app/mcp/server.py` 是一个 FastMCP stdio 服务器，主进程用 `langchain_mcp_adapters` 建**持久 session** 加载工具（避免每次调用重新 spawn 子进程 + 重新初始化 SQLite/ChromaDB/embedding）。

#### 工具清单（25 个）

**读工具（14 个，所有智能体共享）**：

`list_novels` `get_novel_state` `get_story_bible` `get_novel_outline` `get_character_profile` `get_world_building` `get_writing_context` `get_novel_progress` `search_long_term_memory` `get_short_term_context` `get_chapter` `search_similar_content` `get_writing_issues` `read_source_docs`

**写工具（11 个，权限边界见下）**：

`create_novel` `update_novel_progress` `save_to_long_term` `save_chapter` `patch_chapter` `update_short_term` `save_writing_issue` `export_chapters` `delete_long_term_entry` `save_run_log` `lock_entry`

#### 权限边界（硬约束）

硬约束：「只有 Supervisor 拥有修改存储记忆的权限」。实现为：

| 角色 | 工具 | 说明 |
|------|------|------|
| Supervisor | 全部 25 个 | 独占定稿/删除/精准修改/进度/导出/日志/建项目 |
| Architect | 14 读 + `save_to_long_term` | 首建设计草稿（人物/世界观/大纲） |
| Writer | 14 读 + `save_chapter` + `update_short_term` | 写正文/子情节草稿 |
| Editor | 14 读 + `save_writing_issue` | 记审核问题 |
| Reader | 14 读 + `save_writing_issue` | 记一致性问题 |

**Supervisor 独占（子智能体拿不到）**：`create_novel` `update_novel_progress` `patch_chapter` `export_chapters` `delete_long_term_entry` `save_run_log` `lock_entry`。

这保证了：子智能体只能写自己的**草稿**，最终定稿（`lock_entry`）、删除、精准修改、进度、导出全部由 Supervisor 拍板。

---

### 4.5 三层记忆系统

#### 第一层：长期记忆（SQLite，`data/novels.db`）

| 表名 | 内容 |
|------|------|
| `novels` | 小说项目（novel_id, title, genre, synopsis, status, current_chapter） |
| `characters` | 人物档案（name, role_type, personality, background, locked） |
| `world_settings` | 世界观设定 |
| `outlines` | 大纲条目（chapter_seq, summary, key_events） |
| `main_plots` | 主线情节 |
| `source_docs` | 每个项目自己的源文档（多项目隔离） |
| `character_registry` | 权威角色名表（落库前硬校验） |

#### 第二层：短期记忆（SQLite，`data/short_term.db`）

| 表名 | 内容 |
|------|------|
| `sub_plots` | 子情节/支线 |
| `chapter_drafts` | 章节草稿（多版本，draft_id + version） |

每次重写保留历史版本，可追溯修改历程。

#### 第三层：向量检索（ChromaDB，`data/vector_db/`）

| Collection | 内容 | 用途 |
|-----------|------|------|
| `novel_characters` | 人物描述嵌入 | "哪个角色勇敢正直？" |
| `novel_settings` | 世界观嵌入 | "魔法体系设定有哪些？" |
| `novel_plots` | 情节片段嵌入 | "暗线伏笔有哪些？" |
| `chapter_content` | 章节正文嵌入 | "找到写战斗场景的段落" |

> 关键实现：ChromaDB **不绑定** embedding function，而是由代码显式调用 embedding API 生成向量后传入——这样重启后向量仍生效，不依赖 ChromaDB 自动嵌入（那无法持久化）。

---

### 4.6 多项目隔离与角色名防护

这是本项目为「角色人名混乱」和「多项目串数据」问题做的两个重要加固。

#### 多项目隔离

- 每个 `novel_id` 拥有自己的**源文档**（`source_docs` 表）、**权威名表**（`character_registry` 表）、**易混淆字表**。
- `create_novel(title, ..., source_dir)` 会读取该项目的源文档目录，把源文档、名表按 novel_id 落库。
- 不同项目的角色名互相隔离：A 项目拒绝 B 项目的角色名，反之亦然。

#### 角色名硬校验

- 从源文档提取权威角色名（`###` 标题 + `**name**：` 加粗 + 群体成员），排除「外貌/性格/动机」等字段标签误提。
- 落库前校验：角色名必须与权威名表逐字一致（`枫` ≠ `楓`，`林峰` ≠ `林鳳`），不一致直接**代码层拒绝写入**。
- 正文保存时用字符级易混淆映射（`_CONFUSABLE_CHARS`）自动纠正错字：`枫→峰/风/風/楓/栴`，`荆→荊`，`静→靖` 等。

---

### 4.7 上下文控制与 token 节约

历史上曾出现 O(N²) token 爆炸（Supervisor 每轮重发完整历史）。现用以下手段控制（**不靠中途截断**）：

1. **定稿锁 `lock_entry`**：定稿时只传 category+条目名，不重发全文。
2. **极简进度卡 `get_novel_progress`**：日常推进只拉几 KB 的进度卡，而非十几万 token 的全量快照 `get_novel_state`。
3. **一次性写作上下文 `get_writing_context`**：Writer 一次拉全「故事圣经+本单元大纲+上一章结尾+历史问题」，替代多次零散查询。
4. **源文档只读一次**：architect 首轮 `read_source_docs` 消化成结构化设定，此后一律从记忆库取，禁止重读 120KB 源文档。
5. **按单元重置**：每完成一个情节单元（约 5 章）落库进度后，重启 Supervisor 的消息历史（新 thread_id + `get_novel_progress` 恢复），保持上下文有界。
6. **子智能体最小读取**：editor/reader 拆成「大纲审核」「章节审核」两条流程，各自只读所需工具，避免「读一切」。

---

## 5. 代码逐文件解析

### 5.1 `app/agent.py` — 核心编排文件

```python
from deepagents import create_deep_agent, SubAgent
from app.core.model_client import get_model_registry
from app.prompts import SUPERVISOR_PROMPT, ARCHITECT_PROMPT, WRITER_PROMPT, EDITOR_PROMPT, READER_PROMPT
```

#### 权限集合定义

```python
_FS_TOOLS = frozenset({"ls","read_file","write_file","edit_file","delete","glob","grep","execute"})

_WRITE_TOOLS = frozenset({
    "create_novel", "update_novel_progress", "save_to_long_term",
    "save_chapter", "patch_chapter", "update_short_term",
    "save_writing_issue", "export_chapters", "delete_long_term_entry",
    "save_run_log", "lock_entry",
})

# 子智能体各自的写库职责边界
_ARCHITECT_WRITE_TOOLS = frozenset({"save_to_long_term"})
_WRITER_WRITE_TOOLS    = frozenset({"save_chapter", "update_short_term"})
_EDITOR_WRITE_TOOLS    = frozenset({"save_writing_issue"})
```

#### `_build_sub_agents(all_tools, registry)`

```python
def _build_sub_agents(all_tools, registry):
    read_tools = [t for t in all_tools if t.name not in _WRITE_TOOLS]  # 14 个读工具
    def _tools_for(write_names):
        return read_tools + [t for t in all_tools if t.name in write_names]
    return [
        {"name": "architect", "system_prompt": ARCHITECT_PROMPT,
         "model": registry.get_model("architect"), "tools": _tools_for(_ARCHITECT_WRITE_TOOLS)},
        {"name": "writer",    "system_prompt": WRITER_PROMPT,
         "model": registry.get_model("writer"),    "tools": _tools_for(_WRITER_WRITE_TOOLS)},
        {"name": "editor",    "system_prompt": EDITOR_PROMPT,
         "model": registry.get_model("editor"),    "tools": _tools_for(_EDITOR_WRITE_TOOLS)},
        {"name": "reader",    "system_prompt": READER_PROMPT,
         "model": registry.get_model("reader"),    "tools": _tools_for(_EDITOR_WRITE_TOOLS)},
    ]
```

> 注意：`_build_sub_agents` 返回的是 **dict 列表**（deepagents 接受的规格），不是旧版的手写工具过滤。子智能体共享 14 个读工具 + 各自的写工具；**不配任何 summarization 中间件**。

#### `create_novel_agent(checkpoint_db_path=None)`

```python
def create_novel_agent(checkpoint_db_path=None):
    registry = get_model_registry()
    registry.warmup_models()                    # 并行预热 5 个模型实例

    all_tools = _load_mcp_tools()               # 通过 MCP 持久 session 加载 25 个工具

    from deepagents.backends.filesystem import FilesystemBackend
    fs_backend = FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=False)

    sub_agents = _build_sub_agents(all_tools, registry)

    checkpointer = None                          # AsyncSqliteSaver（astream 异步运行）
    if checkpoint_db_path:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async def _make_saver():
            conn = await aiosqlite.connect(checkpoint_db_path)
            return AsyncSqliteSaver(conn)
        checkpointer = run(_make_saver())

    _register_no_filesystem_profile()            # 禁用文件系统工具 + general-purpose

    agent = create_deep_agent(
        model=registry.get_model("supervisor"),
        tools=all_tools,                         # Supervisor 拥有全部 25 个工具
        system_prompt=SUPERVISOR_PROMPT,
        subagents=sub_agents,
        checkpointer=checkpointer,
        backend=fs_backend,
        name="novel_supervisor",
    )
    return agent
```

> 对比旧版：**没有** `_build_summarization_middleware`，**没有** `middleware=[...]`。`fs_backend` 仍保留（deepagents 需要文件系统 backend，但文件工具已通过 HarnessProfile 禁用）。

### 5.2 `app/main.py` — 交互式 CLI

关键设计在 `NovelCreationCLI`：

- `thread_id` 是 LangGraph checkpoint 的隔离键，不同会话历史通过它区分。
- `astream(..., stream_mode="messages")` 逐 token 流式打印（MCP 工具是 async-only，必须用 astream）。
- **自动推进**：检测到「创作总结」才停；否则持续喂「继续推进」指令。
- **大纲审核硬约束**：大纲定稿、正文尚未开始时，必须交用户审核。
- **按单元重置**：`UNIT_CHAPTERS=5`，跨过单元边界后落库进度并重启 Supervisor 上下文（新 thread_id + `get_novel_progress` 恢复）。
- **运行追踪**：每次任务注入 `RunTracer` 回调，记录 token 消耗与工具调用。

### 5.3 `app/tools/factory.py` — 工具工厂

```python
class NovelMemoryTools:
    def __init__(self, ltm, stm, vs):
        self._ltm = ltm    # 长期记忆后端（注入）
        self._stm = stm    # 短期记忆后端
        self._vs = vs      # 向量后端

    def _search_similar_content(self, query, novel_id="", collection="chapter_content", k=5):
        """向量语义检索"""
        where = {"novel_id": novel_id} if novel_id else None
        results = self._vs.search(collection, query, k=k, where=where)
        ...

    def get_tools(self) -> list:
        # 把实例方法包装为 tool 并重命名，返回 25 个工具
        ...
```

**关键方法**（除记忆读写外，还承载加固逻辑）：

- `_save_to_long_term`：写前做角色名硬校验（权威名表核对 + 易混淆字纠正）。
- `_save_chapter`：正文写前自动纠正易混淆角色名（`林峰`→`林楓`）。
- `patch_chapter`：精准替换（old_text→new_text），只改硬伤不改文笔。
- `_lock_entry`：定稿加锁（locked=True），锁定条目只能被 locked=True 覆盖。
- `_get_novel_progress`：极简进度卡（当前章号 + 名表 + 大纲目录）。
- `_get_writing_context`：Writer 一次性写作上下文。
- `_create_novel` / `_ingest_source_docs`：建项目 + 按 novel_id 落库源文档/名表。

### 5.4 `app/prompts/__init__.py` — 提示词管理

```python
def load_system_prompt(agent_name: str) -> str:
    # 从 conf/prompts.yaml 加载，并对 writer/reader 内联 skills
    ...
```

### 5.5 `app/core/config.py` 和 `app/core/model_client.py`

**config.py**（Pydantic Settings，环境变量优先）：

```python
class Settings(BaseSettings):
    llm_provider: str = Field(default="deepseek", alias="LLM_PROVIDER")
    llm_model: str = Field(default="deepseek-v4-pro", alias="LLM_MODEL")
    max_tokens: int = Field(default=200000, alias="LLM_MAX_TOKENS")

    architect_temperature: float = Field(default=1.5, alias="ARCHITECT_TEMPERATURE")
    writer_temperature:    float = Field(default=0.8, alias="WRITER_TEMPERATURE")
    editor_temperature:    float = Field(default=0.2, alias="EDITOR_TEMPERATURE")
    reader_temperature:    float = Field(default=0.5, alias="READER_TEMPERATURE")
    supervisor_temperature: float = Field(default=0.1, alias="SUPERVISOR_TEMPERATURE")

    architect_max_tokens: int = Field(default=200000, alias="ARCHITECT_MAX_TOKENS")
    writer_max_tokens:    int = Field(default=200000, alias="WRITER_MAX_TOKENS")
    editor_max_tokens:    int = Field(default=16384,  alias="EDITOR_MAX_TOKENS")
    reader_max_tokens:    int = Field(default=16384,  alias="READER_MAX_TOKENS")
    supervisor_max_tokens: int = Field(default=16384, alias="SUPERVISOR_MAX_TOKENS")
```

**model_client.py**（模型池）：从 `conf/app_config.yaml` 加载 `models` 段和 `agents` 段，`_temperature_for`/`_max_tokens_for` 按智能体名取温度/输出上限，模型实例按 `slot@temperature@max_tokens` 缓存。

---

## 6. 核心设计模式

### 6.1 Supervisor 编排模式

```
                   ┌──────────────┐
                   │  Supervisor  │  ← 主智能体，分析意图、分配任务、独占定稿权
                   │   t = 0.1    │
                   └──────┬───────┘
                          │ task() 工具调用
            ┌─────────────┼─────────────┬─────────────┐
            ▼             ▼              ▼             ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
     │Architect │ │ Writer   │ │ Editor   │ │ Reader   │
     │ t = 1.5  │ │ t = 0.8  │ │ t = 0.2  │ │ t = 0.5  │
     └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

- **Supervisor 用最低温度 0.1**：任务分配需要一致性，相同的输入应产生相同的路由决策。
- **Architect 用最高温度 1.5**：创意工作需要多样性。

### 6.2 闭包工厂模式（工具）

依赖通过 `__init__` 显式注入，方法通过 `self` 访问后端，`get_tools()` 是唯一外部接口。优势：测试传 mock、线程安全、依赖显式。

### 6.3 单例模式（配置）

`get_settings()` / `get_model_registry()` 延迟初始化，全应用共享同一份配置；测试用 `reset_model_registry()` / `reload_settings()` 重置。

### 6.4 工具最小权限原则

```
Supervisor:  25 tools（全部，独占定稿/删除/精准修改/进度/导出）
Architect:   15 tools（14 读 + save_to_long_term）
Writer:      16 tools（14 读 + save_chapter + update_short_term）
Editor:      15 tools（14 读 + save_writing_issue）
Reader:      15 tools（14 读 + save_writing_issue）
```

这防止 Writer 意外修改人物设定，或 Editor 意外删除大纲、定稿内容。

---

## 7. 自定义和扩展指南

### 7.1 添加新的子智能体

**步骤 1**：在 `conf/prompts.yaml` 添加提示词。

**步骤 2**：在 `app/prompts/__init__.py` 加载为常量。

**步骤 3**：在 `app/agent.py` 的 `_build_sub_agents` 里加一项（决定它的专属写工具）：

```python
{"name": "translator",
 "description": "专业文学翻译家，将小说翻译为其他语言。",
 "system_prompt": TRANSLATOR_PROMPT,
 "model": registry.get_model("translator"),   # 需在 app_config.yaml 的 agents 段加映射
 "tools": _tools_for(frozenset({"save_chapter"})),  # 只给读工具 + 需要的写工具
}
```

> 新增的写工具若要给子智能体用，记得加进对应 `_*_WRITE_TOOLS`；定稿/删除/进度等写工具**不要**下放给子智能体。

### 7.2 添加新的记忆工具

1. 在 `app/memory/long_term.py`（或 short_term/vector_store）加后端方法。
2. 在 `app/tools/factory.py` 加 `_xxx` 方法 + 在 `get_tools()` 里包装。
3. 在 `app/mcp/server.py` 用 `mcp.tool()` 暴露给 MCP（让主进程能通过 MCP 加载到它）。
4. 若它是写工具，加入 `_WRITE_TOOLS`（以及某个子智能体的 `_*_WRITE_TOOLS`，如果需要下放）。

### 7.3 调整智能体温度 / 输出上限

编辑 `.env`：

```bash
ARCHITECT_TEMPERATURE=1.3    # 设计师略微收敛
READER_TEMPERATURE=0.1       # 读者检查更严格稳定
SUPERVISOR_TEMPERATURE=0.1   # 总编决策稳定
```

> 若模型输出过长导致服务端断连，优先下调 `ARCHITECT_MAX_TOKENS` / `WRITER_MAX_TOKENS`（见 8.3）。

### 7.4 更换/切换模型

模型池和映射都在 `conf/app_config.yaml`，切换零代码：

```yaml
models:
  deepseek_pro:    {provider: deepseek, model: deepseek-v4-pro, base_url: https://api.deepseek.com, api_key_env: DEEPSEEK_API_KEY}
  deepseek_flash:  {provider: deepseek, model: deepseek-v4-flash, base_url: https://api.deepseek.com, api_key_env: DEEPSEEK_API_KEY}

agents:
  supervisor: {model: deepseek_pro}
  architect:  {model: deepseek_pro}
  writer:     {model: deepseek_flash}
  editor:     {model: deepseek_flash}
  reader:     {model: deepseek_flash}
```

密钥只在 `.env`（`DEEPSEEK_API_KEY`），不进 YAML。

---

## 8. 常见问题与调试

### 8.1 模型密钥未设置

**错误**：`If using default api base, DEEPSEEK_API_KEY must be set`

**解决**：确认 `conf/app_config.yaml` 里槽位的 `api_key_env` 指向的变量在 `.env` 中存在且名称完全一致。

### 8.2 智能体不按预期调用 task()

**原因**：Supervisor 提示词不够明确。

**解决**：在 `conf/prompts.yaml` 的 supervisor 段强化工作流指令；同时可下调 `SUPERVISOR_TEMPERATURE` 到 0.0~0.1 提高遵循度。

### 8.3 流式输出被服务端中断（peer closed connection）

**错误**：`peer closed connection without sending complete message body (incomplete chunked read)`

**原因**：单次输出上限设得过大（`ARCHITECT_MAX_TOKENS`/`WRITER_MAX_TOKENS` 默认 200000），模型一次生成超长内容、跑太久，DeepSeek 服务器主动断连。

**解决**：
- 把 `ARCHITECT_MAX_TOKENS` / `WRITER_MAX_TOKENS` 降到服务端实际支持的量级（如 16384）。
- 让 architect **分阶段出大纲**（一次一个阶段，别一次全 2000 章），每回合输出有界。
- 给模型加 `max_retries`、在流处理里对断连做重试。

### 8.4 子智能体反复读取同一内容（token 膨胀）

**现象**：editor/reader 一次审核输入 token 从几 K 涨到几十万。

**原因**：子智能体无状态（每次委派从零开始）+ 提示词把「读一切」都写成必做 + 单次委派内上下文无界累积。

**解决**：让子智能体走「最小读取」流程（如 editor 的「大纲审核」只读大纲、「章节审核」只读该单元正文），并明确「同一工具同一内容只调一次」。

### 8.5 如何查看对话历史

```python
from app.agent import create_novel_agent
agent = create_novel_agent(checkpoint_db_path="data/checkpoints.db")
config = {"configurable": {"thread_id": "session_abc123"}}
state = agent.get_state(config)
for msg in state.values.get("messages", []):
    print(f"[{type(msg).__name__}] {msg.content[:100]}")
```

### 8.6 如何重置记忆系统

```bash
rm data/novels.db data/short_term.db data/checkpoints.db
rm -rf data/vector_db/
python -m app.main   # 重启自动创建空库
```

### 8.7 调试工具调用 / 审计 token

- `.env` 设 `LOG_LEVEL=DEBUG` 看工具调用输入输出。
- 看 `data/tracing/latest.txt`（运行追踪快照：token 消耗 + 工具调用明细）。
- 看 `output/run_logs/<novel_id>/`（save_run_log 导出的阶段日志）。

---

## 小结

本教程覆盖了从环境搭建到核心架构再到自定义扩展的完整流程。系统核心理念：

1. **模型注册表（ModelRegistry）** 支持多供应商、多型号，每个智能体独立指定模型/温度/输出上限，切换零代码
2. **MCP 工具暴露** 统一记忆工具接口，持久 session 复用，避免反复初始化后端
3. **权限收归 Supervisor** 定稿/删除/精准修改/进度/导出独占，子智能体只写草稿
4. **三层记忆** 分层管理不同时间尺度的创作上下文，向量检索用显式 embedding
5. **多项目隔离 + 角色名硬校验** 从源头杜绝串数据和人名混乱
6. **不截断 + 最小读取** 用「避免重复读取/重复调用」而非「中途压缩」来节约 token
7. **技能内联** writing-style + novel-anti-ai-style 约束 writer/reader 的文风与去 AI 味

详细的代码实现请参考项目源文件，每个函数都有中文文档字符串说明。
