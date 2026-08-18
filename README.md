# 小说创作智能体系统 (Novel Creation Agent System)

基于 LangGraph + MCP 的多智能体小说创作系统。采用 Supervisor 编排模式，由 4 个专用智能体协同完成小说创作全流程。

## 系统架构

```
用户 → Supervisor(主控) → Architect(大纲) → Writer(撰写) → Editor(编辑) → Reader(读者)
         ↕                ↕               ↕              ↕             ↕
                        MCP 统一接口层
         ↕                ↕               ↕              ↕             ↕
              SQLite(长期记忆) + ChromaDB(向量检索) + SQLite(短期记忆)
```

## 智能体角色

| 智能体 | 角色 | Temperature | 职责 |
|-------|------|------------|------|
| Supervisor | 总控 | 0.3 | 任务分配、信息路由、状态评估 |
| Architect | 大纲制定者 | 1.0 | 人物背景设定、小说整体走向 |
| Writer | 章节撰写者 | 0.8 | 具体章节内容生成 |
| Editor | 出版社编辑 | 0.2 | 质量把控、完整故事审核 |
| Reader | 读者 | 0.5 | 设定一致性、逻辑严谨性检查 |

## 快速开始

### 环境准备

```bash
# Python >= 3.10
python --version

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入真实的 API 密钥
```

### 启动系统

```bash
python -m app.main
```

## 项目结构

```
wangwen_creat/
├── conf/                 # 配置文件
├── app/                  # 主应用
│   ├── core/             # 核心基础设施
│   ├── models/           # 数据模型
│   ├── agents/           # 智能体定义
│   ├── memory/           # 记忆模块
│   ├── mcp/              # MCP 模块
│   ├── graph/            # LangGraph 工作流
│   └── utils/            # 工具函数
├── data/                 # 本地数据存储
│   ├── novels/           # 小说项目
│   └── vector_db/        # ChromaDB
└── tests/                # 测试
```

## 安全性

- API 密钥通过 `.env` 管理，不纳入版本控制
- `data/` 目录不纳入版本控制
- 日志不记录敏感信息
