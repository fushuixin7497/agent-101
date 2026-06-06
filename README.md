# 🤖 Agent 101 —— 从零手写 AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-managed-purple.svg)](https://github.com/astral-sh/uv)

> **从零开始，不用框架，手写理解 Agent 的每一个原子能力。**

这是一个面向开发者的渐进式学习仓库。我们不使用 LangChain、AutoGPT 等现成框架，而是从**最底层的 HTTP 请求**出发，逐步构建一个完整、健壮的 Agent 系统。

## 🎯 适合谁？

- 想深入理解 Agent 原理，而不只是调包调参的开发者
- 有 Python 基础（能刷 LeetCode），但没做过大型项目的同学
- 希望建立系统化知识图谱，而非碎片化教程的学习者

## 🗺️ 课程路线图

| 模块 | 主题 | 核心能力 | 代码位置 |
|------|------|---------|---------|
| **Lesson 01** | 环境与第一次 API 调用 | 理解 LLM 请求/响应最小结构、Token 计费、流式 vs 非流式 | [`lesson-01/`](lesson-01/) |
| **Lesson 02** | 手写 Agent Loop | 状态机循环、Tool Calling 协议、工具注册表 | [`lesson-02/`](lesson-02/) |
| **Lesson 03** | 工具设计 | 参数校验、幂等性、错误处理、description 即 prompt | [`lesson-03/`](lesson-03/) |
| Lesson 04 | 记忆与上下文管理 | 滑动窗口、摘要压缩、长期记忆存储 | 🚧 待更新 |
| Lesson 05 | 结构化输出与规划 | JSON Mode、ReAct / CoT 思考链 | 🚧 待更新 |
| Lesson 06 | RAG 与知识检索 | 向量数据库、文档切片、Embedding | 🚧 待更新 |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（Python 包管理器）
- 一个大模型 API Key（OpenAI / [Kimi](https://platform.moonshot.cn) / [DeepSeek](https://platform.deepseek.com)）

### 1. 克隆仓库

```bash
git clone https://github.com/fushuixin7497/agent-101.git
cd agent-101
```

### 2. 选择课时进入

每个 `lesson-xx` 都是独立的 uv 项目，可直接运行：

```bash
cd lesson-01
export LLM_API_KEY="sk-xxxxx"
export LLM_BASE_URL="https://api.moonshot.cn/v1"  # 或 OpenAI / DeepSeek
export LLM_MODEL="moonshot-v1-8k"

uv sync        # 安装依赖
uv run python chat.py
```

> 💡 **提示**：把 `export` 语句写入 `~/.zshrc` 并执行 `source ~/.zshrc`，可永久生效。

### 3. 各课时运行命令

| 课时 | 主要脚本 | 运行命令 |
|------|---------|---------|
| `lesson-01` | `chat.py` — 多轮对话 + Token 监控 | `uv run python chat.py` |
| `lesson-01` | `raw_call.py` — 观察裸 HTTP 结构 | `uv run python raw_call.py` |
| `lesson-02` | `agent.py` — Agent Loop + 工具调用 | `uv run python agent.py` |
| `lesson-03` | `agent.py` — 健壮工具 + 幂等设计 | `uv run python agent.py` |

## 📂 项目结构

```
agent-101/
├── lesson-01/              # 模块 1：环境与第一次 API 调用
│   ├── chat.py             # 多轮对话脚本（流式/非流式切换）
│   ├── raw_call.py         # 裸 HTTP 请求/响应观察
│   └── README.md           # 课时文档
├── lesson-02/              # 模块 2：手写 Agent Loop
│   ├── agent.py            # 最小完整的多步 Agent
│   └── README.md           # 课时文档
├── lesson-03/              # 模块 3：工具设计
│   ├── tools.py            # 工具注册表 + Schema + 健壮实现
│   ├── agent.py            # 接入健壮工具的 Agent Loop
│   └── README.md           # 课时文档
├── .gitignore
└── README.md               # 本文件
```

## 🎓 学习建议

1. **按顺序学**：每个模块建立在前一个之上，不要跳课。
2. **先跑起来，再抠细节**：先让代码运行，观察现象，建立直觉，再回头理解语法。
3. **改参数做实验**：改 `temperature`、改 `description`、故意传错参数，看模型怎么反应。
4. **读报错信息**：模块 2/3 的错误信息是设计的一部分，模型会根据错误提示自我修正。

## 🔑 核心概念速查

| 概念 | 一句话解释 |
|------|-----------|
| **System Prompt** | 模型的全局人设指令，不在对话中显示但每轮都参与计算 |
| **Messages** | 对话历史数组，`system` → `user` → `assistant` → `user` → `assistant`... |
| **Token** | 模型计费的最小单位，上下文越长 `prompt_tokens` 越大，费用越高 |
| **Tool Calling** | 模型生成调用指令（`tool_calls`），由本地 Python 执行，结果回填给模型继续推理 |
| **Agent Loop** | 调模型 → 解析工具调用 → 执行工具 → 结果回填 → 重复直到 `finish_reason="stop"` |
| **幂等性** | 同样输入执行 N 次效果与 1 次相同，防止有副作用工具被重复调用时产生混乱 |

## 🛠️ 技术栈

- **HTTP 客户端**：[`httpx`](https://www.python-httpx.org/)（直接发裸请求，理解底层结构）
- **包管理**：[`uv`](https://github.com/astral-sh/uv)（极速、现代）
- **协议**：OpenAI 兼容 Chat Completions API（支持 GPT-4o、Kimi、DeepSeek 等）

## 📜 License

[MIT](LICENSE) — 自由学习、自由使用、自由分享。

---

> **Agent 不是魔法，是结构。** 祝你学习愉快！
