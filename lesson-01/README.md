# 🚀 Agent 开发第一课：环境与第一次 API 调用

## 一、这个任务在干什么？

Agent（智能体）的本质是：**让大语言模型（LLM）在一个循环中不断思考、调用工具、观察结果、继续思考**。  
而在这套复杂的逻辑之前，你必须先掌握最底层的原子能力——**向 LLM 发送一次 HTTP 请求，并正确解析它的响应**。

本课的目标就是搭好环境，理解一次 LLM 请求/响应的**最小结构**。

---

## 二、核心概念速览

### 1. SDK vs 裸 HTTP
- **SDK**（如 `openai` Python 包）帮你封装了网络请求、重试、鉴权等细节，适合生产。
- **裸 HTTP**（如 `httpx`/`curl`）让你直接看到请求长什么样，适合学习。**本课使用 `httpx`，因为学习 Agent 必须先懂结构。**

### 2. 最小调用三要素

| 要素 | 作用 | 示例 |
|------|------|------|
| **System Prompt** | 定义模型的全局人设/行为边界 | `"你是一个严谨的代码审查助手"` |
| **Messages** | 对话历史，包含 `system` / `user` / `assistant` 三种角色 | `[{"role":"user","content":"你好"}]` |
| **模型参数** | 控制生成行为：`model`（选哪个模型）、`temperature`（随机性）、`stream`（是否流式）等 | `"gpt-4o-mini"`, `temperature=0.7` |

### 3. Token 计费与长度约束
- **Token** 是模型处理文本的最小单位，1 个中文汉字 ≈ 1~2 token，1 个英文单词 ≈ 1 token。
- `prompt_tokens`：你发送的所有内容（system + 历史 + 当前输入）被切分后的数量。
- `completion_tokens`：模型生成的内容被切分后的数量。
- **计费** ≈ 输入单价 × `prompt_tokens` + 输出单价 × `completion_tokens`。
- **长度约束**：模型有 `max_context_length`（如 128K）。messages 越长，费用越高；超过上限时，最旧的消息会被截断，导致模型"遗忘"早期对话。

### 4. 流式（Stream） vs 非流式（Blocking）

| 模式 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **非流式** | 发一个 POST，等服务器把整个回复生成完，一次性返回 JSON | 代码简单，usage（token 消耗）一定准确 | 用户要等很久才能看到第一个字 |
| **流式** | 服务器用 SSE 协议，生成一个 token 就推一个 token | 用户体验接近"实时打字"，延迟低 | 需要客户端逐行解析 `data: {...}`，部分服务商不返回流式 usage |

---

## 三、环境配置

### 1. 安装依赖（已自动完成）

本项目使用 `uv` 管理依赖，已安装 `httpx`：

```bash
cd ~/learn-agent/lesson-01
uv sync        # 安装依赖
```

### 2. 获取 API Key

选一个你喜欢的平台（推荐用 **Kimi** 或 **DeepSeek**，国内访问稳定且有免费额度）：

| 平台 | 注册地址 | 推荐模型 | Base URL |
|------|----------|----------|----------|
| **Moonshot (Kimi)** | [platform.moonshot.cn](https://platform.moonshot.cn) | `moonshot-v1-8k` | `https://api.moonshot.cn/v1` |
| **DeepSeek** | [platform.deepseek.com](https://platform.deepseek.com) | `deepseek-chat` | `https://api.deepseek.com/v1` |
| **OpenAI** | [platform.openai.com](https://platform.openai.com) | `gpt-4o-mini` | `https://api.openai.com/v1` |

> 💡 `gpt-4o-mini` 是 OpenAI 最便宜的模型，适合学习和测试。

### 3. 配置环境变量

在终端执行（以 Kimi 为例，把 `sk-xxxxx` 换成你的真实 Key）：

```bash
export LLM_API_KEY="sk-xxxxx"
export LLM_BASE_URL="https://api.moonshot.cn/v1"
export LLM_MODEL="moonshot-v1-8k"
```

如果想永久生效，写入 `~/.zshrc`：

```bash
echo 'export LLM_API_KEY="sk-xxxxx"' >> ~/.zshrc
echo 'export LLM_BASE_URL="https://api.moonshot.cn/v1"' >> ~/.zshrc
echo 'export LLM_MODEL="moonshot-v1-8k"' >> ~/.zshrc
source ~/.zshrc
```

---

## 四、运行脚本

### 脚本 1：raw_call.py —— 观察一次调用的裸结构

```bash
uv run python raw_call.py
```

这个脚本只做一次请求，然后把**完整的请求 JSON** 和**完整的响应 JSON** 打印出来。  
适合第一次运行时观察：服务器到底回了什么？token 消耗在哪里？

### 脚本 2：chat.py —— 多轮对话 + Token 监控

```bash
uv run python chat.py
```

交互命令：
- 直接输入文字 → 与模型对话
- `reset` → 清空上下文（保留 system prompt）
- `stream` → 切换流式 / 非流式模式
- `quit` / `exit` → 退出

**观察重点：**
1. 连续对话几轮后，注意 `Token => 输入` 数字会**越来越大**——这就是 context 增长带来的成本增长。
2. 输入 `stream` 切换到流式模式，感受逐字输出的体验差异。

#### 开启流式模式的环境变量方式：

```bash
export LLM_STREAM="true"
uv run python chat.py
```

---

## 五、动手练习任务

完成以下操作，确保你理解了每个环节：

1. [ ] 运行 `raw_call.py`，截图保存返回的 JSON 结构。
2. [ ] 运行 `chat.py`，连续对话 5 轮以上，观察 `输入 token` 从多少增长到多少。
3. [ ] 在对话中输入 `reset`，再发一条消息，确认 `输入 token` 跌回低位。
4. [ ] 输入 `stream` 切换模式，对比流式与非流式的输出体验。
5. [ ] （进阶）用 `curl` 手动复刻一次请求：

```bash
curl -s https://api.moonshot.cn/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d '{
    "model": "moonshot-v1-8k",
    "messages": [{"role":"user","content":"你好"}]
  }' | jq
```

---

## 六、文件结构

```
lesson-01/
├── pyproject.toml      # uv 项目配置
├── README.md           # 本文件
├── chat.py             # 多轮对话脚本（主要产出物）
└── raw_call.py          # 裸 HTTP 结构观察脚本
```

---

## 七、常见问题

**Q: 运行时报 401 Unauthorized？**  
A: API Key 未设置或已过期。检查 `echo $LLM_API_KEY` 是否有值。

**Q: 运行时报 429 Too Many Requests？**  
A: 免费额度用完或触发限速。等几秒再试，或换一个服务商。

**Q: 为什么用 `httpx` 而不是官方 `openai` SDK？**  
A: 学习阶段看裸 HTTP 更能理解本质。后续课程会引入 `openai` SDK 做更复杂的 Agent 逻辑。

**Q: 上下文太长怎么办？**  
A: 这是 Agent 开发的核心难题之一。后续课程会讲滑动窗口、摘要压缩、向量检索等解决方案。
