#!/usr/bin/env python3
"""
模块 3：Context 管理核心
========================
处理长对话导致的 context 窗口爆掉问题。

核心策略：
1. 截断（truncate）：丢弃最老的消息，保留 system + 最近消息。
2. 摘要（summarize）：用模型把早期对话压缩成摘要。
3. 外置记忆（memory）：把早期对话写入文件，messages 中只保留引用。

设计原则：
- 常驻 system prompt 永远保留。
- 压缩只作用于动态信息（user / assistant / tool）。
- 提供 token 估算，方便观察窗口占用。
"""

import os
import json
import httpx
from typing import List, Dict, Any, Callable


# ============================================================
# Token 估算
# ============================================================
def estimate_tokens(text: str) -> int:
    """
    粗略估算 token 数：中文字符按 1 token，其他按 4 字符/token。
    教学用，不依赖 tiktoken，简单可控。
    """
    if not text:
        return 0
    cn_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - cn_count
    return cn_count + max(1, other_count // 4)


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """估算单条消息的 token 数。"""
    total = 0
    content = message.get("content") or ""
    total += estimate_tokens(content)

    # tool_calls / function 调用也有 arguments
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        args = tc.get("function", {}).get("arguments", "")
        total += estimate_tokens(args)
    return total


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """估算消息列表的总 token 数。"""
    return sum(estimate_message_tokens(m) for m in messages)


# ============================================================
# Context Manager
# ============================================================
class ContextManager:
    """
    管理对话历史，防止 context 窗口爆掉。

    用法：
        cm = ContextManager(max_tokens=2000, strategy="summarize", client=client)
        cm.set_system_prompt("你是一个聪明的 Agent...")
        cm.add({"role": "user", "content": "..."})
        cm.fit()  # 触发压缩策略
        messages = cm.get_messages()
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        strategy: str = "truncate",
        client: httpx.Client = None,
        headers: Dict[str, str] = None,
        url: str = None,
        model: str = None,
        memory_path: str = None,
    ):
        if strategy not in ("truncate", "summarize", "memory"):
            raise ValueError(f"不支持的 strategy: {strategy}")

        self.max_tokens = max_tokens
        self.strategy = strategy
        self.client = client
        self.headers = headers or {}
        self.url = url
        self.model = model
        self.memory_path = memory_path or os.path.expanduser(
            "~/.learn-agent/lesson-04-memory.md"
        )

        self.messages: List[Dict[str, Any]] = []
        self.system_prompt: str = ""

        # 统计信息
        self.last_compression_info: Dict[str, Any] = {}

    def set_system_prompt(self, content: str):
        """设置常驻 system prompt。"""
        self.system_prompt = content
        # 如果 messages 里已有 system，替换它
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def add(self, message: Dict[str, Any]):
        """添加一条消息。"""
        self.messages.append(message)

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取当前 messages。"""
        return self.messages

    def current_tokens(self) -> int:
        """当前 messages 估算 token 数。"""
        return estimate_messages_tokens(self.messages)

    def fit(self) -> Dict[str, Any]:
        """
        检查 token 是否超限，如果超限则执行压缩策略。
        返回压缩信息字典，供上层打印/记录。
        """
        current = self.current_tokens()
        self.last_compression_info = {
            "strategy": self.strategy,
            "before_tokens": current,
            "after_tokens": current,
            "compressed": False,
            "detail": "",
        }

        if current <= self.max_tokens:
            return self.last_compression_info

        # 需要压缩
        if self.strategy == "truncate":
            self._truncate()
        elif self.strategy == "summarize":
            self._summarize()
        elif self.strategy == "memory":
            self._dump_to_memory()

        self.last_compression_info["after_tokens"] = self.current_tokens()
        self.last_compression_info["compressed"] = True
        return self.last_compression_info

    # --------------------------------------------------------
    # 策略 1：截断
    # --------------------------------------------------------
    def _truncate(self):
        """
        丢弃最老的消息，直到 token 数低于阈值。
        永远保留 system prompt（第一条）。
        """
        system = []
        if self.messages and self.messages[0].get("role") == "system":
            system = [self.messages[0]]
            dynamic = self.messages[1:]
        else:
            dynamic = self.messages[:]

        # 保留 system 后，从最早一条动态消息开始删
        while dynamic and estimate_messages_tokens(system + dynamic) > self.max_tokens:
            removed = dynamic.pop(0)
            self.last_compression_info["detail"] += f"[截断] {removed.get('role')}\n"

        self.messages = system + dynamic

    # --------------------------------------------------------
    # 策略 2：摘要
    # --------------------------------------------------------
    def _summarize(self):
        """
        把早期对话发给模型生成摘要，然后用摘要消息替换原始消息。
        保留 system + 最近 2 轮对话 + 摘要。
        """
        system = []
        if self.messages and self.messages[0].get("role") == "system":
            system = [self.messages[0]]
            dynamic = self.messages[1:]
        else:
            dynamic = self.messages[:]

        # 最近 2 轮（4 条消息：user/assistant/tool... 大致保留末尾）不摘要
        keep_recent = 4
        to_summarize = dynamic[:-keep_recent] if len(dynamic) > keep_recent else []
        recent = dynamic[-keep_recent:] if len(dynamic) > keep_recent else dynamic[:]

        if not to_summarize:
            # 消息不多，直接截断兜底
            self._truncate()
            return

        summary_text = self._call_llm_for_summary(to_summarize)
        summary_message = {
            "role": "user",
            "content": (
                "以下是此前对话的摘要，请基于它继续回答后续问题：\n"
                f"{summary_text}"
            ),
        }

        self.messages = system + [summary_message] + recent
        self.last_compression_info["detail"] = (
            f"[摘要] 将 {len(to_summarize)} 条早期消息压缩为 1 条摘要"
        )

    def _call_llm_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """调用模型生成摘要。如果调用失败，回退为简单文本拼接。"""
        if not self.client or not self.url or not self.model:
            # 无模型可用，退化为拼接
            return self._fallback_summary(messages)

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "请用一段简短的中文总结以下对话中的关键事实和决策，"
                    "保留用户原始需求、已执行的关键步骤和结果。"
                    "不要添加没有依据的内容。"
                ),
            },
            {
                "role": "user",
                "content": "对话记录：\n" + json.dumps(messages, ensure_ascii=False, indent=2),
            },
        ]

        payload = {
            "model": self.model,
            "messages": prompt_messages,
            "temperature": 0.3,
        }

        try:
            resp = self.client.post(self.url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"].get("content", "").strip()
        except Exception as e:
            print(f"⚠️ 摘要生成失败，回退为简单拼接: {e}")
            return self._fallback_summary(messages)

    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        """摘要失败时的兜底：只保留 user 消息的前 200 字。"""
        parts = []
        for m in messages:
            if m.get("role") == "user":
                text = (m.get("content") or "")[:200]
                parts.append(f"用户曾问：{text}")
        return "\n".join(parts) or "（无可用摘要）"

    # --------------------------------------------------------
    # 策略 3：外置记忆
    # --------------------------------------------------------
    def _dump_to_memory(self):
        """
        把早期对话写入本地文件，messages 里只保留一条引用。
        保留 system + 最近 2 轮 + 引用。
        """
        system = []
        if self.messages and self.messages[0].get("role") == "system":
            system = [self.messages[0]]
            dynamic = self.messages[1:]
        else:
            dynamic = self.messages[:]

        keep_recent = 4
        to_store = dynamic[:-keep_recent] if len(dynamic) > keep_recent else []
        recent = dynamic[-keep_recent:] if len(dynamic) > keep_recent else dynamic[:]

        if not to_store:
            self._truncate()
            return

        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                f.write("# Lesson-04 外置记忆\n\n")
                for i, m in enumerate(to_store, 1):
                    f.write(f"## 消息 {i} [{m.get('role')}]\n\n")
                    f.write(json.dumps(m, ensure_ascii=False, indent=2))
                    f.write("\n\n")
        except Exception as e:
            print(f"⚠️ 外置记忆写入失败: {e}")
            self._truncate()
            return

        memory_message = {
            "role": "user",
            "content": (
                "早期对话已写入外置记忆文件，当前不再完整保留。"
                f"如需参考请读取：{self.memory_path}\n"
                "请继续基于已有信息和后续问题作答。"
            ),
        }

        self.messages = system + [memory_message] + recent
        self.last_compression_info["detail"] = (
            f"[外置记忆] 将 {len(to_store)} 条早期消息写入 {self.memory_path}"
        )
