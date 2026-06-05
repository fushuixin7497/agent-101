#!/usr/bin/env python3
"""
raw_call.py —— 解剖一次 LLM API 调用的最小结构
===============================================
这个脚本只做一件事：发一次非流式请求，然后把
【HTTP 请求体】和【HTTP 响应体】完整打印出来，
让你对 LLM API 的 JSON 结构建立直观认识。

运行方式：
    export LLM_API_KEY="your-api-key"
    uv run python raw_call.py
"""

import os
import sys
import json
import httpx

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not API_KEY:
    print("❌ 请先设置环境变量 LLM_API_KEY")
    sys.exit(1)

# -------------------------------------------------
# 1) 构造请求体（这就是你通过网络发送的 JSON）
# -------------------------------------------------
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "中国有多少平方公里？"},
    ],
    "stream": False,
    "temperature": 0.7,
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

url = f"{BASE_URL.rstrip('/')}/chat/completions"

print("=" * 60)
print("📤 HTTP 请求体（Request Body）")
print("=" * 60)
print(json.dumps(payload, indent=2, ensure_ascii=False))
print()

# -------------------------------------------------
# 2) 发送请求
# -------------------------------------------------
with httpx.Client(timeout=60.0) as client:
    resp = client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()

# -------------------------------------------------
# 3) 打印原始响应（这就是服务器返回的 JSON）
# -------------------------------------------------
print("=" * 60)
print("📥 HTTP 响应体（Response Body）")
print("=" * 60)
print(json.dumps(data, indent=2, ensure_ascii=False))
print()

# -------------------------------------------------
# 4) 提取关键字段并解读
# -------------------------------------------------
choice = data["choices"][0]
usage = data.get("usage", {})

print("=" * 60)
print("🔍 关键字段解读")
print("=" * 60)
print(f"id              : {data.get('id')}")
print(f"model           : {data.get('model')}")
print(f"finish_reason   : {choice.get('finish_reason')}")
print(f"assistant_reply : {choice['message']['content']}")
print()
print("【Token 消耗明细】")
print(f"  prompt_tokens     (输入)  : {usage.get('prompt_tokens', 0)}")
print(f"  completion_tokens (输出)  : {usage.get('completion_tokens', 0)}")
print(f"  total_tokens      (总计)  : {usage.get('total_tokens', 0)}")
print()
print("💡 计费公式 ≈ 输入单价 × prompt_tokens + 输出单价 × completion_tokens")
