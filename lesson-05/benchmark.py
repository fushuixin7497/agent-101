#!/usr/bin/env python3
"""
模块 5：Failure-First Design —— 自动验证 Benchmark
=====================================================
自动注入各种错误，验证 Harness 防护效果。

运行方式：
    export LLM_API_KEY="sk-xxxxx"
    export LLM_BASE_URL="https://api.moonshot.cn/v1"
    export LLM_MODEL="moonshot-v1-8k"
    uv run python benchmark.py
"""

import os
import sys
import json
import httpx
from typing import Dict, Any, List

from harness import AgentHarness
from context_manager import ContextManager, estimate_messages_tokens
from tools import get_tools, execute_tool, TOOL_REGISTRY
from injector import ErrorInjector


API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}
URL = f"{BASE_URL.rstrip('/')}/chat/completions"

SYSTEM_PROMPT = (
    "你是一个聪明的 Agent，可以调用工具帮用户完成任务。"
    "请一步一步思考，每次调用一个工具。"
    "如果工具返回错误，请仔细阅读错误信息，修正参数后再次尝试。"
)


# ============================================================
# 测试场景定义
# ============================================================
TEST_SCENARIOS = [
    {
        "name": "死循环检测",
        "injector_preset": None,
        "user_input": "搜索我的主目录",
        "expect_harness_action": "检测到死循环",
        "description": "连续 3 次相同调用应被检测并打断",
        "test_type": "direct",  # 直接测试 Harness 单元逻辑
    },
    {
        "name": "未知工具拦截",
        "injector_preset": "unknown_tool",
        "user_input": "用 flaky_tool 查询数据库",
        "expect_harness_action": "未知工具",
        "description": "未知工具调用应被拦截并返回可用列表",
        "test_type": "agent",
    },
    {
        "name": "参数越界修正",
        "injector_preset": None,
        "user_input": "读取文件 ~/test.txt limit=100",
        "expect_harness_action": "参数修正",
        "description": "limit=100 应被自动修正为 30",
        "test_type": "agent",
    },
    {
        "name": "部分失败恢复",
        "injector_preset": "partial_failure",
        "user_input": "读取文件 ~/test.txt",
        "expect_harness_action": "记录错误",
        "description": "3 个调用中 1 个失败应记录并继续",
        "test_type": "agent",
    },
    {
        "name": "限流重试",
        "injector_preset": "rate_limit_burst",
        "user_input": "用 flaky_tool 查询数据库",
        "expect_harness_action": "重试",
        "description": "429 错误应触发指数退避重试",
        "test_type": "agent",
    },
    {
        "name": "正常执行",
        "injector_preset": None,
        "user_input": "计算 365 * 24",
        "expect_harness_action": "正常完成",
        "description": "无错误注入时应正常完成",
        "test_type": "agent",
    },
]


def run_single_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """运行单个测试场景，返回结果报告。"""
    print(f"\n{'='*60}")
    print(f"🧪 测试场景: {scenario['name']}")
    print(f"   描述: {scenario['description']}")
    print(f"   输入: {scenario['user_input']}")
    print(f"   期望: {scenario['expect_harness_action']}")
    print("="*60)

    if not API_KEY:
        return {"status": "SKIP", "reason": "未设置 LLM_API_KEY"}

    # 死循环检测：直接测试 Harness 单元逻辑，不依赖模型行为
    if scenario.get("test_type") == "direct":
        return _test_dead_loop_direct(scenario)

    # 准备注入器
    injector = None
    if scenario["injector_preset"]:
        injector = ErrorInjector()
        getattr(injector, f"preset_{scenario['injector_preset']}")()
        print(f"   🧪 已注入: {scenario['injector_preset']}")

    # 初始化 Harness
    harness = AgentHarness(max_steps=10)
    client = httpx.Client(timeout=60.0)

    cm = ContextManager(
        max_tokens=2000,
        strategy="truncate",
        client=client,
        headers=HEADERS,
        url=URL,
        model=MODEL,
    )
    cm.set_system_prompt(SYSTEM_PROMPT)
    cm.add({"role": "user", "content": scenario["user_input"]})

    tools = get_tools()
    step = 0
    max_steps = 10
    harness_triggered = False
    harness_action = ""

    while step < max_steps:
        step += 1

        # 压缩 context
        cm.fit()
        messages = cm.get_messages()

        # 添加恢复提示
        recovery_hint = harness.get_recovery_hint()
        if recovery_hint:
            cm.add({"role": "user", "content": recovery_hint})
            messages = cm.get_messages()

        payload = {
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.3,
        }

        # API 调用
        try:
            resp = client.post(URL, headers=HEADERS, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {
                "status": "API_ERROR",
                "error": str(e),
                "harness_triggered": False,
            }

        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]

        if finish_reason == "stop":
            result = message.get("content", "")
            print(f"   ✅ 完成: {result[:100]}...")
            break

        if finish_reason == "tool_calls":
            tool_calls = message.get("tool_calls", [])

            # 死循环检测
            dead_loop = harness.check_dead_loop(step, tool_calls)
            if dead_loop:
                print(f"   🛑 Harness 触发: {dead_loop}")
                harness_triggered = True
                harness_action = "死循环检测"
                result = dead_loop
                break

            # 目标偏离检测
            drift = harness.check_drift(scenario["user_input"], tool_calls)
            if drift:
                print(f"   ⚠️ Harness 触发: {drift}")
                harness_triggered = True
                harness_action = "目标偏离警告"
                cm.add({"role": "user", "content": drift})

            cm.add(message)

            for tc in tool_calls:
                tc_id = tc["id"]
                func_name = tc["function"]["name"]
                func_args_json = tc["function"]["arguments"]

                # 未知工具拦截
                unknown = harness.intercept_unknown_tool(func_name)
                if unknown:
                    print(f"   ❌ Harness 拦截: {unknown}")
                    harness_triggered = True
                    harness_action = "未知工具拦截"
                    harness.record_partial_failure(func_name, unknown, step)
                    cm.add({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": func_name,
                        "content": unknown,
                    })
                    continue

                # 解析参数
                try:
                    args = json.loads(func_args_json)
                except json.JSONDecodeError as e:
                    error = f"参数解析失败: {e}"
                    harness.record_partial_failure(func_name, error, step)
                    cm.add({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": func_name,
                        "content": error,
                    })
                    continue

                # 参数修正
                intercept_msg, corrected_args = harness.intercept_invalid_args(func_name, args)
                if intercept_msg:
                    print(f"   ⚠️ Harness 修正: {intercept_msg}")
                    harness_triggered = True
                    harness_action = "参数修正"
                    args = corrected_args

                # 执行工具
                if injector and not injector.is_clean():
                    result = injector.run_with_injection(execute_tool, func_name, args)
                else:
                    result = execute_tool(func_name, args)

                # 记录错误
                if result.startswith("错误：") or result.startswith("❌"):
                    enriched = harness.enrich_error(result, func_name)
                    harness.record_partial_failure(func_name, enriched, step)
                    print(f"   ❌ 工具错误: {enriched[:100]}...")
                else:
                    print(f"   ✅ 工具成功: {func_name}")

                cm.add({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": func_name,
                    "content": str(result),
                })

            continue

        # 异常终止
        result = message.get("content", "")
        break

    else:
        result = "达到最大步数限制"

    client.close()

    # 生成报告
    status_report = harness.get_status_report()
    passed = (
        harness_triggered
        and scenario["expect_harness_action"] in harness_action
    ) or (
        not harness_triggered
        and scenario["expect_harness_action"] == "正常完成"
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "scenario": scenario["name"],
        "harness_triggered": harness_triggered,
        "harness_action": harness_action,
        "steps": step,
        "errors": status_report["total_errors"],
        "result": result[:200],
    }


def _test_dead_loop_direct(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    直接测试死循环检测 Harness 单元逻辑。
    
    不依赖模型行为，直接模拟模型连续调用相同工具的场景。
    """
    print("   🧪 直接测试 Harness 死循环检测逻辑（不调用模型）")
    
    harness = AgentHarness(max_steps=10, repeat_threshold=3)
    
    # 模拟连续 3 次相同的工具调用
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_files",
                "arguments": '{"directory": "~", "keyword": "."}'
            }
        }
    ]
    
    harness_triggered = False
    harness_action = ""
    result = ""
    
    for step in range(1, 5):
        print(f"   Step {step}: 模拟调用 search_files(directory='~', keyword='.')")
        dead_loop_msg = harness.check_dead_loop(step, tool_calls)
        if dead_loop_msg:
            print(f"   🛑 Harness 触发: {dead_loop_msg}")
            harness_triggered = True
            harness_action = "死循环检测"
            result = dead_loop_msg
            break
    
    if not harness_triggered:
        print("   ⚠️ 死循环检测未触发（异常）")
        result = "死循环检测未触发"
    
    passed = harness_triggered and "检测到死循环" in harness_action
    
    return {
        "status": "PASS" if passed else "FAIL",
        "scenario": scenario["name"],
        "harness_triggered": harness_triggered,
        "harness_action": harness_action,
        "steps": step if harness_triggered else 4,
        "errors": 0,
        "result": result[:200],
    }


def main():
    print("=" * 70)
    print("🛡️ 模块 5 Failure-First Design —— Harness 防护验证")
    print(f"   模型: {MODEL}")
    print("=" * 70)

    if not API_KEY:
        print("❌ 请先设置环境变量 LLM_API_KEY")
        sys.exit(1)

    results = []
    for scenario in TEST_SCENARIOS:
        result = run_single_scenario(scenario)
        results.append(result)

    # 汇总报告
    print("\n\n" + "=" * 70)
    print("📊 验证结果汇总")
    print("=" * 70)
    print(f"{'场景':<20} {'状态':<8} {'Harness':<12} {'步数':<6} {'错误数':<6}")
    print("-" * 70)

    passed = 0
    failed = 0
    for r in results:
        status_icon = "✅" if r["status"] == "PASS" else "❌"
        print(
            f"{r['scenario']:<20} {status_icon} {r['status']:<6} "
            f"{'是' if r['harness_triggered'] else '否':<10} "
            f"{r['steps']:<6} {r['errors']:<6}"
        )
        if r["status"] == "PASS":
            passed += 1
        else:
            failed += 1

    print("-" * 70)
    print(f"总计: {passed} 通过, {failed} 失败, {len(results)} 场景")

    if failed == 0:
        print("\n🎉 所有 Harness 防护机制验证通过！")
    else:
        print(f"\n⚠️ {failed} 个场景未通过，请检查 Harness 实现。")


if __name__ == "__main__":
    main()
