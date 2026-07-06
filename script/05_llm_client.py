"""
05 llm_client: MiniMax-M3 的 API 调用封装（OpenAI 兼容格式）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def create_client():
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat(messages, model=None, temperature=0.7, max_tokens=2000):
    """原生 chat completion，传入完整 messages 列表"""
    client = create_client()
    resp = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def simple_chat(user_message, system_message=None):
    """单轮问答的快捷方法"""
    msgs = []
    if system_message:
        msgs.append({"role": "system", "content": system_message})
    msgs.append({"role": "user", "content": user_message})
    return chat(msgs)


if __name__ == "__main__":
    r = simple_chat("你好，请用一句话介绍你自己")
    print(f"模型: {LLM_MODEL}")
    print(f"回复: {r}")
