"""共享的纯文本小工具（不 import 任何项目模块，方便到处复用 + 单测）。"""
import re


def strip_think(text: str) -> str:
    """剥掉 LLM（MiniMax-M3）的 <think>...</think> 思维链块，保留最终答案。

    之所以单独拎出来：M3 默认会在答案前输出一段 <think> 推理，有时会吃光
    max_tokens 预算、把真正答案挤没；更麻烦的是被 max_tokens 截断时会留下一个
    **没闭合**的裸 <think> 一直写到结尾，只处理闭合标签的话整段思维链会漏进
    答案 / Cypher 里。所以这里两种都剥：

    1. 闭合的 <think>...</think>
    2. 裸 <think> 到结尾（截断情形）
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()
