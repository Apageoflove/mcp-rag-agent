"""共享的纯文本小工具（不 import 任何项目模块，方便到处复用 + 单测）。"""
import re
from pathlib import Path


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


def chunk_filename_to_pdf(filename: str) -> str:
    """把分块文件名还原成 PDF 名。

    分块 JSON 命名约定是「PDF 名 + .json」（如 bge_paper.pdf.json），
    所以去掉末尾的 .json 就行。原先用 stem.replace('.pdf','')+'.pdf'
    能跑但脆弱——它假设所有 chunk 文件名都含 .pdf，不含的就错加一个。
    直接处理双扩展名更稳：剥一层 .json 后如果还以 .pdf 结尾就保留，否则原样。
    """
    name = filename
    # 剥掉末尾的 .json（分块文件的统一扩展名）
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return name
