"""路由 Agent：看问题长啥样，决定走哪条检索路径。

输出：type（summary/relation/fact）、level（检索层级）、tools、max_hops。

两套规则并行：先用正则匹配问题里的意图词（summary/quantity/relation/fact），
快、不要钱；规则拿不准（落到 fact 默认分支、置信度低）再请 M3 仲裁一次。
正则那部分只锚定词首不锚定词尾，这样 summarize/summarizes、use/used/uses
都能命中。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m05 = SourceFileLoader("m05", str(Path(__file__).resolve().parent / "05_llm_client.py")).load_module()

# ── 意图词正则 ──
# 只锚定词首不锚定词尾，这样能匹配词缀变化
# （summar→Summarize、develop→developed、author→authors、use→used/uses）。
_SUMMARY_RE = re.compile(
    r"\b(summar|overview|main idea|main contribution|abstract|key point|"
    r"big picture|give me .* about (?:the )?paper|describe the paper|tl;?dr)",
    re.IGNORECASE,
)
_QUANTITY_RE = re.compile(
    r"\b(how many|how much|how long|how big|number of|amount of|size of|"
    r"cost of|total of|count of)",
    re.IGNORECASE,
)
_RELATION_RE = re.compile(
    r"\b(relation|relationship|compare|comparison|versus|\bvs\.?\b|differ(?:ence)?|"
    r"between|based on|outperform|use[sd]?|utilize|employ|propos|develop|creat|"
    r"author|consist of|derive from|built on| powered by)",
    re.IGNORECASE,
)
# 图片类问题（多模态路由用）
_IMAGE_RE = re.compile(
    r"\b(figure|chart|diagram|image|picture|plot|table|graph)\b",
    re.IGNORECASE,
)


def _rule_classify(question: str) -> tuple[str, float]:
    """规则判定问题类型，返回 (type, confidence)。

    优先级：summary > quantity(fact) > relation > fact。
    confidence: summary/quantity/relation 命中=0.95（规则明确），fact 默认=0.5。
    """
    if _SUMMARY_RE.search(question):
        return "summary", 0.95
    if _QUANTITY_RE.search(question):
        return "fact", 0.95
    if _RELATION_RE.search(question):
        return "relation", 0.95
    return "fact", 0.5


def _llm_classify(question: str) -> tuple[str, float]:
    """LLM 二次路由（规则置信度低时用）。返回 (type, confidence)。"""
    prompt = (
        "Classify the user's question into exactly one of:\n"
        "- summary: asks to summarize / overview / give main idea of a paper/topic\n"
        "- relation: asks about a relationship/comparison between entities, OR what X uses/is-based-on/outperforms, OR who proposed/developed/authored X\n"
        "- fact: asks for a specific fact, number, definition, or detail\n\n"
        f"Question: {question}\n\n"
        'Reply ONLY in JSON: {{"type": "...", "confidence": 0.0-1.0}}'
    )
    try:
        resp = m05.simple_chat(prompt)
        m = re.search(r'\{[^}]*"type"[^}]*\}', resp, re.DOTALL)
        if not m:
            return "fact", 0.3
        import json as _json
        obj = _json.loads(m.group(0))
        t = str(obj.get("type", "fact")).strip().lower()
        c = float(obj.get("confidence", 0.5))
        if t not in ("summary", "relation", "fact"):
            t = "fact"
        return t, max(0.0, min(1.0, c))
    except Exception as e:
        print(f"  [信息] LLM 路由失败: {type(e).__name__}", file=sys.stderr)
        return "fact", 0.3


def _level_and_tools(qtype: str, question: str) -> tuple[str, list[str], int]:
    """根据类型 + 问题特征，定检索层级/工具/跳数。"""
    if qtype == "summary":
        return "document", ["summary_index", "vector"], 1
    if qtype == "relation":
        # 关系类：图谱为主，向量兜底
        return "section", ["graph", "vector"], 3
    # fact：段落级向量；若含图片词，加 VLM
    tools = ["vector", "summary_index"]
    if _IMAGE_RE.search(question):
        tools.append("vlm")
    return "paragraph", tools, 1


def route(question: str, use_llm_fallback: bool = True) -> dict:
    """路由主入口。

    Returns:
        {question, type, level, tools, max_hops, confidence, source}
        source: rule | llm | rule+llm
    """
    rtype, rc = _rule_classify(question)
    source = "rule"

    # 规则置信度低 → LLM 仲裁
    if rc < 0.9 and use_llm_fallback:
        ltype, lc = _llm_classify(question)
        if lc > rc + 0.1:  # LLM 明显更确信
            rtype, rc = ltype, lc
            source = "llm"
        elif ltype != rtype and lc >= rc:
            # 平局但 LLM 有不同意见，记为融合
            source = "rule+llm"

    level, tools, max_hops = _level_and_tools(rtype, question)
    return {
        "question": question,
        "type": rtype,
        "level": level,
        "tools": tools,
        "max_hops": max_hops,
        "confidence": round(rc, 3),
        "source": source,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="路由 Agent")
    ap.add_argument("question", nargs="*", help="问题")
    ap.add_argument("--no-llm", action="store_true", help="禁用 LLM 仲裁")
    args = ap.parse_args()
    if not args.question:
        print("用法: python3 12_router_agent.py 你的问题")
        sys.exit(1)
    q = " ".join(args.question)
    r = route(q, use_llm_fallback=not args.no_llm)
    print(f"问题: {q}")
    print(f"类型: {r['type']}  层级: {r['level']}  工具: {r['tools']}  "
          f"跳数: {r['max_hops']}  置信度: {r['confidence']}  来源: {r['source']}")
