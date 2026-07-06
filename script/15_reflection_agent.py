"""反思 Agent：验证答案是否被检索片段支撑，输出忠实度分数与无依据片段。

核心方法（检索式验证，不用 LLM-as-judge，避免"同模型验证自己"）：
  1. 把答案拆成原子断言（句子）。
  2. 每条断言抽「关键证据词」：专有名词、数字、连字符术语（剥引文）。
  3. 对每条断言，去检索片段里查这些证据词的命中率：
       - 命中率 ≥ 阈值 → supported
       - 否则 → unsupported（标记为潜在幻觉）
  4. 忠实度 = supported 断言数 / 总断言数。
  5. 低忠实度（< REFLECTION_THRESHOLD）时，把无依据断言作为新查询建议返回，
     供编排 Agent 触发重检索。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m05 = SourceFileLoader("m05", str(Path(__file__).resolve().parent / "05_llm_client.py")).load_module()
from _eval_helpers import _normalize_text
from config import REFLECTION_THRESHOLD

_UPPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}\b")
_NUM_RE = re.compile(r"\b\d[\d,\.]*\b")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_HYPH_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")  # 含数字，避免 MiniMax-M1 被切成 MiniMax-M
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\*\-])")
_CITATION_RE = re.compile(
    r"\[[^\]]*\]|\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?[^)]*\)|"
    r"\b[A-Z][a-z]+\s+et\s+al\.?|\bPassage\s+\d+"
)
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "at", "by", "from", "we", "our", "their", "they",
    "which", "such", "than", "then", "so", "if", "not", "no", "can", "will", "shall",
    "may", "might", "must", "have", "has", "had", "do", "does", "did", "about",
    "into", "over", "under", "more", "less", "most", "very", "also", "between",
    "based", "using", "used", "use", "according", "passage", "key", "entities",
    "note", "however", "therefore", "thus", "since", "because",
}


def _split_claims(answer: str) -> list[str]:
    """把答案拆成原子断言（句子）。去掉「Supporting evidence」与「Key entities」附录行。

    这些附录是 14 按构造从 passages 逐字引用的，本身就被片段支撑，不应再
    进入忠实度校验（否则里面的公式/数字会被误判为幻觉，触发无谓重检索）。
    """
    answer = re.sub(r"\n\s*Key entities from passages:.*$", "", answer, flags=re.DOTALL)
    answer = re.sub(r"\n\s*Supporting evidence.*$", "", answer, flags=re.DOTALL)
    answer = re.sub(r"\n\s*Additional supporting details:.*$", "", answer, flags=re.DOTALL)
    answer = re.sub(r"\[[^\]]*\]", " ", answer)  # 去引用标 [Passage N]
    sents = [s.strip(" -/*\t") for s in _SENT_SPLIT.split(answer)]
    return [s for s in sents if len(s) > 15]


_MAGNITUDE = {"thousand", "million", "billion", "trillion", "dozen", "hundred"}


def _evidence_terms(text: str) -> list:
    """抽一条断言的「证据词」，按在原文首次出现顺序排列（首词通常是主语）。

    返回有序 list（主语在前），供共指宽容使用。
    """
    text = _CITATION_RE.sub(" ", text)
    low = text.lower()
    seen = {}
    # 专有名词 / 含数字连字符术语 / 全大写缩写，按出现位置记录
    for m in re.finditer(r"[A-Z][A-Za-z0-9\-]{2,}", text):
        w = m.group(0)
        wl = w.lower()
        if wl in _STOP or len(w) < 3:
            continue
        seen.setdefault(wl, m.start())
    for m in _HYPH_RE.finditer(text):
        w = m.group(0)
        if len(w) >= 4:
            seen.setdefault(w.lower(), m.start())
    for m in _NUM_RE.finditer(text):
        n = m.group(0)
        if not _YEAR_RE.match(n) and len(n.replace(".", "").replace(",", "")) >= 2:
            seen.setdefault(n, m.start())
    for mag in _MAGNITUDE:
        mm = re.search(r"\b" + mag + r"\b", low)
        if mm:
            seen.setdefault(mag, mm.start())
    # 按位置排序返回
    return [w for w, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def _passage_token_set(passages_text: str) -> set:
    """归一化后切成 token 集合（token 级匹配，避免 bert 命中 roberta 这种子串误判）。"""
    return set(_normalize_text(passages_text).split())


def _claim_supported(claim: str, passages_text: str,
                     threshold: float = 0.5) -> tuple[bool, float, list]:
    """判断断言是否被 passages 支撑。

    核心规则（对命名实体/数字从严，处理主语共指）：
      - 把 salient 证据词里「第一个」(通常是断言主语) 视为给定上下文，
        允许它在片段里以代词形式出现（共指消解的工程妥协）。
      - 剩余 salient 实体必须「全部」以 token 形式出现在 passages 里——
        任何一个找不到就判 unsupported（命名实体/数字/量级是幻觉主信号）。
      - 无 salient 证据词（纯泛述）或只剩主语 → supported（无法证伪）。
    返回 (supported, score, missing)。
    """
    terms = list(_evidence_terms(claim))
    if not terms:
        return True, 1.0, []
    p_tokens = _passage_token_set(passages_text)
    # 主语宽容：salient 词 ≥2 时，把第一个（通常是主语）视为给定上下文，允许
    # 它在片段里以代词出现；只有 1 个 salient 词时不能丢（否则无证据可判）。
    check_terms = terms[1:] if len(terms) >= 2 else terms
    if not check_terms:
        return True, 1.0, []
    missing = [t for t in check_terms if _normalize_text(t) not in p_tokens]
    hit = len(check_terms) - len(missing)
    score = hit / len(check_terms)
    supported = (len(missing) == 0)
    return supported, score, missing


def verify_answer(answer: str, passages: list[dict],
                  threshold: float = REFLECTION_THRESHOLD) -> dict:
    """反思主入口：返回忠实度评分与无依据断言。

    Returns:
        {faithfulness, n_claims, n_supported, unsupported_claims,
         needs_retry, threshold}
    """
    if not answer or not passages:
        return {"faithfulness": 1.0, "n_claims": 0, "n_supported": 0,
                "unsupported_claims": [], "needs_retry": False,
                "threshold": threshold}
    passages_text = " ".join(p.get("text", "") for p in passages)
    claims = _split_claims(answer)
    unsupported = []
    n_supported = 0
    for c in claims:
        ok, score, missing = _claim_supported(c, passages_text)
        if ok:
            n_supported += 1
        else:
            unsupported.append({"claim": c, "score": round(score, 3),
                                "missing": missing})
    faith = n_supported / len(claims) if claims else 1.0
    return {
        "faithfulness": round(faith, 3),
        "n_claims": len(claims),
        "n_supported": n_supported,
        "unsupported_claims": unsupported,
        "needs_retry": faith < threshold,
        "threshold": threshold,
    }


def is_claim_supported_by_passage(claim: str, passage: str,
                                  threshold: float = 0.5) -> bool:
    """单断言-单片段支撑判定（幻觉检测二分类用）。"""
    ok, _, _ = _claim_supported(claim, passage, threshold=threshold)
    return ok


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="反思 Agent")
    ap.add_argument("--answer", required=True)
    ap.add_argument("--passages-json", required=True)
    args = ap.parse_args()
    passages = json.load(open(args.passages_json))
    r = verify_answer(args.answer, passages)
    print(f"忠实度: {r['faithfulness']} ({r['n_supported']}/{r['n_claims']})")
    print(f"需重检索: {r['needs_retry']}")
    for u in r["unsupported_claims"]:
        print(f"  ❌ {u['claim'][:60]} (缺: {u['missing'][:5]})")
