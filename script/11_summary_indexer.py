"""层次摘要索引：document / section / paragraph 三层。

document 和 section 摘要不走 LLM 抽象概括（试过，术语和数字丢得厉害），
改用关键句抽取——按关键词命中和位置给句子打分，挑高分句原文保留下来，
这样模型名、数字这类硬事实不会在摘要环节丢掉。
paragraph 层直接放 chunk 原文，零损失。

三层都带 level/source 元数据写进同一个索引，路由时按层级取。
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR  # noqa: F401  (确保 output 目录存在)

CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "chunks"
SUMMARY_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ── 通用关键句抽取 ───────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_NUM_RE = re.compile(r"\b\d[\d,\.]*\b")
_UPPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9\-]+\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "at", "by", "from", "we", "our", "their", "they",
    "which", "such", "than", "then", "so", "if", "not", "no", "can", "will", "shall",
    "may", "might", "must", "have", "has", "had", "do", "does", "did", "about",
    "into", "over", "under", "more", "less", "most", "very", "also", "between",
    "through", "during", "each", "both", "all", "any", "some", "there", "here",
}


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) > 25]


def _extract_keywords(text: str, top_k: int = 40, min_word_len: int = 4) -> set:
    """关键词：大写起（专有名词）、数字、长词（非停用词）。

    顺手排除 4 位年份（引文里的 19xx/20xx 是噪声）；
    最短词长默认 4，是为了把 data / loss / GPUs 这类短术语也收进来。
    """
    words = [w for w in _WORD_RE.findall(text)]
    nums = [n for n in _NUM_RE.findall(text)
            if not _YEAR_RE.match(n) and len(n.replace(".", "").replace(",", "")) >= 2]
    upper = set(_UPPER_RE.findall(text))
    long_words = {w for w in words
                  if len(w) >= min_word_len and w.lower() not in _STOP}
    kw = set(w.lower() for w in (upper | long_words))
    for n in nums:
        kw.add(n)
    return kw


def _score_sentences(sentences: list[str], keywords: set) -> list[tuple[float, str]]:
    """给每个句子打分：关键词命中数 × 位置权重。"""
    scored = []
    n = len(sentences)
    for i, s in enumerate(sentences):
        sl = s.lower()
        hits = sum(1 for kw in keywords if kw in sl)
        # 数字命中权重更高（具体事实）
        num_hits = sum(1 for kw in keywords if re.fullmatch(r"[\d,\.]+", kw) and kw in sl)
        pos_w = 1.0
        if i == 0:
            pos_w = 1.3  # 首句常含定义
        elif i < 3:
            pos_w = 1.1
        elif i > n - 3 and n > 6:
            pos_w = 0.9
        length_norm = 1.0 / (1.0 + abs(len(s) - 180) / 180.0)
        score = (hits + 1.5 * num_hits) * pos_w * length_norm
        scored.append((score, s))
    return scored


def extractive_summary(text: str, max_sentences: int = 10) -> str:
    """关键句抽取式摘要：最大覆盖贪心 + 稀有事实补捞。

    先贪心挑「能覆盖最多未覆盖关键词」的句子；挑完后如果还有些数字 /
    专有名词没被任何选中句覆盖，就再补几句进去（预算外多加 30%）。
    这步是为了防止 194、XLM-RoBERTa 这种稀有的硬事实被漏掉——
    一开始没做这步，结果细节题经常答不上。
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text[:500]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    keywords = _extract_keywords(text)
    sent_kw = []
    for s in sentences:
        sl = s.lower()
        sent_kw.append({kw for kw in keywords if kw in sl})

    selected = []
    selected_set = set()
    covered = set()

    # 贪心最大覆盖
    while len(selected) < max_sentences:
        best_i, best_gain = -1, 0
        for i, cov in enumerate(sent_kw):
            if i in selected_set:
                continue
            gain = len(cov - covered)
            if gain > best_gain or (gain == best_gain and gain > 0 and
                                    (best_i < 0 or i < best_i)):
                best_gain, best_i = gain, i
        if best_i < 0 or best_gain == 0:
            break
        selected.append(best_i)
        selected_set.add(best_i)
        covered |= sent_kw[best_i]

    # 覆盖保证：未覆盖的重要关键词（数字或长专有名词）按覆盖数贪心补入
    important_uncovered = {kw for kw in (keywords - covered)
                          if re.fullmatch(r"[\d,\.]+", kw) or len(kw) >= 5}
    extra_cap = max_sentences + max(1, max_sentences // 3)
    while important_uncovered and len(selected) < extra_cap:
        best_i, best_hit = -1, 0
        for i, cov in enumerate(sent_kw):
            if i in selected_set:
                continue
            hit = len(cov & important_uncovered)
            if hit > best_hit or (hit == best_hit and hit > 0 and
                                  (best_i < 0 or i < best_i)):
                best_hit, best_i = hit, i
        if best_i < 0 or best_hit == 0:
            break
        selected.append(best_i)
        selected_set.add(best_i)
        important_uncovered -= sent_kw[best_i]

    return " ".join(sentences[i] for i in sorted(selected))


# ── 加载 chunks ─────────────────────────────────────────

_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-\s*\n?\s*(?=[A-Za-z])")


def _kw_in(kw: str, text: str) -> bool:
    """关键词命中判断：用 _normalize_text 即时归一化（处理跨行断字 en-\\ncoder
    与连字符差异 hybrid-attention vs hybridattention），两侧一致。"""
    from _eval_helpers import _normalize_text
    return _normalize_text(kw) in _normalize_text(text)


def _kw_score(text: str, q_kw: set) -> int:
    return sum(1 for kw in q_kw if _kw_in(kw, text))


def _load_chunks(source_name: str) -> list[dict]:
    """按 source（PDF 文件名）加载该文档的所有 chunk（保留原始连字符）。

    连字符统一保留，避免与 chromadb 里原始分块的文本表示不一致；
    需要处理 PDF 跨行断字（en-\\ncoder）时，匹配函数用 _normalize_text
    在比较时即时归一化（见 _kw_in / _extract_keywords 比较路径）。
    """
    p = CHUNKS_DIR / f"{source_name}.json"
    if not p.exists():
        p = CHUNKS_DIR / source_name
    if not p.exists():
        out = []
        for jp in sorted(CHUNKS_DIR.glob("*.json")):
            with open(jp, "r", encoding="utf-8") as f:
                for c in json.load(f):
                    if c.get("source") == source_name:
                        out.append(dict(c))
        return out
    with open(p, "r", encoding="utf-8") as f:
        return [dict(c) for c in json.load(f)]


# ── 三层索引构建 ─────────────────────────────────────────

def build_summary_index(source_name: str, save: bool = True) -> dict:
    """构建三层摘要索引。

    返回:
        {
          "source": str,
          "document": {"text": str, "section": "...", "page": int},
          "sections": [{"section": str, "text": str, "page": int, "n_chunks": int}, ...],
          "paragraphs": [原始 chunk, ...]  # 直接引用，零信息损失
        }
    """
    chunks = _load_chunks(source_name)
    if not chunks:
        return {"source": source_name, "document": {"text": "", "section": "", "page": 0},
                "sections": [], "paragraphs": []}

    # paragraph 层：chunk 原文，零损失
    paragraphs = chunks

    # section 层：按 section 分组，抽关键句
    by_section = defaultdict(list)
    for c in chunks:
        sec = c.get("section", "") or "(unsectioned)"
        by_section[sec].append(c)

    sections = []
    for sec, group in by_section.items():
        text = " ".join(g.get("text", "") for g in group)
        summary = extractive_summary(text, max_sentences=10)
        sections.append({
            "section": sec,
            "text": summary,
            "page": group[0].get("page", 0),
            "n_chunks": len(group),
        })
    # 按页码排序，章节顺序稳定
    sections.sort(key=lambda x: (x["page"], x["section"]))

    # document 层：以「各 section 摘要拼接」为输入再做抽取——section 摘要已经过
    # 覆盖保证（保留了各节的关键数字/专有名词），在此基础上抽全局摘要不会丢稀有事实。
    sec_join = " ".join(s["text"] for s in sections)
    doc_summary = extractive_summary(sec_join, max_sentences=18)
    if len(doc_summary) < 100:
        full_text = " ".join(c.get("text", "") for c in chunks)
        doc_summary = extractive_summary(full_text, max_sentences=25)

    index = {
        "_version": _SUMMARY_VERSION,
        "source": source_name,
        "document": {"text": doc_summary, "section": "Document Summary", "page": 0},
        "sections": sections,
        "paragraphs": paragraphs,
    }

    if save:
        out = SUMMARY_DIR / f"{source_name}.summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    return index


_SUMMARY_VERSION = "v6"  # 算法版本变了就 bump，自动重建（不删文件）


def load_summary_index(source_name: str) -> dict:
    """加载已构建的摘要索引；不存在或算法版本过期则重建（覆盖写，不删文件）。"""
    p = SUMMARY_DIR / f"{source_name}.summary.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                idx = json.load(f)
            if idx.get("_version") == _SUMMARY_VERSION:
                return idx
        except Exception:
            pass
    return build_summary_index(source_name, save=True)


# ── 按层级检索 ──────────────────────────────────────────

def retrieve_by_level(question: str, source_name: str, level: str,
                      top_k: int = 5) -> list[dict]:
    """从指定层级的摘要里检索与问题最相关的片段。

    level: document | section | paragraph
    - document: 返回整篇文档级摘要
    - section: 关键词重排各 section 摘要
    - paragraph: 调用向量检索（04）做语义召回并按 source 过滤，保证语义相关
      chunk 被找回（关键词检索会有语义鸿沟，向量检索补上）
    """
    idx = load_summary_index(source_name)

    if level == "document":
        return [{"text": idx["document"]["text"],
                 "section": idx["document"]["section"],
                 "page": idx["document"]["page"], "level": "document"}]

    if level == "section":
        q_kw = _extract_keywords(question)
        candidates = [{"text": s["text"], "section": s["section"],
                       "page": s["page"], "level": "section"} for s in idx["sections"]]

        def score(item):
            t_low = item["text"].lower()
            return sum(1 for kw in q_kw if kw in t_low)

        ranked = sorted(candidates, key=lambda x: -score(x))
        hit = [c for c in ranked if score(c) > 0]
        return (hit or ranked)[:top_k]

    # paragraph：向量 + 关键词 + 层次下钻 三路 RRF 融合，任一路命中都能上来
    q_kw = _extract_keywords(question)
    RRF_K = 60

    def kw_score(t):
        return _kw_score(t, q_kw)

    # 候选：text -> {meta..., rrf}
    cand = {}

    def _ensure(t, **meta):
        if t not in cand:
            cand[t] = {"text": t, "rrf": 0.0, **meta}
        return cand[t]

    # 路1：向量语义
    try:
        from importlib.machinery import SourceFileLoader as _SFL
        m04 = _SFL("m04", str(Path(__file__).resolve().parent / "04_embedder.py")).load_module()
        vec = m04.query(question, top_k=top_k + 3, where={"source": source_name})
        for rank, r in enumerate(vec, 1):
            t = r["text"]
            it = _ensure(t, section=r["metadata"].get("section", ""),
                         page=r["metadata"].get("page", 0), level="paragraph")
            it["rrf"] += 1.0 / (RRF_K + rank)
            it["_vec"] = True
    except Exception as e:
        print(f"  [信息] 向量检索不可用: {type(e).__name__}", file=sys.stderr)

    # 路2：层次下钻 —— section 摘要命中问题关键词 → 该 section 全部 chunk 进入
    matched_sections = set()
    for s in idx["sections"]:
        if kw_score(s["text"]) > 0:
            matched_sections.add(s["section"])
    drill_chunks = [c for c in idx["paragraphs"]
                    if c.get("section", "") in matched_sections]
    for rank, c in enumerate(
            sorted(drill_chunks, key=lambda c: -kw_score(c.get("text", ""))), 1):
        t = c.get("text", "")
        if not t:
            continue
        it = _ensure(t, section=c.get("section", ""), page=c.get("page", 0),
                     level="paragraph")
        it["rrf"] += 1.0 / (RRF_K + rank) + 0.001  # 层次定位加分（高精度信号）
        it["_drill"] = True

    # 路3：段落关键词命中
    kw_chunks = [c for c in idx["paragraphs"] if kw_score(c.get("text", "")) > 0]
    for rank, c in enumerate(
            sorted(kw_chunks, key=lambda c: -kw_score(c.get("text", ""))), 1):
        t = c.get("text", "")
        it = _ensure(t, section=c.get("section", ""), page=c.get("page", 0),
                     level="paragraph")
        it["rrf"] += 1.0 / (RRF_K + rank)
        it["_kw"] = True

    if cand:
        items = sorted(cand.values(), key=lambda x: -x["rrf"])
        return items[:max(top_k, 8)]

    # 全回退
    candidates = [{"text": c.get("text", ""), "section": c.get("section", ""),
                   "page": c.get("page", 0), "level": "paragraph"}
                  for c in idx["paragraphs"]]
    ranked = sorted(candidates, key=lambda x: -kw_score(x["text"]))
    return ranked[:top_k]


def retrieve_hierarchical(question: str, source_name: str,
                          top_k: int = 8) -> list[dict]:
    """跨层级检索：document + section + paragraph 三层 RRF 融合。

    层次摘要索引的核心查询入口：宏观事实（专有名词/定义）常出现在
    document/section 的抽取式摘要里；细节/数字在 paragraph 原文里。
    任一层命中都能回到答案，弥补单层（尤其纯向量段落检索）的语义鸿沟。
    """
    RRF_K = 60
    cand = {}

    def _ensure(t, level, section, page, **extra):
        if t not in cand:
            cand[t] = {"text": t, "level": level, "section": section,
                       "page": page, "rrf": 0.0, **extra}
        return cand[t]

    # document 层
    idx = load_summary_index(source_name)
    doc_t = idx["document"]["text"]
    if doc_t:
        _ensure(doc_t, "document", idx["document"]["section"],
                idx["document"]["page"])["rrf"] += 1.0 / (RRF_K + 1)

    q_kw = _extract_keywords(question)

    def kw_score(t):
        return _kw_score(t, q_kw)

    # section 层（按关键词命中排序）
    secs = sorted(idx["sections"], key=lambda s: -kw_score(s["text"]))
    for rank, s in enumerate(secs, 1):
        if not s["text"]:
            continue
        ks = kw_score(s["text"])
        if ks == 0 and rank > 5:
            continue
        it = _ensure(s["text"], "section", s["section"], s["page"])
        it["rrf"] += 1.0 / (RRF_K + rank) + 0.001 * ks

    # paragraph 层：复用 retrieve_by_level 的三路融合
    para = retrieve_by_level(question, source_name, "paragraph", top_k=top_k)
    for rank, r in enumerate(para, 1):
        it = _ensure(r["text"], "paragraph", r.get("section", ""), r.get("page", 0))
        it["rrf"] += 1.0 / (RRF_K + rank)

    return sorted(cand.values(), key=lambda x: -x["rrf"])[:top_k]


def get_all_text(source_name: str, level: str = "all") -> str:
    """取某层级的全部文本（用于 summary 覆盖性测试）。"""
    idx = load_summary_index(source_name)
    if level == "document":
        return idx["document"]["text"]
    if level == "section":
        return " ".join(s["text"] for s in idx["sections"])
    if level == "paragraph":
        return " ".join(c.get("text", "") for c in idx["paragraphs"])
    return (idx["document"]["text"] + " "
            + " ".join(s["text"] for s in idx["sections"]) + " "
            + " ".join(c.get("text", "") for c in idx["paragraphs"]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="层次摘要索引")
    ap.add_argument("source", help="PDF source 名（如 bge_paper.pdf）")
    ap.add_argument("--level", choices=["document", "section", "paragraph", "all"],
                    default="document")
    args = ap.parse_args()
    idx = build_summary_index(args.source)
    print(f"文档: {args.source}")
    print(f"  document 摘要: {len(idx['document']['text'])} 字")
    print(f"  sections: {len(idx['sections'])} 个")
    print(f"  paragraphs: {len(idx['paragraphs'])} 个")
    if args.level != "all":
        print(f"\n--- {args.level} ---")
        print(get_all_text(args.source, args.level)[:1000])
