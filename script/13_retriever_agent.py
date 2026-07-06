"""检索 Agent：照路由策略做多跳混合检索，再用 MMR 去重。

按路由给的 tools 决定启用哪几路：向量(06) / 图谱(10) / 层次摘要(11)。
关系类问题走图谱做多跳，路径断了就向量兜底；几路结果用 RRF 融合；
最后 MMR 在相关性和多样性之间折中一下，免得 top-k 全是差不多的 chunk。
"""

# 多路召回 + MMR + cross-encoder rerank

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m06 = SourceFileLoader("m06", str(Path(__file__).resolve().parent / "06_rag_query.py")).load_module()
m10 = SourceFileLoader("m10", str(Path(__file__).resolve().parent / "10_kg_query.py")).load_module()
m11 = SourceFileLoader("m11", str(Path(__file__).resolve().parent / "11_summary_indexer.py")).load_module()
m12 = SourceFileLoader("m12", str(Path(__file__).resolve().parent / "12_router_agent.py")).load_module()
from _eval_helpers import _normalize_text  # 复用统一归一化


def _doc_id(item: dict) -> str:
    """用 (source, page, chunk_index) 或文本前 80 字做去重 key。"""
    meta = item.get("metadata", item)
    src = meta.get("source") or meta.get("pdf") or ""
    page = meta.get("page", "")
    cidx = meta.get("chunk_index", "")
    if src and page != "":
        return f"{src}::p{page}::c{cidx}"
    return _normalize_text(item.get("text", ""))[:80]


def mmr_rerank(candidates: list[dict], query: str, top_k: int = 5,
               lambda_: float = 0.6) -> list[dict]:
    """Maximal Marginal Relevance：在相关性与多样性之间折中选 top_k。

    score(d) = λ * rel(d) - (1-λ) * max_{d' in S} sim(d, d')
      - rel(d)：与 query 的 token 重叠（通用、跨工具一致，避免不同工具的分数
        量纲不一致导致偏向某一路）。
      - sim：候选间 token-Jaccard（多样性，避免近似重复 chunk）。
    """
    if not candidates:
        return []

    def tokens(t: str) -> set:
        return set(_normalize_text(t).split())

    q_tokens = tokens(query)
    cand_tokens = [tokens(c.get("text", "")) for c in candidates]

    def rel(i: int) -> float:
        if not q_tokens:
            return 0.0
        return len(cand_tokens[i] & q_tokens) / len(q_tokens)

    def pair_sim(i: int, j: int) -> float:
        uni = cand_tokens[i] | cand_tokens[j]
        if not uni:
            return 0.0
        return len(cand_tokens[i] & cand_tokens[j]) / len(uni)

    selected = []
    remaining = list(range(len(candidates)))
    if remaining:
        first = max(remaining, key=lambda i: rel(i))
        selected.append(first)
        remaining.remove(first)

    while len(selected) < top_k and remaining:
        best_i, best_score = -1, -1e9
        for i in remaining:
            diversity = max((pair_sim(i, j) for j in selected), default=0.0)
            score = lambda_ * rel(i) - (1 - lambda_) * diversity
            if score > best_score:
                best_score, best_i = score, i
        if best_i < 0:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


def _retrieve_vector(query: str, top_k: int, source_filter: str = None) -> list[dict]:
    """向量+BM25 混合检索。给定 source_filter 时限定来源。

    返回项的 score 为向量相似度（语义相关度），保留下来供 MMR 保底用。
    BM25 结果作为补充（不覆盖向量 score）。
    """
    try:
        if source_filter:
            m04 = SourceFileLoader(
                "m04", str(Path(__file__).resolve().parent / "04_embedder.py")
            ).load_module()
            where = {"source": source_filter}
            vec = m04.query(query, top_k=top_k, where=where)  # score = 1 - cos dist
            # BM25 补充：把向量没召回的同来源 chunk 加进来（不覆盖向量分）
            try:
                bm = m06.get_bm25_index().search(query, top_k=top_k)
                seen = {_doc_id_like(r) for r in vec}
                for r in bm:
                    if r.get("metadata", {}).get("source") != source_filter:
                        continue
                    key = _doc_id_like(r)
                    if key in seen:
                        continue
                    seen.add(key)
                    r["score"] = 0.3  # BM25 兜底分（低于向量分）
                    vec.append(r)
            except Exception:
                pass
            return vec
        return m06.retrieve(query, top_k=top_k, use_bm25=True,
                            use_hyde=True, use_rerank=False)
    except Exception as e:
        print(f"  [信息] 向量检索失败: {type(e).__name__}", file=sys.stderr)
        return []


def _doc_id_like(item: dict) -> str:
    meta = item.get("metadata", item)
    src = meta.get("source") or ""
    page = meta.get("page", "")
    cidx = meta.get("chunk_index", "")
    if src and page != "":
        return f"{src}::p{page}::c{cidx}"
    return _normalize_text(item.get("text", ""))[:80]


def _retrieve_graph(query: str, max_hops: int) -> list[dict]:
    """图谱检索：返回三元组路径文本块。"""
    try:
        result = m10.graph_search(query, fallback_to_vector=False)
        if result.get("source") != "graph":
            return []
        # 把 graph_results 拍平成文本片段
        out = []
        for rec in result.get("graph_results", [])[:top_k_default()]:
            txt = ", ".join(f"{k}={v}" for k, v in rec.items()
                            if k not in ("source_page", "source_section"))
            out.append({
                "text": txt,
                "metadata": {
                    "source": "knowledge_graph",
                    "page": rec.get("source_page", 0),
                    "section": "graph",
                },
                "score": 0.9,
                "source_tool": "graph",
            })
        return out
    except Exception as e:
        print(f"  [信息] 图谱检索失败: {type(e).__name__}", file=sys.stderr)
        return []


def _retrieve_summary(query: str, level: str, source_filter: str = None) -> list[dict]:
    try:
        if source_filter:
            items = m11.retrieve_hierarchical(query, source_filter, top_k=8)
        else:
            items = m11.retrieve_hierarchical(query, _any_source(), top_k=8)
        return [{
            "text": it.get("text", ""),
            "metadata": {
                "source": source_filter or _any_source(),
                "page": it.get("page", 0),
                "section": it.get("section", ""),
                "level": it.get("level", ""),
            },
            "score": it.get("rrf", 0.5),
            "source_tool": "summary",
        } for it in items]
    except Exception as e:
        print(f"  [信息] 摘要检索失败: {type(e).__name__}", file=sys.stderr)
        return []


def top_k_default() -> int:
    return 5


def _any_source() -> str:
    """没有 source 约束时的默认文档（取第一份可用摘要索引）。"""
    sd = Path(__file__).resolve().parent.parent.parent / "output" / "summaries"
    if sd.exists():
        files = sorted(sd.glob("*.summary.json"))
        if files:
            return files[0].stem.replace(".summary", "")
    return "bge_paper.pdf"


def retrieve(question: str, strategy: dict = None, top_k: int = 5,
             source_filter: str = None) -> list[dict]:
    """检索 Agent 主入口。

    Args:
        question: 自然语言问题
        strategy: 路由 Agent 的输出（含 tools/level/max_hops）；None 则自动路由
        top_k: 最终返回条数
        source_filter: 限定 PDF（可选）
    """
    if strategy is None:
        strategy = m12.route(question, use_llm_fallback=False)
    tools = strategy.get("tools", ["vector"])
    max_hops = strategy.get("max_hops", 1)

    candidates = []
    vec_items = []
    if "vector" in tools:
        vec_items = _retrieve_vector(question, top_k=max(top_k * 2, 10),
                                     source_filter=source_filter)
        candidates.extend(vec_items)
    if "graph" in tools:
        candidates.extend(_retrieve_graph(question, max_hops))
    if "summary_index" in tools:
        candidates.extend(_retrieve_summary(question, strategy.get("level"),
                                            source_filter))

    # 文档级摘要保底：11 的 document 摘要是个全覆盖概览，把它当候选塞进来，
    # 免得向量/关键词检索因为语义鸿沟把含答案的术语漏掉
    # （比如 authors→机构名、train→softmax 这种词面对不上的情况）。
    if source_filter:
        try:
            idx = m11.load_summary_index(source_filter)
            doc_text = idx["document"]["text"]
            if doc_text and doc_text.strip():
                candidates.append({
                    "text": doc_text,
                    "metadata": {"source": source_filter, "page": 0,
                                 "section": "DocumentSummary", "level": "document"},
                    "score": 0.4, "source_tool": "summary_doc",
                })
        except Exception as e:
            print(f"  [信息] 文档摘要保底失败: {type(e).__name__}", file=sys.stderr)

    # 去重（同 doc_id 取 score 最高的）
    dedup = {}
    for c in candidates:
        cid = _doc_id(c)
        if cid not in dedup or c.get("score", 0) > dedup[cid].get("score", 0):
            dedup[cid] = c
    candidates = list(dedup.values())

    # MMR 多样性重排
    final = mmr_rerank(candidates, question, top_k=max(top_k * 3, 15), lambda_=0.65)

    # cross-encoder 重排序（bge-reranker-v2-m3）：
    # MMR 出的候选再过一遍 (query, passage) 交叉注意力精排，比双塔向量准。
    try:
        from _reranker import rerank
        final = rerank(question, final, top_k=top_k)
    except Exception as e:
        print(f"  [信息] cross-encoder 重排序失败，回退 MMR: {type(e).__name__}",
              file=sys.stderr)
        final = final[:top_k]

    # 语义保底：向量检索的最相关 chunk（语义分最高）必须出现在最终结果里，
    # 避免关键词重排把语义最相关但词面不同的 chunk（如作者列表）挤出去。
    if vec_items:
        top_vec = max(vec_items, key=lambda c: c.get("score", 0))
        top_vec_id = _doc_id(top_vec)
        if not any(_doc_id(f) == top_vec_id for f in final):
            if final:
                final[-1] = top_vec
            else:
                final = [top_vec]

    # 文档摘要保底：document 摘要覆盖论文全部关键术语，必须保留在最终结果里
    # （MMR 可能因多样性把它挤掉）。它在答案完备性上是"安全网"。
    doc_summ = next((c for c in candidates
                     if c.get("metadata", {}).get("section") == "DocumentSummary"), None)
    if doc_summ and not any(_doc_id(f) == _doc_id(doc_summ) for f in final):
        if len(final) >= top_k:
            final[-1] = doc_summ
        else:
            final.append(doc_summ)
    return final


def retrieve_tool(question: str, top_k: int = 5) -> str:
    """MCP 工具接口：多跳混合检索。"""
    import json
    results = retrieve(question, top_k=top_k)
    simplified = []
    for r in results:
        meta = r.get("metadata", {})
        simplified.append({
            "text": r.get("text", "")[:500],
            "source": meta.get("source", ""),
            "page": meta.get("page", ""),
            "section": meta.get("section", "")[:40],
            "score": round(r.get("score", 0), 4),
            "tool": r.get("source_tool", ""),
        })
    return json.dumps(simplified, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="检索 Agent")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    if not args.question:
        print("用法: python3 13_retriever_agent.py 你的问题")
        sys.exit(1)
    q = " ".join(args.question)
    strat = m12.route(q, use_llm_fallback=False)
    print(f"路由: type={strat['type']} tools={strat['tools']} hops={strat['max_hops']}")
    results = retrieve(q, strategy=strat, top_k=args.top_k)
    print(f"\n检索到 {len(results)} 条:")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"  [{i}] [{meta.get('source','')[:25]}] p{meta.get('page','')} "
              f"tool={r.get('source_tool','')} | {r.get('text','')[:100]}")
