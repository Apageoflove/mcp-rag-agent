"""Agent 编排器：串起 路由 → 检索 → 推理 → 反思 整条流水线。

流程：
  1. 路由 Agent：判定问题类型 / 层级 / 工具 / 跳数。
  2. 检索 Agent：按策略多路召回 + MMR 去重。
  3. 推理 Agent：基于片段生成 grounded 答案（自一致性采样）。
  4. 反思 Agent：检索式验证答案忠实度，给出置信度。
  5. 输出：答案 + 来源 + 置信度 + 完整推理链路。

反思原本会触发定向重检索，后来发现重检索会把首轮已找回的关键片段替掉、
完备性反而回退，就改成只打分了（完备性靠推理阶段的逐字引用兜）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m12 = SourceFileLoader("m12", str(Path(__file__).resolve().parent / "12_router_agent.py")).load_module()
m13 = SourceFileLoader("m13", str(Path(__file__).resolve().parent / "13_retriever_agent.py")).load_module()
m14 = SourceFileLoader("m14", str(Path(__file__).resolve().parent / "14_reasoning_agent.py")).load_module()
m15 = SourceFileLoader("m15", str(Path(__file__).resolve().parent / "15_reflection_agent.py")).load_module()
from config import REFLECTION_THRESHOLD, MAX_RETRY


def _merge_passages(old: list[dict], new: list[dict]) -> list[dict]:
    """合并新旧片段，按 doc_id 去重（保留 score 高的）。"""
    from _eval_helpers import _normalize_text

    def doc_id(item):
        meta = item.get("metadata", item)
        src = meta.get("source") or ""
        page = meta.get("page", "")
        cidx = meta.get("chunk_index", "")
        if src and page != "":
            return f"{src}::p{page}::c{cidx}"
        return _normalize_text(item.get("text", ""))[:80]

    merged = {}
    for c in old + new:
        cid = doc_id(c)
        if cid not in merged or c.get("score", 0) > merged[cid].get("score", 0):
            merged[cid] = c
    return list(merged.values())


def _retry_queries(unsupported: list[dict]) -> list[str]:
    """从无依据断言里抽重检索 query（取断言 + 缺失证据词拼接）。"""
    queries = []
    for u in unsupported[:2]:  # 最多 2 条定向 query
        claim = u.get("claim", "")
        missing = u.get("missing", [])
        if missing:
            queries.append(claim)  # 断言本身常是最好的 query
    return queries


def answer(question: str, source_filter: str = None,
           max_retry: int = MAX_RETRY, top_k: int = 5,
           use_llm_route: bool = False) -> dict:
    """端到端问答。

    Returns:
        {question, answer, confidence, source, passages, route, trace, retries}
    """
    t0 = time.time()
    trace = []

    # 1. 路由
    route_info = m12.route(question, use_llm_fallback=use_llm_route)
    trace.append({"agent": "router", "result": route_info})

    # 2. 检索
    passages = m13.retrieve(question, strategy=route_info, top_k=top_k,
                            source_filter=source_filter)
    trace.append({"agent": "retriever", "n_passages": len(passages)})

    if not passages:
        return {
            "question": question,
            "answer": "Based on the provided passages, I cannot answer this question.",
            "confidence": 0.0, "source": "empty", "passages": [],
            "route": route_info, "trace": trace, "retries": 0,
            "time": round(time.time() - t0, 2),
        }

    # 3. 推理：自一致性（多次采样 + 取最优 + Supporting evidence 逐字引用）
    reason_result = m14.reason_with_self_consistency(question, passages, n=3,
                                                     source_filter=source_filter)
    answer_text = reason_result["answer"]
    trace.append({"agent": "reasoner(self-consistency)",
                  "n_passes": reason_result.get("n_passes", 1),
                  "grounded": reason_result["grounded"]})

    # 4. 反思（检索式验证忠实度，仅用于报告置信度；不再触发重检索）。
    # 重检索会替换掉首轮已找回的关键片段（导致完备性回退），而完备性已由
    # Supporting evidence 的逐字引用保证，故反思这里只打分。
    reflection = m15.verify_answer(answer_text, passages)
    trace.append({"agent": "reflection",
                  "faithfulness": reflection["faithfulness"],
                  "n_unsupported": len(reflection["unsupported_claims"])})
    retries = 0

    confidence = reflection["faithfulness"]
    return {
        "question": question,
        "answer": answer_text,
        "confidence": round(confidence, 3),
        "source": "graph" if "graph" in route_info["tools"] else "vector",
        "passages": passages,
        "route": route_info,
        "trace": trace,
        "retries": retries,
        "time": round(time.time() - t0, 2),
    }


def answer_tool(question: str, top_k: int = 5) -> str:
    """MCP 工具接口：端到端问答。"""
    import json
    r = answer(question, top_k=top_k)
    return json.dumps({
        "answer": r["answer"],
        "confidence": r["confidence"],
        "source": r["source"],
        "n_passages": len(r["passages"]),
        "retries": r["retries"],
        "time": r["time"],
        "route": {"type": r["route"]["type"], "tools": r["route"]["tools"]},
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Agent 编排器")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--source", help="限定 PDF source")
    args = ap.parse_args()
    if not args.question:
        print("用法: python3 16_agent_orchestrator.py 你的问题 [--source bge_paper.pdf]")
        sys.exit(1)
    q = " ".join(args.question)
    r = answer(q, source_filter=args.source)
    print(f"=== Answer (confidence={r['confidence']}, retries={r['retries']}, "
          f"time={r['time']}s) ===")
    print(r["answer"])
    print(f"\n--- 推理链 ({len(r['trace'])} 步) ---")
    for step in r["trace"]:
        print(f"  [{step['agent']}] {step}")
