"""评估框架：RAGAS 那套指标的简化实现。

四个维度：
  1. Context Recall：金标准答案的每条事实是否被检索上下文覆盖。
  2. Faithfulness：答案的每条断言是否被检索上下文支撑（复用 15 的检索式验证）。
  3. Answer Relevancy：问题-答案的 embedding cosine 相似度。
  4. Answer Completeness：金标准关键术语在答案里的命中率。

DeepEval 试过，但它和 MiniMax-M3 的 <think> 输出格式对不上（JSON 解析老失败），
干脆按 RAGAS 公开定义自己实现一遍，至少能跑、能复现。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m16 = SourceFileLoader("m16", str(Path(__file__).resolve().parent / "16_agent_orchestrator.py")).load_module()
m15 = SourceFileLoader("m15", str(Path(__file__).resolve().parent / "15_reflection_agent.py")).load_module()
from _eval_helpers import keyword_hit_ratio


def answer_relevancy(question: str, answer: str) -> float:
    """RAGAS Answer Relevancy：问题与答案的语义相似度（embedding cosine）。

    RAGAS 原始定义是从答案反生成问题再算相似度；这里用直接 embedding cosine
    作为等价近似（bge-m3），值域 0~1，越高越相关。
    """
    try:
        from importlib.machinery import SourceFileLoader as _SFL
        m04 = _SFL("m04", str(Path(__file__).resolve().parent / "04_embedder.py")).load_module()
        qv = m04._embed([question], is_query=True)[0]
        av = m04._embed([answer[:1000]], is_query=False)[0]
        return float(sum(a * b for a, b in zip(qv, av)))
    except Exception:
        return 0.0

DATASET = Path(__file__).resolve().parent / "18_eval_dataset.json"


def load_dataset() -> list[dict]:
    with open(DATASET, "r", encoding="utf-8") as f:
        return json.load(f)["qa"]


def evaluate_one(qa: dict, source_filter: str = None,
                 runner=None) -> dict:
    """跑单题，返回各维度指标。"""
    runner = runner or m16.answer
    t0 = time.time()
    result = runner(qa["question"], source_filter=source_filter, top_k=5)
    latency = round(time.time() - t0, 2)

    passages = result.get("passages", [])
    passages_text = " ".join(p.get("text", "") for p in passages)

    # 检索召回：answer_keys 在检索片段里的命中率
    r_hit, r_total, _ = keyword_hit_ratio(qa.get("answer_keys", []), passages_text)
    retrieval_recall = r_hit / r_total if r_total else 1.0

    # 忠实度：用 15 的检索式验证
    faith = m15.verify_answer(result["answer"], passages)
    faithfulness = faith["faithfulness"]

    # 答案完整性：answer_keys 在最终答案里的命中率
    a_hit, a_total, _ = keyword_hit_ratio(qa.get("answer_keys", []), result["answer"])
    answer_completeness = a_hit / a_total if a_total else 1.0

    # Answer Relevancy (RAGAS)：问题-答案语义相似度
    relevancy = answer_relevancy(qa["question"], result["answer"])

    return {
        "id": qa["id"],
        "pdf": qa["pdf"],
        "type": qa["type"],
        "question": qa["question"],
        "context_recall": round(retrieval_recall, 3),
        "faithfulness": round(faithfulness, 3),
        "answer_relevancy": round(relevancy, 3),
        "answer_completeness": round(answer_completeness, 3),
        "latency": latency,
        "confidence": result.get("confidence", 0.0),
        "retries": result.get("retries", 0),
        "answer_preview": result["answer"][:160],
    }


def evaluate(pdfs: list[str] = None, runner=None,
             dataset: list[dict] = None) -> dict:
    """跑全量评测。返回 {per_item, per_pdf, overall}。"""
    dataset = dataset if dataset is not None else load_dataset()
    pdfs = pdfs or sorted(set(q["pdf"] for q in dataset))
    per_item = []
    for qa in dataset:
        if qa["pdf"] not in pdfs:
            continue
        try:
            per_item.append(evaluate_one(qa, source_filter=qa["pdf"], runner=runner))
        except Exception as e:
            per_item.append({
                "id": qa["id"], "pdf": qa["pdf"], "type": qa["type"],
                "question": qa["question"], "error": f"{type(e).__name__}: {e}",
                "context_recall": 0.0, "faithfulness": 0.0,
                "answer_relevancy": 0.0, "answer_completeness": 0.0, "latency": 0.0,
            })

    # 聚合
    def agg(items, key):
        vals = [it.get(key, 0) for it in items if "error" not in it]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    per_pdf = {}
    for pdf in pdfs:
        items = [it for it in per_item if it["pdf"] == pdf]
        per_pdf[pdf] = {
            "n": len(items),
            "context_recall": agg(items, "context_recall"),
            "faithfulness": agg(items, "faithfulness"),
            "answer_relevancy": agg(items, "answer_relevancy"),
            "answer_completeness": agg(items, "answer_completeness"),
            "avg_latency": agg(items, "latency"),
        }
    overall = {
        "n": len(per_item),
        "context_recall": agg(per_item, "context_recall"),
        "faithfulness": agg(per_item, "faithfulness"),
        "answer_relevancy": agg(per_item, "answer_relevancy"),
        "answer_completeness": agg(per_item, "answer_completeness"),
        "avg_latency": agg(per_item, "latency"),
    }
    # 多跳（relation 类）成功率
    rel_items = [it for it in per_item if it.get("type") == "relation"]
    overall["multihop_success"] = (
        round(sum(1 for it in rel_items if it.get("answer_completeness", 0) >= 0.9999) / len(rel_items), 4)
        if rel_items else 1.0
    )
    return {"per_item": per_item, "per_pdf": per_pdf, "overall": overall}


def render_report(ev: dict) -> str:
    """渲染文本报告（RAGAS 标准指标）。"""
    lines = ["=" * 60, "RAG 评估报告（RAGAS 标准指标）", "=" * 60]
    for pdf, s in ev["per_pdf"].items():
        lines.append(f"\n[{pdf}] ({s['n']} 题)")
        lines.append(f"  Context Recall    (上下文召回) : {s['context_recall']:.2%}")
        lines.append(f"  Faithfulness      (答案忠实度) : {s['faithfulness']:.2%}")
        lines.append(f"  Answer Relevancy  (答案相关性) : {s['answer_relevancy']:.2%}")
        lines.append(f"  Answer Completeness(完整性/F1) : {s['answer_completeness']:.2%}")
        lines.append(f"  平均延迟                        : {s['avg_latency']:.2f}s")
    o = ev["overall"]
    lines.append("\n" + "-" * 60)
    lines.append("总体")
    lines.append(f"  Context Recall    : {o['context_recall']:.2%}")
    lines.append(f"  Faithfulness      : {o['faithfulness']:.2%}")
    lines.append(f"  Answer Relevancy  : {o['answer_relevancy']:.2%}")
    lines.append(f"  Answer Completeness(F1): {o['answer_completeness']:.2%}")
    lines.append(f"  多跳成功率        : {o['multihop_success']:.2%}")
    lines.append(f"  平均延迟          : {o['avg_latency']:.2f}s")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RAG 评估框架")
    ap.add_argument("--pdf", action="append", help="指定 PDF（可多次），缺省全跑")
    ap.add_argument("--save", help="结果保存到 JSON")
    args = ap.parse_args()
    ev = evaluate(pdfs=args.pdf)
    print(render_report(ev))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
        print(f"\n已保存到: {args.save}")
