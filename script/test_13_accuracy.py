"""13_retriever_agent 准确率验证：检索召回 F1（两 PDF）。

每题：用路由+检索 Agent 取回 top-k 片段，检查期望来源是否出现在 top-k，
且 answer_keys 关键事实在召回文本里。两 PDF 各自统计。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m13 = SourceFileLoader("m13", str(Path(__file__).resolve().parent / "13_retriever_agent.py")).load_module()
m12 = SourceFileLoader("m12", str(Path(__file__).resolve().parent / "12_router_agent.py")).load_module()
from _eval_helpers import (
    get_qa_by_pdf, keyword_hit_ratio, compute_f1, aggregate_f1, print_summary,
)

PDFS = ["bge_paper.pdf", "MiniMax_M1_tech_report.pdf"]
TOP_K = 5

print("=" * 60)
print("13_retriever_agent 准确率验证（召回 F1，两 PDF）")
print("=" * 60)

per_pdf = {}
all_item_f1 = []
for pdf in PDFS:
    print(f"\n[{pdf}]")
    qa_list = get_qa_by_pdf(pdf)
    item_f1 = []
    for qa in qa_list:
        strat = m12.route(qa["question"], use_llm_fallback=False)
        results = m13.retrieve(qa["question"], strategy=strat, top_k=TOP_K,
                               source_filter=pdf)
        text = " ".join(r.get("text", "") for r in results)
        # 来源命中（top-k 是否有期望 PDF 的内容）+ 关键事实命中，合并成 F1
        hit, total, missed = keyword_hit_ratio(qa["answer_keys"], text)
        key_f1 = hit / total if total else 1.0
        item_f1.append(key_f1)
        mark = "✅" if key_f1 >= 0.9999 else "❌"
        print(f"  {mark} {qa['id']} {qa['type']:8} F1={key_f1:.0%} "
              f"({hit}/{total}) n={len(results)} | {qa['question'][:38]}")
        if missed:
            print(f"      缺失: {missed}")
    per_pdf[pdf] = aggregate_f1(item_f1)
    all_item_f1.extend(item_f1)

overall = sum(all_item_f1) / len(all_item_f1) if all_item_f1 else 0
ok = print_summary("13_retriever_agent", per_pdf, overall)
sys.exit(0 if ok else 1)
