"""14_reasoning_agent 准确率验证：答案事实 F1（两 PDF）。

对每题：路由→检索→推理，验证最终答案是否包含金标准 answer_keys 的关键事实。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m12 = SourceFileLoader("m12", str(Path(__file__).resolve().parent / "12_router_agent.py")).load_module()
m13 = SourceFileLoader("m13", str(Path(__file__).resolve().parent / "13_retriever_agent.py")).load_module()
m14 = SourceFileLoader("m14", str(Path(__file__).resolve().parent / "14_reasoning_agent.py")).load_module()
from _eval_helpers import (
    get_qa_by_pdf, keyword_hit_ratio, aggregate_f1, print_summary,
)

PDFS = ["bge_paper.pdf", "MiniMax_M1_tech_report.pdf"]
TOP_K = 5

print("=" * 60)
print("14_reasoning_agent 准确率验证（答案事实 F1，两 PDF）")
print("=" * 60)

per_pdf = {}
all_item_f1 = []
for pdf in PDFS:
    print(f"\n[{pdf}]")
    qa_list = get_qa_by_pdf(pdf)
    item_f1 = []
    for qa in qa_list:
        strat = m12.route(qa["question"], use_llm_fallback=False)
        passages = m13.retrieve(qa["question"], strategy=strat, top_k=TOP_K,
                                source_filter=pdf)
        # 用推理 Agent 的生产入口（自一致性），它是 14 对外提供的实际能力
        r = m14.reason_with_self_consistency(qa["question"], passages, n=3,
                                             source_filter=pdf)
        hit, total, missed = keyword_hit_ratio(qa["answer_keys"], r["answer"])
        f1 = hit / total if total else 1.0
        item_f1.append(f1)
        mark = "✅" if f1 >= 0.9999 else "❌"
        print(f"  {mark} {qa['id']} F1={f1:.0%} ({hit}/{total}) | {qa['question'][:40]}")
        if missed:
            print(f"      缺失: {missed}")
    per_pdf[pdf] = aggregate_f1(item_f1)
    all_item_f1.extend(item_f1)

overall = sum(all_item_f1) / len(all_item_f1) if all_item_f1 else 0
ok = print_summary("14_reasoning_agent", per_pdf, overall)
sys.exit(0 if ok else 1)
