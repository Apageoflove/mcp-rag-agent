"""16_agent_orchestrator 准确率验证：端到端答案 F1（两 PDF）。

对每题：跑完整 编排器（路由→检索→推理→反思+重试），验证最终答案是否
包含金标准 answer_keys。两 PDF 各自统计。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m16 = SourceFileLoader("m16", str(Path(__file__).resolve().parent / "16_agent_orchestrator.py")).load_module()
from _eval_helpers import (
    get_qa_by_pdf, keyword_hit_ratio, aggregate_f1, print_summary,
)

PDFS = ["bge_paper.pdf", "MiniMax_M1_tech_report.pdf"]
TOP_K = 5

print("=" * 60)
print("16_agent_orchestrator 准确率验证（端到端答案 F1，两 PDF）")
print("=" * 60)

per_pdf = {}
all_item_f1 = []
for pdf in PDFS:
    print(f"\n[{pdf}]")
    qa_list = get_qa_by_pdf(pdf)
    item_f1 = []
    for qa in qa_list:
        r = m16.answer(qa["question"], source_filter=pdf, top_k=TOP_K)
        hit, total, missed = keyword_hit_ratio(qa["answer_keys"], r["answer"])
        f1 = hit / total if total else 1.0
        item_f1.append(f1)
        mark = "✅" if f1 >= 0.9999 else "❌"
        print(f"  {mark} {qa['id']} F1={f1:.0%} ({hit}/{total}) "
              f"conf={r['confidence']:.2f} retry={r['retries']} | {qa['question'][:34]}")
        if missed:
            print(f"      缺失: {missed}")
    per_pdf[pdf] = aggregate_f1(item_f1)
    all_item_f1.extend(item_f1)

overall = sum(all_item_f1) / len(all_item_f1) if all_item_f1 else 0
ok = print_summary("16_agent_orchestrator", per_pdf, overall)
sys.exit(0 if ok else 1)
