"""12_router_agent 准确率验证：问题类型分类 F1（两 PDF）。

规则路由是确定性通用英文模式，对两份 PDF 的金标准问题都应 100% 命中类型。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m12 = SourceFileLoader("m12", str(Path(__file__).resolve().parent / "12_router_agent.py")).load_module()
from _eval_helpers import get_qa_by_pdf, compute_f1, print_summary

PDFS = ["bge_paper.pdf", "MiniMax_M1_tech_report.pdf"]

print("=" * 60)
print("12_router_agent 准确率验证（类型分类 F1，纯规则路由）")
print("=" * 60)

per_pdf = {}
all_item_f1 = []
for pdf in PDFS:
    print(f"\n[{pdf}]")
    qa_list = get_qa_by_pdf(pdf)
    item_f1 = []
    for qa in qa_list:
        r = m12.route(qa["question"], use_llm_fallback=False)
        gold = {qa["type"]}
        pred = {r["type"]}
        p, rr, f1 = compute_f1(pred, gold)
        item_f1.append(f1)
        mark = "✅" if f1 >= 0.9999 else "❌"
        print(f"  {mark} {qa['id']} gold={qa['type']:8} pred={r['type']:8} "
              f"src={r['source']:6} | {qa['question'][:42]}")
    from _eval_helpers import aggregate_f1
    per_pdf[pdf] = aggregate_f1(item_f1)
    all_item_f1.extend(item_f1)

overall = sum(all_item_f1) / len(all_item_f1) if all_item_f1 else 0
ok = print_summary("12_router_agent", per_pdf, overall)
sys.exit(0 if ok else 1)
