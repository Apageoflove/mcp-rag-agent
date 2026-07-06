"""15_reflection_agent 准确率验证：幻觉检测 F1（两 PDF）。

对 reflection_cases 里每条 (claim, passage, supported)：
  用 15 的检索式验证判定 supported/unsupported，对比金标准算 F1。
两 PDF 各自统计 + 总体。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m15 = SourceFileLoader("m15", str(Path(__file__).resolve().parent / "15_reflection_agent.py")).load_module()
from _eval_helpers import compute_f1, print_summary

PDFS = ["bge_paper.pdf", "MiniMax_M1_tech_report.pdf"]

print("=" * 60)
print("15_reflection_agent 准确率验证（幻觉检测 F1，两 PDF）")
print("=" * 60)

# 加载两 PDF 的反思用例
import json
with open(Path(__file__).resolve().parent / "18_eval_dataset.json", "r", encoding="utf-8") as f:
    all_cases = json.load(f)["reflection_cases"]

per_pdf = {}
all_pred_gold = []  # (pdf, pred_set, gold_set)
for pdf in PDFS:
    print(f"\n[{pdf}]")
    cases = [c for c in all_cases if c["pdf"] == pdf]
    tp = fp = fn = tn = 0
    details = []
    for c in cases:
        pred_supported = m15.is_claim_supported_by_passage(c["claim"], c["passage"])
        gold_supported = c["supported"]
        # 二分类 F1（positive = supported）
        if pred_supported and gold_supported:
            tp += 1
        elif pred_supported and not gold_supported:
            fp += 1
        elif not pred_supported and gold_supported:
            fn += 1
        else:
            tn += 1
        mark = "✅" if pred_supported == gold_supported else "❌"
        print(f"  {mark} {c['id']} pred={'sup' if pred_supported else 'unsup':4} "
              f"gold={'sup' if gold_supported else 'unsup':4} | {c['claim'][:48]}")
        details.append((c["id"], pred_supported, gold_supported))
    # F1（supported 视为正类）
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    per_pdf[pdf] = {"avg_f1": f1, "pass_rate": (tp + tn) / len(cases) if cases else 1.0,
                    "n": len(cases)}
    all_pred_gold.append((pdf, tp, fp, fn, tn))

print("\n" + "=" * 60)
print("15_reflection_agent 汇总")
print("=" * 60)
all_tp = sum(x[1] for x in all_pred_gold)
all_fp = sum(x[2] for x in all_pred_gold)
all_fn = sum(x[3] for x in all_pred_gold)
all_tn = sum(x[4] for x in all_pred_gold)
op = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 1.0
orr = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 1.0
of1 = 2 * op * orr / (op + orr) if (op + orr) else 0.0

all_ok = True
for pdf in PDFS:
    s = per_pdf[pdf]
    status = "✅" if s["avg_f1"] >= 0.9999 else "❌"
    print(f"  {pdf}: F1={s['avg_f1']:.2%}, acc={s['pass_rate']:.0%} ({s['n']} 例) {status}")
    if s["avg_f1"] < 0.9999:
        all_ok = False
status = "✅" if of1 >= 0.9999 else "❌"
print(f"  总体 F1={of1:.2%} {status}")
if all_ok and of1 >= 0.9999:
    print(f"  ✅ 15_reflection_agent 两 PDF F1 均 100% PASS")
else:
    print(f"  ❌ 15_reflection_agent 未达标")
print("=" * 60)
sys.exit(0 if all_ok and of1 >= 0.9999 else 1)
