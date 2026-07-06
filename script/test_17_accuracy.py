"""17_eval_framework 准确率验证：度量计算正确性 F1。

用合成用例（已知 question/answer/passages → 已知预期指标）验证评估框架的
四个度量函数算得对。计算值与预期精确一致 = F1=100%。两 PDF 各覆盖。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m17 = SourceFileLoader("m17", str(Path(__file__).resolve().parent / "17_eval_framework.py")).load_module()
from _eval_helpers import print_summary

PDFS = ["bge_paper.pdf", "MiniMax_M1_tech_report.pdf"]

print("=" * 60)
print("17_eval_framework 准确率验证（度量计算正确性 F1，两 PDF）")
print("=" * 60)


def _runner_factory(answer: str, passages: list[dict], confidence: float = 1.0):
    """造一个假的 runner，返回固定 answer/passages，用来隔离测试度量计算。"""
    def _run(question, source_filter=None, top_k=5):
        return {"answer": answer, "passages": passages,
                "confidence": confidence, "retries": 0}
    return _run


# 合成用例：每条给出 answer_keys、answer、passages，以及「预期」的三个[0,1]度量。
# 预期值是按度量定义人工算出的标准答案（评测标签，不是算法）。
SYNTH = [
    {  # 完美：keys 全在 passages，全在 answer，answer 全被支撑
        "id": "s1", "pdf": "bge_paper.pdf", "type": "fact",
        "question": "What is X?", "answer_keys": ["MiniMax-M1", "MoE"],
        "answer": "MiniMax-M1 uses MoE architecture.",
        "passages": [{"text": "MiniMax-M1 uses MoE architecture."}],
        "expect": {"context_recall": 1.0, "answer_completeness": 1.0,
                   "faithfulness": 1.0},
    },
    {  # 检索漏一个 key
        "id": "s2", "pdf": "MiniMax_M1_tech_report.pdf", "type": "fact",
        "question": "What is X?", "answer_keys": ["DeepSeek-R1", "Qwen3-235B"],
        "answer": "DeepSeek-R1 is compared.",
        "passages": [{"text": "DeepSeek-R1 is mentioned here."}],
        "expect": {"context_recall": 0.5, "answer_completeness": 0.5,
                   "faithfulness": 1.0},
    },
    {  # 答案引入幻觉实体（BERT 不在 passages）
        "id": "s3", "pdf": "bge_paper.pdf", "type": "fact",
        "question": "backbone?", "answer_keys": ["XLM-RoBERTa"],
        "answer": "The backbone is BERT.",  # BERT 不在 passages → 幻觉
        "passages": [{"text": "the encoder is XLM-RoBERTa adapted by RetroMAE."}],
        "expect": {"context_recall": 1.0, "answer_completeness": 0.0,
                   "faithfulness": 0.0},
    },
    {  # 部分忠实（两条断言一条无依据）
        "id": "s4", "pdf": "MiniMax_M1_tech_report.pdf", "type": "relation",
        "question": "rel?", "answer_keys": ["CISPO"],
        "answer": "CISPO clips importance sampling. MiniMax-M1 uses GPT-4.",
        "passages": [{"text": "CISPO clips importance sampling weights."}],
        "expect": {"context_recall": 1.0, "answer_completeness": 1.0,
                   "faithfulness": 0.5},
    },
]


per_pdf = {pdf: {"correct": 0, "total": 0} for pdf in PDFS}
all_correct = 0
all_total = 0

for pdf in PDFS:
    print(f"\n[{pdf}]")
    for c in SYNTH:
        if c["pdf"] != pdf:
            continue
        runner = _runner_factory(c["answer"], c["passages"])
        ev = m17.evaluate_one(c, source_filter=pdf, runner=runner)
        # 比对三个度量（四舍五入到 3 位）
        ok = True
        diffs = []
        for k in ("context_recall", "answer_completeness", "faithfulness"):
            got = round(ev[k], 3)
            exp = round(c["expect"][k], 3)
            if abs(got - exp) > 0.001:
                ok = False
                diffs.append(f"{k}: got={got} exp={exp}")
        per_pdf[pdf]["total"] += 1
        all_total += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} {c['id']} {c['question']:12} "
              f"R={ev['context_recall']:.2f} C={ev['answer_completeness']:.2f} "
              f"F={ev['faithfulness']:.2f}")
        if diffs:
            print(f"      " + "; ".join(diffs))
        if ok:
            per_pdf[pdf]["correct"] += 1
            all_correct += 1

# 每个 PDF 的 F1 = 正确算出的度量数 / 总度量数
per_f1 = {}
item_f1_all = []
for pdf in PDFS:
    # 每题 3 个度量，统计正确率
    metrics_correct = 0
    metrics_total = 0
    for c in SYNTH:
        if c["pdf"] != pdf:
            continue
        runner = _runner_factory(c["answer"], c["passages"])
        ev = m17.evaluate_one(c, source_filter=pdf, runner=runner)
        for k in ("context_recall", "answer_completeness", "faithfulness"):
            metrics_total += 1
            if abs(round(ev[k], 3) - round(c["expect"][k], 3)) <= 0.001:
                metrics_correct += 1
    f1 = metrics_correct / metrics_total if metrics_total else 1.0
    per_f1[pdf] = {"avg_f1": f1, "pass_rate": (1.0 if f1 >= 0.9999 else 0.0),
                   "n": metrics_total}

overall_f1 = sum(p["avg_f1"] for p in per_f1.values()) / len(per_f1)
ok = print_summary("17_eval_framework", per_f1, overall_f1)
sys.exit(0 if ok else 1)
