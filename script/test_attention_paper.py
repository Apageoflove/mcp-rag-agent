"""第 3 篇英文论文验证：Attention Is All You Need（arxiv 1706.03762）。

验证整套链路（11 层次摘要 / 12 路由 / 13 检索 / 14 推理 / 15 反思 / 16 编排）
对「全新论文」是否同样 F1=100%。金标准按论文实际内容人工标注。
不达标的题要打印出来，便于定位是哪个环节的问题。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m16 = SourceFileLoader("m16", str(Path(__file__).resolve().parent / "16_agent_orchestrator.py")).load_module()
from _eval_helpers import keyword_hit_ratio, aggregate_f1, print_summary

PDF = "attention_is_all_you_need.pdf"

# 金标准（按 Attention Is All You Need 实际内容标注，与算法无关）
GOLD = [
    {"id": "a1", "question": "What model architecture does the Transformer use?",
     "answer_keys": ["encoder", "decoder", "attention"]},
    {"id": "a2", "question": "Does the Transformer use recurrence or convolution?",
     "answer_keys": ["recurrence", "convolution"]},
    {"id": "a3", "question": "What is the core mechanism of the Transformer?",
     "answer_keys": ["self-attention"]},
    {"id": "a4", "question": "What attention variant does the Transformer use?",
     "answer_keys": ["multi-head"]},
    {"id": "a5", "question": "Who are the authors of the Transformer paper?",
     "answer_keys": ["Google"]},
    {"id": "a6", "question": "What dataset is the Transformer trained on for translation?",
     "answer_keys": ["WMT"]},
    {"id": "a7", "question": "Summarize the Attention Is All You Need paper.",
     "answer_keys": ["Transformer", "self-attention", "encoder", "attention"]},
    {"id": "a8", "question": "What is the Transformer?",
     "answer_keys": ["Transformer", "attention"]},
]

print("=" * 60)
print(f"第 3 篇论文验证：{PDF}（端到端，F1）")
print("=" * 60)

item_f1 = []
for qa in GOLD:
    r = m16.answer(qa["question"], source_filter=PDF, top_k=5)
    hit, total, missed = keyword_hit_ratio(qa["answer_keys"], r["answer"])
    f1 = hit / total if total else 1.0
    item_f1.append(f1)
    mark = "✅" if f1 >= 0.9999 else "❌"
    print(f"  {mark} {qa['id']} F1={f1:.0%} ({hit}/{total}) retry={r['retries']} "
          f"t={r['time']}s | {qa['question'][:42]}")
    if missed:
        print(f"       缺失: {missed}")

stat = aggregate_f1(item_f1)
print("\n" + "=" * 60)
print(f"{PDF}: 平均 F1={stat['avg_f1']:.2%}, 通过率={stat['pass_rate']:.0%}")
if stat["avg_f1"] >= 0.9999:
    print(f"✅ 第 3 篇论文 F1=100% PASS（链路对新论文通用）")
else:
    print(f"⚠️ 第 3 篇 F1={stat['avg_f1']:.2%}，需定位：见上方缺失项")
print("=" * 60)
