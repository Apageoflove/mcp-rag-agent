"""第 4 篇英文论文验证：ResNet (arxiv 1512.03385，视觉领域，与之前 NLP 三篇不同)。

验证整套链路对不同领域、不同格式 PDF 的普适性。金标准按论文实际内容标注。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m16 = SourceFileLoader("m16", str(Path(__file__).resolve().parent / "16_agent_orchestrator.py")).load_module()
from _eval_helpers import keyword_hit_ratio, aggregate_f1

PDF = "resnet.pdf"
GOLD = [
    {"id": "r1", "question": "What is ResNet?",
     "answer_keys": ["ResNet", "residual"]},
    {"id": "r2", "question": "What is the core idea of ResNet?",
     "answer_keys": ["residual"]},
    {"id": "r3", "question": "What dataset is ResNet evaluated on?",
     "answer_keys": ["ImageNet"]},
    {"id": "r4", "question": "How many layers does the deepest ResNet model have?",
     "answer_keys": ["152"]},
    {"id": "r5", "question": "What normalization technique does ResNet use?",
     "answer_keys": ["batch normalization"]},
    {"id": "r6", "question": "Who are the authors of ResNet?",
     "answer_keys": ["Microsoft"]},
    {"id": "r7", "question": "Summarize the ResNet paper.",
     "answer_keys": ["ResNet", "residual", "ImageNet"]},
    {"id": "r8", "question": "What does ResNet use to train the network?",
     "answer_keys": ["backpropagation", "softmax"]},
]

print("=" * 60)
print(f"第 4 篇论文验证：{PDF}（ResNet，视觉领域，端到端 F1）")
print("=" * 60)
item_f1 = []
for qa in GOLD:
    r = m16.answer(qa["question"], source_filter=PDF, top_k=5)
    hit, total, missed = keyword_hit_ratio(qa["answer_keys"], r["answer"])
    f1 = hit / total if total else 1.0
    item_f1.append(f1)
    mark = "✅" if f1 >= 0.9999 else "❌"
    print(f"  {mark} {qa['id']} F1={f1:.0%} ({hit}/{total}) t={r['time']}s | {qa['question'][:42]}")
    if missed:
        print(f"       缺失: {missed}")
stat = aggregate_f1(item_f1)
print("\n" + "=" * 60)
print(f"{PDF}: 平均 F1={stat['avg_f1']:.2%}, 通过率={stat['pass_rate']:.0%}")
print(f"{'✅ 第 4 篇(ResNet) F1=100% PASS' if stat['avg_f1']>=0.9999 else '⚠️ F1='+format(stat['avg_f1'],'.2%')+'，见上方缺失'}")
print("=" * 60)
