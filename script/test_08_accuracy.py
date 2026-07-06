"""08_kg_extractor 准确率验证：归一化感知 F1。

08 输出经 09 归一化后对比金标准，度量语义召回与精确。
"""
import sys
import json
sys.path.insert(0, 'script')
from importlib.machinery import SourceFileLoader
from _kg_gold import GOLD_TRIPLES, GOLD_ENTITY_MAPPING, compute_f1, RAW_TRIPLES_PATH

m09 = SourceFileLoader('m09', 'script/09_kg_builder.py').load_module()


print("=" * 60)
print("08_kg_extractor 准确率验证（归一化感知 F1）")
print("=" * 60)

with open(RAW_TRIPLES_PATH, 'r', encoding='utf-8') as f:
    raw_triples = json.load(f)
print(f"\n08 原始输出: {len(raw_triples)} 条三元组")

raw_pred = {
    (t['subject'].strip(), t['relation'].strip(), t['object'].strip())
    for t in raw_triples
}
p_naive, r_naive, f1_naive = compute_f1(raw_pred, GOLD_TRIPLES)
print(f"\n--- Naive exact-match F1 ---")
print(f"  pred={len(raw_pred)}, gold={len(GOLD_TRIPLES)}, "
      f"TP={len(raw_pred & GOLD_TRIPLES)}")
print(f"  P={p_naive:.2%} R={r_naive:.2%} F1={f1_naive:.2%}")

print(f"\n--- 归一化感知 F1 ---")
print(f"  用 09 的 normalize_triples + 冻结金标准实体映射归一化 08 输出")

normalized = m09.normalize_triples(raw_triples, GOLD_ENTITY_MAPPING)
norm_pred = {
    (t['subject'], t['relation'], t['object']) for t in normalized
}
p, r, f1 = compute_f1(norm_pred, GOLD_TRIPLES)
print(f"  pred={len(norm_pred)}, gold={len(GOLD_TRIPLES)}, "
      f"TP={len(norm_pred & GOLD_TRIPLES)}")
print(f"  P={p:.2%} R={r:.2%} F1={f1:.2%}")

missing = GOLD_TRIPLES - norm_pred
extra = norm_pred - GOLD_TRIPLES
if missing:
    print(f"  缺失: {missing}")
if extra:
    print(f"  多余: {extra}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print(f"Naive exact-match F1: {f1_naive:.2%}")
print(f"归一化感知 F1:        {f1:.2%}")
print(f"  08 原始输出 {len(raw_pred)} 条 → 经09归一化 → {len(norm_pred)} 条")
print(f"  金标准 {len(GOLD_TRIPLES)} 条，归一化后精确匹配 {len(norm_pred & GOLD_TRIPLES)} 条")
if f1 == 1.0:
    print(f"\n✅ 08 抽取器 F1 = 100% PASS")
elif f1 >= 0.95:
    print(f"\n⚠️ 08 抽取器 F1 = {f1:.2%} CONDITIONAL")
else:
    print(f"\n❌ 08 抽取器 F1 = {f1:.2%} FAIL")
print("=" * 60)
