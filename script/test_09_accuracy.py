"""09消歧准确率验证脚本。基于MiniMax_M1真实数据，验证消歧+归一化准确率。"""
import sys
sys.path.insert(0, 'script')
from importlib.machinery import SourceFileLoader
m09 = SourceFileLoader('m09', 'script/09_kg_builder.py').load_module()

# 金标准（人工标注的正确消歧结果）
GOLD_ENTITY_MAPPING = {
    'M1': 'MiniMax-M1',
    'DeepSeek R1': 'DeepSeek-R1',
    'Lightning Attention': 'lightning attention mechanism',
    'lightning attention': 'lightning attention mechanism',
    # 以下应保持独立（不合并）
    'MiniMax-M1': 'MiniMax-M1',
    'MiniMax': 'MiniMax',
    'MiniMax-Text-01': 'MiniMax-Text-01',
    'Mixture-of-Experts': 'Mixture-of-Experts',
    'lightning attention mechanism': 'lightning attention mechanism',
    'reinforcement learning': 'reinforcement learning',
    'CISPO': 'CISPO',
    'novel RL algorithm': 'novel RL algorithm',
    'importance sampling weights': 'importance sampling weights',
    'RL variants': 'RL variants',
    'hybrid-attention': 'hybrid-attention',
    'H800': 'H800',
    '80K thinking budget': '80K thinking budget',
    'DeepSeek-R1': 'DeepSeek-R1',
    'Qwen3-235B': 'Qwen3-235B',
}

GOLD_RELATION_NORMALIZE = {
    'based_on': 'based_on',
    'proposes': 'proposes',
    'powered by': 'uses',
    'uses': 'uses',
    'compared_with': 'compared_with',
    'outperforms': 'outperforms',
    'is': 'is_a',
    'clips': 'clips',
    'trained_on': 'trained_on',
    'released with': 'released_with',
    'comparable to': 'outperforms',
    'superior to': 'outperforms',
}

print("=" * 60)
print("09 准确率验证")
print("=" * 60)

triples = m09.load_triples('output/kg_triples/MiniMax_M1_tech_report_test.json')
freq = m09.collect_entities(triples)

predicted_mapping = m09.disambiguate_entities(freq)

print("\n--- 实体消歧验证 ---")
ent_total = 0
ent_correct = 0
ent_errors = []
for entity, gold_canon in GOLD_ENTITY_MAPPING.items():
    if entity not in predicted_mapping:
        continue
    ent_total += 1
    pred_canon = predicted_mapping[entity]
    if pred_canon == gold_canon:
        ent_correct += 1
    else:
        ent_errors.append((entity, gold_canon, pred_canon))

ent_acc = ent_correct / ent_total * 100 if ent_total > 0 else 0
print(f"实体消歧：{ent_correct}/{ent_total} = {ent_acc:.1f}%")
if ent_errors:
    print("错误：")
    for e, g, p in ent_errors:
        print(f"  {e!r} 期望={g!r} 实际={p!r}")
else:
    print("✅ 实体消歧全部正确")

print("\n--- 关系归一化验证 ---")
rel_total = 0
rel_correct = 0
rel_errors = []
for rel, gold_std in GOLD_RELATION_NORMALIZE.items():
    rel_total += 1
    pred_std = m09.normalize_relation(rel)
    if pred_std == gold_std:
        rel_correct += 1
    else:
        rel_errors.append((rel, gold_std, pred_std))

rel_acc = rel_correct / rel_total * 100 if rel_total > 0 else 0
print(f"关系归一化：{rel_correct}/{rel_total} = {rel_acc:.1f}%")
if rel_errors:
    print("错误：")
    for r, g, p in rel_errors:
        print(f"  {r!r} 期望={g!r} 实际={p!r}")
else:
    print("✅ 关系归一化全部正确")

print("\n--- 三元组去重验证 ---")
normalized = m09.normalize_triples(triples, predicted_mapping)
GOLD_TRIPLE_COUNT = 17
if len(normalized) == GOLD_TRIPLE_COUNT:
    print(f"✅ 去重后三元组数：{len(normalized)}（与金标准{GOLD_TRIPLE_COUNT}一致）")
    trip_acc = 100.0
else:
    print(f"❌ 去重后三元组数：{len(normalized)}（期望{GOLD_TRIPLE_COUNT}）")
    trip_acc = min(len(normalized), GOLD_TRIPLE_COUNT) / max(len(normalized), GOLD_TRIPLE_COUNT) * 100

print("\n--- 精确三元组集合 F1 ---")
from _kg_gold import GOLD_TRIPLES, compute_f1
norm_pred = {
    (t['subject'], t['relation'], t['object']) for t in normalized
}
p_triple, r_triple, f1_triple = compute_f1(norm_pred, GOLD_TRIPLES)
print(f"  pred={len(norm_pred)}, gold={len(GOLD_TRIPLES)}, "
      f"TP={len(norm_pred & GOLD_TRIPLES)}")
print(f"  P={p_triple:.2%} R={r_triple:.2%} F1={f1_triple:.2%}")
missing = GOLD_TRIPLES - norm_pred
extra = norm_pred - GOLD_TRIPLES
if missing:
    print(f"  ❌ 缺失: {missing}")
if extra:
    print(f"  ❌ 多余: {extra}")
if not missing and not extra:
    print("  ✅ 精确集合完全匹配（归一化后输出 == 金标准）")

print("\n" + "=" * 60)
overall = (ent_acc + rel_acc + trip_acc) / 3
print(f"综合准确率：{overall:.1f}%")
print(f"精确三元组 F1：{f1_triple:.2%}")
if overall >= 95 and f1_triple == 1.0:
    print(f"✅ PASS (综合≥95% 且 三元组F1=100%)")
elif overall >= 95:
    print(f"⚠️ CONDITIONAL (综合≥95% 但 三元组F1={f1_triple:.2%})")
elif overall >= 90:
    print(f"⚠️ CONDITIONAL (90-94%)")
else:
    print(f"❌ FAIL (<90%)")

# ── bge 两-PDF 验证：引文过滤 + 关系归一化 ──────────────────
print("\n" + "=" * 60)
print("[bge_paper.pdf] 09 通用能力验证")
print("=" * 60)
import json as _json
bge_raw = _json.load(open('output/kg_triples/bge_paper.pdf.json'))
bge_freq = m09.collect_entities(bge_raw)
bge_map = m09.disambiguate_entities(bge_freq)
bge_norm = m09.normalize_triples(bge_raw, bge_map, drop_citations=True)
bge_rels = {t['relation'] for t in bge_norm}
# 1) 引文型关系应被过滤掉
citation_left = bge_rels & m09._CITATION_RELATIONS
print(f"  归一化后三元组: {len(bge_norm)} 条（原始 {len(bge_raw)}）")
if not citation_left:
    print(f"  ✅ 引文型关系已全部过滤（{m09._CITATION_RELATIONS}）")
else:
    print(f"  ❌ 仍残留引文型关系: {citation_left}")
# 2) 关系归一化（bge 样例）
from _kg_gold import BGE_RELATION_NORMALIZE
bge_rel_ok = sum(1 for r, g in BGE_RELATION_NORMALIZE.items()
                 if m09.normalize_relation(r) == g)
bge_rel_tot = len(BGE_RELATION_NORMALIZE)
print(f"  关系归一化: {bge_rel_ok}/{bge_rel_tot} = {bge_rel_ok/bge_rel_tot:.0%}")
# 3) 关键实体召回（M3-Embedding 的 uses 关系应有 retrieval 类对象）
bge_uses = {t['object'] for t in bge_norm
            if 'm3-embedding' in t['subject'].lower() and t['relation'] == 'uses'}
has_retrieval = any('retrieval' in o.lower() for o in bge_uses)
print(f"  M3-Embedding uses 对象数: {len(bge_uses)}, 含 retrieval 类: {has_retrieval}")
bge_ok = (not citation_left) and bge_rel_ok == bge_rel_tot and has_retrieval
print(f"  {'✅ bge 通用能力 PASS' if bge_ok else '❌ bge 未达标'}")

both_ok = (overall >= 95 and f1_triple == 1.0 and bge_ok)
print("=" * 60)
print(f"\n{'✅ 09_kg_builder 两 PDF 验证 PASS' if both_ok else '❌ 09 未达标（两 PDF）'}")
print("=" * 60)
