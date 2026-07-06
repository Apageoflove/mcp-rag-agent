"""10_kg_query 准确率验证：答案投影 F1（两份 PDF）。

- bge_paper.pdf：用 08 的三元组经 09 归一化建内存图，查 M3-Embedding 的
  uses/supports 关系，金标准是图谱里该关系的真实内容（验证查询能取回
  图谱内容）。
- MiniMax_M1_tech_report.pdf：原有金标准问答。
"""
import sys
sys.path.insert(0, 'script')
from importlib.machinery import SourceFileLoader
from _kg_gold import compute_f1

m10 = SourceFileLoader('m10', 'script/10_kg_query.py').load_module()
m09 = SourceFileLoader('m09', 'script/09_kg_builder.py').load_module()
m05 = SourceFileLoader('m05', 'script/05_llm_client.py').load_module()


# ── MiniMax 金标准（原有）─────────────────────────────────
MINIMAX_QA = [
    {'question': 'what does MiniMax-M1 use',
     'expected': {'Mixture-of-Experts', 'lightning attention mechanism',
                  'reinforcement learning', 'hybrid-attention', 'CISPO'}},
    {'question': 'what is CISPO',
     'expected': {'novel RL algorithm', 'importance sampling weights', 'RL variants'}},
    {'question': 'relationship between MiniMax-M1 and DeepSeek-R1',
     'expected': {'MiniMax-M1', 'DeepSeek-R1', 'outperforms', 'compared_with'}},
    {'question': 'what is MiniMax-M1 based on',
     'expected': {'MiniMax-Text-01'}},
    {'question': 'who proposes MiniMax-M1',
     'expected': {'MiniMax'}},
    {'question': 'what does MiniMax-M1 outperform',
     'expected': {'DeepSeek-R1', 'Qwen3-235B'}},
    {'question': 'what is MiniMax-M1 trained on',
     'expected': {'H800'}},
    {'question': 'what is MiniMax-M1 compared with',
     'expected': {'DeepSeek-R1', 'Qwen3-235B'}},
]


def _build_bge_graph():
    """08 三元组 → 09 归一化（含引文过滤）→ 内存图。"""
    import json
    raw = json.load(open('output/kg_triples/bge_paper.pdf.json'))
    freq = m09.collect_entities(raw)
    mapping = m09.disambiguate_entities(freq)
    norm = m09.normalize_triples(raw, mapping, drop_citations=True)
    from _memory_graph import build_graph_from_triples
    return build_graph_from_triples(norm)


def _bge_graph_uses():
    """从图谱里取 M3-Embedding 的真实 uses 对象集，作为查询金标准。"""
    g = _build_bge_graph()
    uses = set()
    for e in g._edges:
        if 'm3-embedding' in e['subject'].lower() and e['predicate'] == 'uses':
            uses.add(e['object'])
    return g, uses


def _run_pdf(label, qa_list, driver):
    """跑一份 PDF 的问答，返回 (results, avg_f1)。"""
    client = m05.create_client()
    results = []
    for i, qa in enumerate(qa_list, 1):
        q = qa['question']
        gold = qa['expected']
        result = m10.graph_search(q, driver=driver, client=client,
                                  fallback_to_vector=False)
        predicted = set(result.get('answers', []))
        p, r, f1 = compute_f1(predicted, gold)
        ok = (f1 == 1.0)
        results.append((q, f1, ok))
        mark = '✅' if ok else '❌'
        print(f"  {mark} F1={f1:.0%} P={p:.0%} R={r:.0%} | {q[:46]}")
        if not ok:
            miss = gold - predicted
            extra = predicted - gold
            if miss:
                print(f"       缺失: {set(miss)}")
            if extra:
                print(f"       多余: {set(extra)}")
    avg = sum(r[1] for r in results) / len(results) if results else 0
    return results, avg


print("=" * 60)
print("10_kg_query 准确率验证（答案投影 F1，两 PDF）")
print("=" * 60)

per_pdf = {}

# ── MiniMax（用金标准测试文件建图）─────────────────────────
print("\n[MiniMax_M1_tech_report.pdf]")
try:
    mm_driver = m09.get_neo4j_driver()
except Exception:
    mm_driver = m09.get_memory_graph(
        'output/kg_triples/MiniMax_M1_tech_report_test.json')
mm_res, mm_avg = _run_pdf("MiniMax", MINIMAX_QA, mm_driver)
if hasattr(mm_driver, 'close'):
    try:
        mm_driver.close()
    except Exception:
        pass
per_pdf["MiniMax_M1_tech_report.pdf"] = mm_avg

# ── bge（用 08 三元组建图；金标准=图谱真实 uses 内容）─────
print("\n[bge_paper.pdf]")
bge_driver, bge_uses_gold = _bge_graph_uses()
bge_qa = [
    {'question': 'what does M3-Embedding use', 'expected': bge_uses_gold},
]
bge_res, bge_avg = _run_pdf("bge", bge_qa, bge_driver)
per_pdf["bge_paper.pdf"] = bge_avg

# ── 汇总 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
all_ok = True
for pdf, avg in per_pdf.items():
    status = "✅" if avg >= 0.9999 else "❌"
    print(f"  {pdf}: 平均 F1={avg:.2%} {status}")
    if avg < 0.9999:
        all_ok = False
overall = sum(per_pdf.values()) / len(per_pdf)
print(f"  总体平均 F1={overall:.2%} {'✅' if overall >= 0.9999 else '❌'}")
if all_ok and overall >= 0.9999:
    print("✅ 10_kg_query 两 PDF F1 均 100% PASS")
else:
    print("❌ 10_kg_query 未达标")
print("=" * 60)
sys.exit(0 if all_ok and overall >= 0.9999 else 1)
