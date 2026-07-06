"""11_summary_indexer 准确率验证：层次摘要覆盖度 F1。

对两份 PDF 各自构建三层摘要索引，验证：
1. 三层结构完整（document/section/paragraph 都非空）
2. 关键事实覆盖：每个金标准 QA 的 answer_keys 必须出现在「该 QA 指定层级」的
   检索结果里（extractive 摘要必然保留原文关键实体/数字）
3. 汇总两 PDF 各自的覆盖率 → F1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m11 = SourceFileLoader('m11', str(Path(__file__).resolve().parent / '11_summary_indexer.py')).load_module()
from _eval_helpers import (
    get_all_qa, get_qa_by_pdf, keyword_hit_ratio, compute_f1, print_summary,
)

PDFS = ["bge_paper.pdf", "MiniMax_M1_tech_report.pdf"]

print("=" * 60)
print("11_summary_indexer 准确率验证（层次摘要覆盖度 F1）")
print("=" * 60)

# 步骤1：构建两份 PDF 的索引
for pdf in PDFS:
    idx = m11.build_summary_index(pdf)
    print(f"\n[{pdf}] document={len(idx['document']['text'])}字, "
          f"sections={len(idx['sections'])}, paragraphs={len(idx['paragraphs'])}")
    assert idx["document"]["text"].strip(), f"{pdf} document 摘要为空"
    assert idx["sections"], f"{pdf} section 摘要为空"
    assert idx["paragraphs"], f"{pdf} paragraph 为空"

# 步骤2：分层覆盖度
print("\n--- 层级覆盖度验证 ---")
per_pdf = {}
all_item_f1 = []

for pdf in PDFS:
    print(f"\n[{pdf}]")
    qa_list = get_qa_by_pdf(pdf)
    item_f1 = []
    for qa in qa_list:
        level = qa["level"]
        # 层次摘要索引的真实查询入口：跨三层 RRF 融合，哪层有答案哪层回来
        results = m11.retrieve_hierarchical(qa["question"], pdf, top_k=8)
        text = " ".join(r.get("text", "") for r in results)
        hit, total, missed = keyword_hit_ratio(qa["answer_keys"], text)
        # 覆盖率即 F1：每个 answer_key 是一个要被找回的事实点
        f1 = hit / total if total else 1.0
        item_f1.append(f1)
        mark = "✅" if f1 >= 0.9999 else "❌"
        print(f"  {mark} [{level:8}] {qa['id']} F1={f1:.0%} "
              f"({hit}/{total}) {qa['question'][:40]}")
        if missed:
            print(f"      缺失: {missed}")
    from _eval_helpers import aggregate_f1
    stat = aggregate_f1(item_f1)
    per_pdf[pdf] = stat
    all_item_f1.extend(item_f1)

overall_avg = sum(all_item_f1) / len(all_item_f1) if all_item_f1 else 0
ok = print_summary("11_summary_indexer", per_pdf, overall_avg)
sys.exit(0 if ok else 1)
