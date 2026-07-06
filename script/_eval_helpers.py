"""评测共享工具：加载金标准数据集 + 通用 F1/关键词命中计算。

被各 test_XX_accuracy.py 复用，保证两份 PDF 的评测口径一致。
本模块只提供度量函数和数据加载，不参与被测算法，也不被算法脚本 import。
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_DATASET_PATH = Path(__file__).resolve().parent / "18_eval_dataset.json"


def load_eval_dataset() -> dict:
    """加载 18_eval_dataset.json。"""
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_qa_by_pdf(pdf: str) -> list[dict]:
    """取某份 PDF 的所有 QA。"""
    ds = load_eval_dataset()
    return [q for q in ds["qa"] if q["pdf"] == pdf]


def get_all_qa() -> list[dict]:
    return load_eval_dataset()["qa"]


def get_reflection_cases() -> list[dict]:
    return load_eval_dataset()["reflection_cases"]


# ── 归一化与命中 ────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)


def _normalize_text(s: str) -> str:
    """归一化：小写 + 去 unicode 标记（é→e） + 折叠多空格。

    顺带处理两类 PDF 排版噪声：
    - 数字内部逗号去掉（534,700 → 534700），便于数字对齐。
    - 行内断字连字符去掉（en-\\ncoder → encoder / self-\\nknowledge → selfknowledge /
      Mixture-of-Experts → MixtureofExperts）。对 text 和 answer_key 同时生效，
      两侧一致，子串匹配不受 PDF 排版断字干扰。
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    # 去掉字母之间（含跨行）的连字符：en-\ncoder → encoder
    s = re.sub(r"(?<=[A-Za-z])-\s*\n?\s*(?=[A-Za-z])", "", s)
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def keyword_hit_ratio(keywords: list[str], text: str) -> tuple[int, int, list[str]]:
    """检查 keywords 在 text 中的命中（大小写/变音/标点无关，子串匹配）。

    返回 (命中数, 总数, 未命中关键词列表)。
    """
    norm_text = _normalize_text(text)
    hit = 0
    missed = []
    for kw in keywords:
        if not kw:
            continue
        if _normalize_text(kw) in norm_text:
            hit += 1
        else:
            missed.append(kw)
    return hit, len([k for k in keywords if k]), missed


def compute_f1(predicted: set, gold: set) -> tuple[float, float, float]:
    """精确集合 F1。"""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


def aggregate_f1(per_item_f1: list[float]) -> dict:
    """聚合每条 QA 的 F1 出整体统计。"""
    n = len(per_item_f1)
    if n == 0:
        return {"avg_f1": 0.0, "pass_rate": 0.0, "n": 0}
    avg = sum(per_item_f1) / n
    pass_rate = sum(1 for f in per_item_f1 if f >= 0.9999) / n
    return {"avg_f1": avg, "pass_rate": pass_rate, "n": n}


def print_summary(title: str, per_pdf: dict, overall_avg: float):
    """统一打印两 PDF + 总体 F1 汇总。

    per_pdf: {pdf_name: {"avg_f1":float, "pass_rate":float, "n":int}}
    """
    line = "=" * 60
    print("\n" + line)
    print(f"{title} 汇总")
    print(line)
    all_pass = True
    for pdf, stat in per_pdf.items():
        status = "✅" if stat["avg_f1"] >= 0.9999 else "❌"
        print(f"  {pdf}: 平均 F1={stat['avg_f1']:.2%}, "
              f"通过率={stat['pass_rate']:.0%} ({stat['n']} 题) {status}")
        if stat["avg_f1"] < 0.9999:
            all_pass = False
    overall = "✅" if all_pass and overall_avg >= 0.9999 else "❌"
    print(f"  总体平均 F1={overall_avg:.2%} {overall}")
    if all_pass and overall_avg >= 0.9999:
        print(f"  ✅ {title} 两 PDF F1 均 100% PASS")
    else:
        print(f"  ❌ {title} 未达标")
    print(line)
    return all_pass
