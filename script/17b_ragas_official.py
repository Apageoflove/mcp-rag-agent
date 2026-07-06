"""ragas 原版指标评估（对照 17_eval_framework 的自定义近似实现）。

用 ragas 0.2.15 库的三个核心指标：
  - context_recall（LLM-as-judge，逐条判断 gold fact 是否被 context 支撑）
  - faithfulness（LLM-as-judge，逐断言判断是否被 context 蕴含）
  - answer_relevancy（从 answer 反生成 question，算与原 question 相似度）

LLM judge = MiniMax-M3（OpenAI 兼容，<think> 块在 wrapper 里剥离）

context_recall / faithfulness 用 ragas 0.2.15 原版（LLM-as-judge）。
answer_relevancy 用 bge-m3 embedding cosine（保留 17 的近似实现，因 ragas_env
无 torch/sentence-transformers，且该指标本质就是 embedding 相似度）。

数据来自 _collect_answers.py 预存的 answer+contexts（避免 env/ragas_env 的
numpy 版本冲突——ragas 评分只依赖 ragas_env，不碰 env）。
输出与 17 的自定义实现对照，看 LLM-as-judge 和关键词命中差多少。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# ragas 用单独的虚拟环境（ragas + 兼容版 langchain 全套），跟主 env 的 numpy
# 版本冲突，所以分开。路径从 RAGAS_ENV 环境变量读，没配就跳过（后面 import ragas 会报错）。
RAGAS_ENV = os.environ.get("RAGAS_ENV", "")
if RAGAS_ENV:
    sys.path.insert(0, RAGAS_ENV)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader

# 复用项目配置（只读常量，不触发 numpy/sklearn）
m01 = SourceFileLoader("m01", str(Path(__file__).resolve().parent / "01_config.py")).load_module()
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MODEL_DIR

ANSWERS_JSON = Path(__file__).resolve().parent.parent.parent / "output" / "answers_for_ragas.json"


def _keyword_hit_ratio(keywords, text):
    """轻量本地版（不 import _eval_helpers，避免拖 env 的 numpy）。"""
    import unicodedata
    def norm(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"(?<=\d),(?=\d)", "", s)
        s = re.sub(r"[^\w]+", " ", s.lower()).strip()
        return s
    nt = norm(text)
    hit = sum(1 for k in keywords if k and norm(k) in nt)
    return hit, len([k for k in keywords if k])


# ── MiniMax-M3 LLM wrapper（关闭 thinking，直接输出干净 JSON）─────────
def _make_ragas_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper

    class _MiniMaxClean(ChatOpenAI):
        """ChatOpenAI 指向 MiniMax，关闭 thinking 模式。

        MiniMax-M3 默认输出 <think>...</think> 推理块，会吃掉 max_tokens 导致
        ragas 要的 JSON 被截断。extra_body={'thinking':{'type':'disabled'}} 关掉
        thinking，让 LLM 直接吐干净 JSON（已验证可用）。
        """

        def _generate(self, *a, **kw):
            return super()._generate(*a, **kw)

        async def _agenerate(self, *a, **kw):
            return await super()._agenerate(*a, **kw)

    llm = _MiniMaxClean(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.0,
        max_tokens=4096,
        max_retries=2,
        timeout=180,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return LangchainLLMWrapper(llm)


def _make_ragas_embeddings():
    """answer_relevancy 用的 embedding。ragas_env 无 torch，用 env 预算好的
    bge-m3 cosine（见 _compute_relevancy.py）。这里返回 None 占位。"""
    return None


def _compute_relevancy_cosine(question: str, answer: str) -> float:
    """用 env 的 bge-m3 算 question-answer cosine（answer_relevancy 近似）。

    在 ragas_env 里没法 import env 的 sentence-transformers（numpy 冲突），
    所以这个函数需要用 env 跑。main() 里会 subprocess 调 _compute_relevancy.py。
    """
    return 0.0


async def _score(metric, sample) -> float:
    try:
        return float(await metric.single_turn_ascore(sample))
    except Exception as e:
        print(f"    [metric error] {type(e).__name__}: {str(e)[:120]}")
        return 0.0


async def _eval_all(samples, ragas_llm):
    from ragas.metrics import context_recall, faithfulness
    from ragas.dataset_schema import SingleTurnSample

    # 注入 LLM 到 metric 实例（answer_relevancy 需 embedding，ragas_env 无 torch，
    # 故该指标用 env 的 bge-m3 单独算，见 _add_relevancy.py）
    for m in (context_recall, faithfulness):
        m.llm = ragas_llm

    results = []
    for i, s in enumerate(samples, 1):
        t0 = time.time()
        if "error" in s:
            print(f"[{i}/{len(samples)}] {s['id']} skip (collect error)")
            results.append({"id": s["id"], "pdf": s["pdf"], "error": s["error"]})
            continue
        sample = SingleTurnSample(
            user_input=s["question"],
            retrieved_contexts=s["contexts"],
            response=s["answer"],
            reference=", ".join(s.get("answer_keys", [])),
        )
        cr = await _score(context_recall, sample)
        fa = await _score(faithfulness, sample)
        a_hit, a_total = _keyword_hit_ratio(s.get("answer_keys", []), s["answer"])
        completeness = a_hit / a_total if a_total else 1.0
        latency = round(time.time() - t0, 2)

        print(f"[{i}/{len(samples)}] {s['id']} {s['pdf'][:18]:18} "
              f"CR={cr:.2f} FA={fa:.2f} "
              f"custF1={completeness:.2f} t={latency}s")
        results.append({
            "id": s["id"], "pdf": s["pdf"], "type": s.get("type", ""),
            "question": s["question"],
            "answer": s["answer"],
            "answer_keys": s.get("answer_keys", []),
            "context_recall": round(cr, 3),
            "faithfulness": round(fa, 3),
            "custom_completeness": round(completeness, 3),
            "latency": latency,
        })
    return results


def main():
    with open(ANSWERS_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print("=" * 64)
    print(f"ragas 0.2.15 原版指标评估（{len(samples)} 题）")
    print(f"LLM judge = {LLM_MODEL}（<think> 块剥离）")
    print(f"指标 = context_recall + faithfulness（LLM-as-judge）")
    print(f"      answer_relevancy 单独用 env 的 bge-m3 算（_add_relevancy.py）")
    print(f"数据 = {ANSWERS_JSON.name}")
    print("=" * 64)

    ragas_llm = _make_ragas_llm()
    print("[init] ragas LLM wrapper OK")

    results = asyncio.run(_eval_all(samples, ragas_llm))

    # 聚合
    def avg(key):
        v = [r[key] for r in results if "error" not in r and key in r]
        return round(sum(v) / len(v), 4) if v else 0.0

    per_pdf = {}
    for pdf in sorted(set(r["pdf"] for r in results)):
        items = [r for r in results if r.get("pdf") == pdf and "error" not in r]
        per_pdf[pdf] = {
            "n": len(items),
            "context_recall": round(sum(r["context_recall"] for r in items) / len(items), 4) if items else 0,
            "faithfulness": round(sum(r["faithfulness"] for r in items) / len(items), 4) if items else 0,
            "custom_completeness": round(sum(r["custom_completeness"] for r in items) / len(items), 4) if items else 0,
        }

    report = {
        "per_item": results,
        "per_pdf": per_pdf,
        "overall": {
            "n": len(results),
            "context_recall": avg("context_recall"),
            "faithfulness": avg("faithfulness"),
            "custom_completeness": avg("custom_completeness"),
        },
    }

    # 渲染
    print("\n" + "=" * 64)
    print("ragas 原版指标报告（context_recall + faithfulness，LLM-as-judge）")
    print("answer_relevancy 见后续 _add_relevancy.py 用 bge-m3 单独算")
    print("=" * 64)
    for pdf, s in per_pdf.items():
        print(f"\n[{pdf}] ({s['n']} 题)")
        print(f"  Context Recall    (ragas, LLM-judge) : {s['context_recall']:.2%}")
        print(f"  Faithfulness      (ragas, LLM-judge) : {s['faithfulness']:.2%}")
        print(f"  Custom Completeness(关键词命中对照)   : {s['custom_completeness']:.2%}")
    o = report["overall"]
    print("\n" + "-" * 64)
    print("总体")
    print(f"  Context Recall    (ragas) : {o['context_recall']:.2%}")
    print(f"  Faithfulness      (ragas) : {o['faithfulness']:.2%}")
    print(f"  Custom Completeness(对照) : {o['custom_completeness']:.2%}")
    print("=" * 64)

    out = Path(__file__).resolve().parent.parent.parent / "output" / "ragas_official_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
