"""推理 Agent：拿检索回来的片段，让 LLM 生成答案。

做法是先抽取再生成（extraction-then-generation）：让 M3 在 <think> 里先把片段中
所有相关实体/数字/术语列出来，再据此写答案，这样关键事实不容易丢、也不会被
乱释义。答案带 [Passage N] 引用，术语尽量逐字保留。最后采样几次取忠实度最高的
那个（自一致性）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m05 = SourceFileLoader("m05", str(Path(__file__).resolve().parent / "05_llm_client.py")).load_module()
from _eval_helpers import _normalize_text


def _strip_think(text: str) -> str:
    """去掉 LLM 的 <think>...</think> 推理块（MiniMax-M3 思维链）。

    之前用了一个错误的正则（乱码片假名），导致 think 块没被剥离，吃光 max_tokens
    预算，真正答案被截断。这里是正确实现：剥离 <think>...</think>（含未闭合的
    裸 <think> 到结尾）。
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


_REASON_SYSTEM = (
    "You are a precise research assistant. Answer the question using ONLY the provided "
    "passages. Rules:\n"
    "1. Use only information from the passages — never hallucinate.\n"
    "2. CRITICAL — EXTRACTION FIRST: Before writing the answer, silently enumerate every "
    "specific entity name, model name, method name, dataset name, number, and technical "
    "term in the passages that is relevant to the question. Then write your answer so it "
    "mentions ALL of them. Copy each term VERBATIM from the passage — do NOT paraphrase, "
    "abbreviate, round, or simplify (e.g. keep '$534,700' not '$0.53M'; keep "
    "'Qwen3-235B' not 'a Qwen model'; keep 'hybrid-attention' not 'the attention type').\n"
    "3. Cite with [Passage N] after each claim.\n"
    "4. Be COMPLETE, not concise — for 'summarize' or list questions, enumerate every "
    "distinct relevant term from the passages. Omitting a relevant term is worse than "
    "being long.\n"
    "5. If the passages lack info, say: 'Based on the provided passages, I cannot answer.'"
)


def _build_prompt(question: str, passages: list[dict]) -> str:
    pblock = ""
    for i, p in enumerate(passages, 1):
        meta = p.get("metadata", p)
        pblock += (f"\n[Passage {i}] (source: {meta.get('source','')}, "
                   f"page: {meta.get('page','')})\n{p.get('text','')[:800]}\n")
    return f"Passages:{pblock}\n\nQuestion: {question}\n\nAnswer:"


def _extract_key_sentences(passages: list[dict], question: str,
                           max_sents: int = 10,
                           source_filter: str = None) -> list[str]:
    """从片段里抽与问题最相关的关键句，按 bge-m3 语义相似度排序。

    source_filter 给定时把全文档 chunk 也并进搜索池——top-k 有时召回不全，
    这样能补上漏掉的句子。
    """
    import re as _re
    # 搜索池 = 检索片段 + 全文档 chunk（source_filter 给定时）
    pool = list(passages)
    if source_filter:
        try:
            m11 = SourceFileLoader(
                "m11", str(Path(__file__).resolve().parent / "11_summary_indexer.py")
            ).load_module()
            idx = m11.load_summary_index(source_filter)
            for c in idx.get("paragraphs", []):
                t = c.get("text", "")
                if t and t.strip():
                    pool.append({"text": t})
        except Exception:
            pass

    all_sents = []
    for pi, p in enumerate(pool):
        text = p.get("text", "")
        for raw in _re.split(r"(?<=[\.\!\?])\s+", text):
            s = raw.strip()
            if len(s) < 15:
                continue
            all_sents.append((pi + 1, s))
    if not all_sents:
        return []

    # 语义排序
    try:
        m04 = SourceFileLoader(
            "m04", str(Path(__file__).resolve().parent / "04_embedder.py")
        ).load_module()
        qv = m04._embed([question], is_query=True)[0]
        svs = m04._embed([s for _, s in all_sents], is_query=False)
        scored = []
        for (pi, s), sv in zip(all_sents, svs):
            sim = sum(a * b for a, b in zip(qv, sv))
            scored.append((sim, pi, s))
        scored.sort(key=lambda x: -x[0])
    except Exception:
        scored = [(0.0, pi, s) for pi, s in all_sents]

    # 去重，取 top
    seen, out = set(), []
    for _, pi, s in scored:
        key = _normalize_text(s)[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append(f"[Passage {pi}] {s}")
        if len(out) >= max_sents:
            break
    return out


def reason(question: str, passages: list[dict], client=None,
           temperature: float = 0.0, source_filter: str = None) -> dict:
    """LLM 精炼生成 + 逐字证据兜底。"""
    if not passages:
        return {"question": question, "answer": "", "facts": [],
                "n_passages": 0, "grounded": False}
    prompt = _build_prompt(question, passages)
    if client is None:
        client = m05.create_client()
    try:
        resp = client.chat.completions.create(
            model=m05.LLM_MODEL,
            messages=[{"role": "system", "content": _REASON_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=1500,
        )
        answer = _strip_think(resp.choices[0].message.content)
    except Exception as e:
        answer = f"[推理失败] {type(e).__name__}: {str(e)[:200]}"

    grounded = "cannot answer" not in answer.lower()[:80]
    return {"question": question, "answer": answer,
            "facts": [], "n_passages": len(passages), "grounded": grounded}


def reason_with_self_consistency(question: str, passages: list[dict],
                                 client=None, n: int = 3,
                                 source_filter: str = None) -> dict:
    """自一致性：采样 n 次，取忠实度最高者。"""
    if not passages:
        return {"question": question, "answer": "", "facts": [],
                "n_passages": 0, "n_passes": 0, "grounded": False}
    if client is None:
        client = m05.create_client()

    drafts = []
    for i in range(max(1, n)):
        try:
            d = reason(question, passages, client=client,
                       temperature=(0.3 if i > 0 else 0.0))
            drafts.append(d)
        except Exception:
            continue
    if not drafts:
        return {"question": question, "answer": "", "facts": [],
                "n_passages": len(passages), "n_passes": 0, "grounded": False}

    # 取忠实度最高者（15 的检索式验证）
    try:
        m15 = SourceFileLoader(
            "m15", str(Path(__file__).resolve().parent / "15_reflection_agent.py")
        ).load_module()
        def faith(d):
            return m15.verify_answer(d["answer"], passages)["faithfulness"]
        drafts.sort(key=lambda d: (-faith(d), -len(d["answer"])))
    except Exception:
        drafts.sort(key=lambda d: -len(d["answer"]))

    best = drafts[0]
    return {"question": question, "answer": best["answer"],
            "facts": best.get("facts", []), "n_passages": len(passages),
            "n_passes": len(drafts), "grounded": best.get("grounded", False)}


def self_refine(*a, **k):
    return reason_with_self_consistency(*a, **k).get("answer", "")


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="推理 Agent")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--passages-json")
    args = ap.parse_args()
    if not args.question:
        print("用法: python3 14_reasoning_agent.py 你的问题 --passages-json p.json")
        sys.exit(1)
    q = " ".join(args.question)
    passages = json.load(open(args.passages_json)) if args.passages_json else []
    r = reason_with_self_consistency(q, passages)
    print(r["answer"])
