"""Cross-Encoder 重排序（bge-reranker-v2-m3）。

bge-reranker-v2-m3 是基于 xlm-roBERTa 的 cross-encoder，对 (query, passage) 对
做交叉注意力打分，精度比双塔向量检索高不少，放在向量召回之后做精排。

用法：rerank(query, candidates, top_k) → 重排序后的 top_k 候选。
"""

# bge-reranker-v2-m3 cross-encoder

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_RERANKER_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "models" / "bge-reranker-v2-m3"
)

_tokenizer = None
_model = None


def _get_reranker():
    """懒加载 cross-encoder 模型（首次调用加载，后续复用）。"""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(_RERANKER_MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(
        _RERANKER_MODEL_PATH, dtype=torch.float16
    )
    _model.eval()
    print(f"  [reranker] bge-reranker-v2-m3 加载完成", file=sys.stderr)
    return _tokenizer, _model


def rerank(query: str, candidates: list[dict], top_k: int = 5,
           max_length: int = 512) -> list[dict]:
    """用 cross-encoder 对候选片段重排序，返回 top_k。

    Args:
        query: 用户问题
        candidates: 候选片段列表 [{text, metadata, score, ...}, ...]
        top_k: 返回数量
        max_length: 每对最大 token 长度

    Returns:
        重排序后的 top_k 候选（加 rerank_score 字段）
    """
    if not candidates:
        return []

    try:
        import torch
        tok, model = _get_reranker()

        # 构造 (query, passage) 对
        pairs = []
        for c in candidates:
            text = c.get("text", "")[:1000]  # 截断长文本
            pairs.append([query, text])

        with torch.no_grad():
            inputs = tok(pairs, padding=True, truncation=True,
                         return_tensors="pt", max_length=max_length)
            logits = model(**inputs).logits.squeeze(-1)
            # 用原始 logit 作为分数（排序更可靠；sigmoid 后值域太窄）
            scores = logits.tolist()

        # 按 rerank 分数排序
        scored = []
        for c, s in zip(candidates, scores):
            c2 = dict(c)
            c2["rerank_score"] = float(s)
            scored.append(c2)
        scored.sort(key=lambda x: -x["rerank_score"])
        return scored[:top_k]

    except Exception as e:
        print(f"  [reranker] 重排序失败，回退原顺序: {type(e).__name__}: {e}",
              file=sys.stderr)
        return candidates[:top_k]


if __name__ == "__main__":
    # 自检
    cands = [
        {"text": "ResNet introduces residual learning for training very deep networks."},
        {"text": "ImageNet is a large-scale visual recognition dataset."},
        {"text": "The Transformer architecture uses self-attention mechanisms."},
    ]
    result = rerank("What is ResNet?", cands, top_k=2)
    for r in result:
        print(f"  rerank={r['rerank_score']:.4f} | {r['text'][:60]}")
