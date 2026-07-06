"""
RAG 问答：检索 + 生成一整套。

最早只有向量检索一路，召回还行但专有名词（模型名、数字这类）老对不上，
于是加了 BM25 关键词路、两路用 RRF 融合；后来又发现 query 太短 chunk 太长，
向量相似度吃亏，就加了 HyDE（让 M3 先编一段假设答案再拿去检索）；最后再让
M3 对候选做一次 LLM 重排。每加一层，Hit@5 大概各提 5~8 个点。

主流程在 retrieve() + answer()，看这俩就够了。
"""

# 向量+BM25+RRF+HyDE 混合检索

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m04 = SourceFileLoader('m04', str(Path(__file__).resolve().parent / '04_embedder.py')).load_module()
m05 = SourceFileLoader('m05', str(Path(__file__).resolve().parent / '05_llm_client.py')).load_module()

from config import RETRIEVE_TOP_K


def _strip_think(text: str) -> str:
    """剔除 MiniMax-M3 的 <think>...</think> 推理块，保留最终答案"""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

# 复用 04 已经定义的 chunks 目录
CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / 'output' / 'chunks'
BM25_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / 'output' / 'bm25_index.pkl'



# BM25 索引
def _tokenize(text: str) -> list[str]:
    """简单分词：英文按空格/标点切，中文按字切"""
    import re
    # 英文：小写化，按非字母数字切
    en_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower()) # 
    # 中文：按单字切
    cn_tokens = re.findall(r'[\u4e00-\u9fff]', text)
    return en_tokens + cn_tokens


class BM25Index:
    """BM25 索引（基于 rank_bm25.BM25Okapi）"""

    def __init__(self):
        self.bm25 = None
        self.docs = []      # 原始chunk dict
        self.tokenized = [] # 分词后的列表

    def build(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self.docs = chunks
        self.tokenized = [_tokenize(c.get('text', '')) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        if not self.bm25:
            return []
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # 取top_k索引
        import heapq
        top_idx = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])
        results = []
        for rank, idx in enumerate(top_idx):
            if scores[idx] <= 0:
                continue
            doc = self.docs[idx]
            results.append({
                'rank': rank + 1,
                'text': doc.get('text', ''),
                'metadata': {
                    'source': doc.get('source', ''),
                    'page': int(doc.get('page', 0) or 0),
                    'section': doc.get('section', ''),
                    'chunk_type': doc.get('content_type', doc.get('type', 'text')),
                    'chunk_index': int(doc.get('chunk_index', 0) or 0),
                },
                'score': float(scores[idx]),
            })
        return results

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({'docs': self.docs, 'tokenized': self.tokenized}, f)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        from rank_bm25 import BM25Okapi
        self.docs = data['docs']
        self.tokenized = data['tokenized']
        self.bm25 = BM25Okapi(self.tokenized)
        return True


_bm25_index = None


def get_bm25_index() -> BM25Index:
    """获取 BM25 索引（带持久化缓存）"""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    _bm25_index = BM25Index()
    # 先尝试加载缓存
    if _bm25_index.load(BM25_INDEX_PATH):
        # 校验：缓存里的 chunks 数应该和磁盘上 JSON 总数一致
        json_total = sum(1 for _ in CHUNKS_DIR.glob('*.json'))
        cached_sources = set(d.get('source', '') for d in _bm25_index.docs)
        if len(cached_sources) >= json_total:
            return _bm25_index
        # 否则重建

    # 从 chunks JSON 重建
    all_chunks = []
    for json_path in sorted(CHUNKS_DIR.glob('*.json')):
        with open(json_path, 'r', encoding='utf-8') as f:
            all_chunks.extend(json.load(f))
    _bm25_index.build(all_chunks)
    _bm25_index.save(BM25_INDEX_PATH)
    return _bm25_index


def rebuild_bm25_index():
    """强制重建 BM25 索引"""
    global _bm25_index
    all_chunks = []
    for json_path in sorted(CHUNKS_DIR.glob('*.json')):
        with open(json_path, 'r', encoding='utf-8') as f:
            all_chunks.extend(json.load(f))
    _bm25_index = BM25Index()
    _bm25_index.build(all_chunks)
    _bm25_index.save(BM25_INDEX_PATH)
    return _bm25_index


# RRF 融合

def rrf_fuse(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 融合多路检索结果

    score(d) = sum over rankings: 1 / (k + rank_in_that_ranking)
    k=60 是 RRF 原论文的默认值，试过 30 / 100 区别不大，就没动。
    """
    rrf_scores = {}
    doc_map = {}
    for results in result_lists:
        for r in results:
            text = r['text']
            rank = r['rank']
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)
            if text not in doc_map:
                doc_map[text] = r

    fused = []
    for idx, (text, score) in enumerate(sorted(rrf_scores.items(), key=lambda x: -x[1])):
        item = dict(doc_map[text])
        item['rrf_score'] = score
        item['rank'] = idx + 1
        fused.append(item)
    return fused


# HyDE

_HYDE_SYSTEM = """You are a helpful assistant. Given a question, generate a short hypothetical answer (150-200 words) as if you were answering from a research paper. Use academic style. Do not say 'I don't know'. Just write the hypothetical content."""

def generate_hyde(query: str) -> str:
    """让 M3 生成假设答案，用于二次向量检索"""
    try:
        return m05.simple_chat(query, system_message=_HYDE_SYSTEM)
    except Exception as e:
        print(f"[HyDE 跳过] {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
        return ""


# LLM 重排序

def rerank_with_llm(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """让 M3 对候选chunk打分（0-10），按分数排序取 Top-K"""
    if len(candidates) <= top_k:
        return candidates[:top_k]

    # 构造打分prompt
    context = ""
    for i, c in enumerate(candidates):
        context += f"\n[{i}] {c['text'][:300]}\n"
    prompt = f"""You are a relevance judge. Score each passage (0-10) for answering the question.
Question: {query}
Passages:{context}

Output format: only JSON list of {{"index": int, "score": int}}, no other text. Example:
[{{"index": 0, "score": 9}}, {{"index": 1, "score": 3}}]

Score criteria:
- 10: directly answers the question
- 7-9: relevant and useful
- 4-6: partially relevant
- 1-3: barely relevant
- 0: irrelevant"""

    try:
        resp = m05.simple_chat(prompt)
        # 剔除 <think>...</think> 推理块，再提取 JSON 数组
        resp_clean = _strip_think(resp)
        m = re.search(r'\[.*\]', resp_clean, re.DOTALL)
        if not m:
            # 备用：逐个提取 {"index":int,"score":int}
            pairs = re.findall(r'\{\s*"index"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*(\d+)\s*\}', resp_clean)
            if not pairs:
                return candidates[:top_k]
            scores = [{"index": int(i), "score": int(s)} for i, s in pairs]
        else:
            scores = json.loads(m.group(0))
        # 把分数加到candidates
        scored = []
        for s in scores:
            idx = s.get('index')
            if 0 <= idx < len(candidates):
                c = dict(candidates[idx])
                c['rerank_score'] = s.get('score', 0)
                scored.append(c)
        # 按分数降序
        scored.sort(key=lambda x: -x.get('rerank_score', 0))
        # 如果某些候选没被评分，追加到末尾
        scored_ids = set(s.get('index') for s in scores)
        for i, c in enumerate(candidates):
            if i not in scored_ids:
                scored.append(c)
        return scored[:top_k]
    except Exception as e:
        print(f"[重排序失败，回退RRF顺序] {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
        return candidates[:top_k]


# 主检索流程

def retrieve(query: str, top_k: int = RETRIEVE_TOP_K,
             use_bm25: bool = True, use_hyde: bool = True, use_rerank: bool = True) -> list[dict]:
    """多路召回 + RRF融合 + HyDE + LLM重排序"""
    candidate_k = max(top_k * 3, 15)

    # 1. 向量检索
    vec_results = m04.query(query, top_k=candidate_k)

    # 2. BM25 检索
    bm25_results = []
    if use_bm25:
        try:
            bm25 = get_bm25_index()
            bm25_results = bm25.search(query, top_k=candidate_k)
        except Exception as e:
            print(f"[BM25 跳过] {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)

    # 3. HyDE：生成假设答案，再做一次向量检索
    hyde_results = []
    if use_hyde:
        hyde_text = generate_hyde(query)
        if hyde_text:
            hyde_results = m04.query(hyde_text, top_k=candidate_k)

    # 4. RRF 融合
    all_lists = [vec_results]
    if bm25_results:
        all_lists.append(bm25_results)
    if hyde_results:
        all_lists.append(hyde_results)
    fused = rrf_fuse(all_lists, k=60)

    # 5. LLM 重排序
    if use_rerank and len(fused) > top_k:
        final = rerank_with_llm(query, fused[:candidate_k], top_k=top_k)
    else:
        final = fused[:top_k]

    return final


# 答案生成

_ANSWER_SYSTEM = """You are a precise research assistant. Answer the question based ONLY on the provided passages. Rules:
1. Use only information from the passages - do not hallucinate.
2. If the passages don't contain enough information, say "Based on the provided passages, I cannot answer this question."
3. Cite sources like [Passage 1] after relevant claims.
4. Be concise and direct."""


def build_prompt(query: str, contexts: list[dict]) -> str:
    passages = ""
    for i, c in enumerate(contexts):
        meta = c.get('metadata', {})
        passages += f"\n[Passage {i+1}] (source: {meta.get('source','')}, page: {meta.get('page','')}, section: {meta.get('section','')})\n{c['text']}\n"
    return f"Passages:{passages}\n\nQuestion: {query}\n\nAnswer:"


def answer(query: str, top_k: int = RETRIEVE_TOP_K,
           use_bm25=True, use_hyde=True, use_rerank=True) -> dict:
    """端到端RAG问答"""
    t0 = time.time()
    # 检索
    contexts = retrieve(query, top_k=top_k,
                        use_bm25=use_bm25, use_hyde=use_hyde, use_rerank=use_rerank)
    t_retrieve = time.time() - t0

    # 生成
    prompt = build_prompt(query, contexts)
    t1 = time.time()
    try:
        ans = m05.simple_chat(prompt, system_message=_ANSWER_SYSTEM)
        ans = _strip_think(ans)  # 剔除 <think> 推理块
    except Exception as e:
        ans = f"[生成失败] {type(e).__name__}: {str(e)[:200]}"
    t_generate = time.time() - t1

    return {
        'query': query,
        'answer': ans,
        'sources': contexts,
        'time_retrieve': round(t_retrieve, 2),
        'time_generate': round(t_generate, 2),
        'time_total': round(time.time() - t0, 2),
        'flags': {'bm25': use_bm25, 'hyde': use_hyde, 'rerank': use_rerank},
    }


# CLI 入口

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description='RAG问答')
    ap.add_argument('query', nargs='*', help='查询语句')
    ap.add_argument('--no-bm25', action='store_true')
    ap.add_argument('--no-hyde', action='store_true')
    ap.add_argument('--no-rerank', action='store_true')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--rebuild-bm25', action='store_true')
    args = ap.parse_args()

    if args.rebuild_bm25:
        idx = rebuild_bm25_index()
        print(f"BM25索引重建: {len(idx.docs)} 条chunk")

    if not args.query:
        print("用法: python3 06_rag_query.py 你的问题")
        print("示例: python3 06_rag_query.py what is M3-Embedding")
        exit()

    q = ' '.join(args.query)
    print(f"\n=== Query ===\n{q}\n")
    result = answer(q, top_k=args.top_k,
                    use_bm25=not args.no_bm25,
                    use_hyde=not args.no_hyde,
                    use_rerank=not args.no_rerank)

    print(f"\n=== Sources (flags: bm25={result['flags']['bm25']}, hyde={result['flags']['hyde']}, rerank={result['flags']['rerank']}) ===")
    for i, src in enumerate(result['sources']):
        meta = src['metadata']
        rrf = src.get('rrf_score', '?')
        rerank = src.get('rerank_score', '-')
        print(f"  [{i+1}] rrf={rrf:.4f} rerank={rerank} [{meta.get('source','')[:30]}] p{meta.get('page','')} {meta.get('section','')[:30]}")
        print(f"      {src['text'][:120]}...")

    print(f"\n=== Answer ===\n{result['answer']}")
    print(f"\n=== 耗时 ===")
    print(f"  检索: {result['time_retrieve']}s, 生成: {result['time_generate']}s, 总: {result['time_total']}s")
