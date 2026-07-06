"""
RAG 检索质量批量测试脚本
- 预设一组问题和"期望命中"的关键词
- 跑全开/关BM25/关HyDE/关重排序 4种模式
- 统计 Top-K 命中率、平均RRF分数、平均重排序分数
- 用法: env/bin/python3 script/test_rag_quality.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m06 = SourceFileLoader('m06', str(Path(__file__).resolve().parent / '06_rag_query.py')).load_module()


# 预设测试集：问题 + 期望Top5里应该出现的关键词（命中任意一个就算对）
# 分两类：简单题（主题明确）+ 难题（细节/陷阱/跨章节）
TEST_CASES = [
    # === 简单题（基线） ===
    {
        'q': 'What is M3-Embedding?',
        'expect_source': 'bge_paper.pdf',
        'expect_keywords': ['multi-lingual', 'multi-functionality', 'multi-granularity', 'M3-Embedding'],
    },
    {
        'q': 'What is CISPO algorithm?',
        'expect_source': 'MiniMax_M1_tech_report.pdf',
        'expect_keywords': ['CISPO', 'importance sampling', 'reinforcement learning', 'RL'],
    },
    {
        'q': 'How does lightning attention work?',
        'expect_source': 'MiniMax_M1_tech_report.pdf',
        'expect_keywords': ['lightning attention', 'linear attention', 'hybrid', 'efficient'],
    },
    {
        'q': 'What languages does BGE-M3 support?',
        'expect_source': 'bge_paper.pdf',
        'expect_keywords': ['100', 'languages', 'multi-lingual', 'cross-lingual'],
    },
    {
        'q': 'What is the model size of MiniMax-M1?',
        'expect_source': 'MiniMax_M1_tech_report.pdf',
        'expect_keywords': ['45.9 billion', 'parameters', 'activated', 'size'],
    },
    {
        'q': 'How is M3-Embedding trained?',
        'expect_source': 'bge_paper.pdf',
        'expect_keywords': ['self-knowledge distillation', 'batching', 'training', 'data curation'],
    },

    # === 难题（细节/陷阱/跨章节） ===
    {
        'q': 'How many H800 GPUs were used to train MiniMax-M1 and what was the rental cost?',
        'expect_source': 'MiniMax_M1_tech_report.pdf',
        'expect_keywords': ['512', 'H800', '0.53M', '3 weeks'],
    },
    {
        'q': 'What is the key difference between CISPO and traditional PPO/GRPO?',
        'expect_source': 'MiniMax_M1_tech_report.pdf',
        'expect_keywords': ['clips importance sampling', 'token updates', 'trust region', 'clip'],
    },
    {
        'q': 'What is the final loss function of M3-Embedding training?',
        'expect_source': 'bge_paper.pdf',
        'expect_keywords': ['Lfinal', 'linear combination', "knowledge distillation", 'L\''],
    },
    {
        'q': 'What problem does the Efficient Batching strategy in M3-Embedding solve?',
        'expect_source': 'bge_paper.pdf',
        'expect_keywords': ['long sequences', 'batch size', 'throughput', 'truncate'],
    },
    {
        'q': 'What was MiniMax-M1\'s accuracy score on AIME 2024?',
        'expect_source': 'MiniMax_M1_tech_report.pdf',
        'expect_keywords': ['86.0', 'AIME', 'open-weight', 'second'],
    },
    {
        'q': 'How many samples are in the general RL dataset of MiniMax-M1?',
        'expect_source': 'MiniMax_M1_tech_report.pdf',
        'expect_keywords': ['25K', 'complex samples', 'RL dataset', 'categorized'],
    },
    {
        'q': 'Why does M3-Embedding sparse retrieval outperform BM25?',
        'expect_source': 'bge_paper.pdf',
        'expect_keywords': ['learned weights', 'sparse', 'BM25', 'tokenizer'],
    },
    {
        'q': 'What is the MCLS method proposed for in M3-Embedding?',
        'expect_source': 'bge_paper.pdf',
        'expect_keywords': ['MCLS', 'Multiple CLS', 'long-context', 'fine-tuning'],
    },
]


def evaluate_one(q: str, expect_source: str, expect_keywords: list[str],
                 use_bm25=True, use_hyde=True, use_rerank=True, top_k=5):
    """返回 (hit_top1, hit_top5, keyword_hits)"""
    try:
        results = m06.retrieve(q, top_k=top_k,
                                use_bm25=use_bm25, use_hyde=use_hyde, use_rerank=use_rerank)
    except Exception as e:
        return {'error': str(e)[:100]}

    # Top-1 命中：第一个chunk的source是否正确
    top1_hit = results[0]['metadata'].get('source', '') == expect_source if results else False
    # Top-5 命中：Top-5里是否有期望source
    top5_hit = any(r['metadata'].get('source', '') == expect_source for r in results[:top_k])
    # 关键词命中：Top-5文本里出现几个期望关键词
    all_text = ' '.join(r['text'].lower() for r in results[:top_k])
    kw_hits = sum(1 for kw in expect_keywords if kw.lower() in all_text)
    kw_total = len(expect_keywords)

    return {
        'top1_hit': top1_hit,
        'top5_hit': top5_hit,
        'kw_hits': kw_hits,
        'kw_total': kw_total,
        'sources': [(r['metadata'].get('source', '')[:25], r['metadata'].get('page', '')) for r in results[:top_k]],
    }


def run_mode(title: str, **flags):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    stats = {'top1': 0, 'top5': 0, 'kw_total': 0, 'kw_hits': 0, 'n': 0}
    for tc in TEST_CASES:
        r = evaluate_one(tc['q'], tc['expect_source'], tc['expect_keywords'], **flags)
        if 'error' in r:
            print(f"  [ERROR] {tc['q'][:40]}: {r['error']}")
            continue
        stats['n'] += 1
        stats['top1'] += int(r['top1_hit'])
        stats['top5'] += int(r['top5_hit'])
        stats['kw_hits'] += r['kw_hits']
        stats['kw_total'] += r['kw_total']
        mark1 = '✓' if r['top1_hit'] else '✗'
        mark5 = '✓' if r['top5_hit'] else '✗'
        print(f"  [{mark1}{mark5}] kw={r['kw_hits']}/{r['kw_total']} {tc['q'][:45]}")
        print(f"        Top5: {r['sources']}")
    if stats['n']:
        print(f"\n  汇总: Top1={stats['top1']}/{stats['n']} ({100*stats['top1']/stats['n']:.0f}%)  "
              f"Top5={stats['top5']}/{stats['n']} ({100*stats['top5']/stats['n']:.0f}%)  "
              f"关键词={stats['kw_hits']}/{stats['kw_total']} ({100*stats['kw_hits']/max(stats['kw_total'],1):.0f}%)")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=int, default=4, choices=[1, 2, 3, 4],
                    help='1=纯向量, 2=+BM25+RRF, 3=+HyDE, 4=全开(默认)')
    args = ap.parse_args()

    print(f"RAG 检索质量测试 (模式{args.mode})")
    print(f"测试集: {len(TEST_CASES)} 个问题\n")

    if args.mode == 1:
        run_mode('模式1: 纯向量基线',
                 use_bm25=False, use_hyde=False, use_rerank=False)
    elif args.mode == 2:
        run_mode('模式2: 向量 + BM25 + RRF',
                 use_bm25=True, use_hyde=False, use_rerank=False)
    elif args.mode == 3:
        run_mode('模式3: 向量 + BM25 + RRF + HyDE',
                 use_bm25=True, use_hyde=True, use_rerank=False)
    elif args.mode == 4:
        run_mode('模式4: 全开 (向量+BM25+RRF+HyDE+重排序)',
                 use_bm25=True, use_hyde=True, use_rerank=True)
