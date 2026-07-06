"""MCP 工具服务：把检索 / 图谱 / 多模态 / 联网搜索封装成 4 个 MCP tool 给 Agent 调。

graph_query 和 web_search 现在还是占位（图谱在 08-10 做、联网搜索预留），
vector_search 和 vlm_analysis 是已经能跑的。用 FastMCP 范式注册，stdio 传输。
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from importlib.machinery import SourceFileLoader
m04 = SourceFileLoader('m04', str(Path(__file__).resolve().parent / '04_embedder.py')).load_module()
m05 = SourceFileLoader('m05', str(Path(__file__).resolve().parent / '05_llm_client.py')).load_module()
m06 = SourceFileLoader('m06', str(Path(__file__).resolve().parent / '06_rag_query.py')).load_module()

from config import RETRIEVE_TOP_K, LLM_MODEL

# ── MCP Server 实例 ──────────────────────────────────────────────
mcp = FastMCP(
    name="MultiModal-RAG-Server",
    instructions="多模态 RAG 增强智能体系统 — 提供向量检索、知识图谱查询、图片分析和联网搜索能力",
)


# Tool 1: vector_search — 向量 + BM25 + HyDE + LLM 重排序

@mcp.tool(
    name="vector_search",
    annotations={
        "title": "向量混合检索",
        "readOnlyHint": True,
        "destructiveHint": False,
    },
)
def vector_search(
    query: str,
    top_k: int = RETRIEVE_TOP_K,
    use_bm25: bool = True,
    use_hyde: bool = True,
    use_rerank: bool = True,
) -> str:
    """多路混合检索：向量语义 + BM25 关键词 + HyDE 假设答案 + LLM 重排序。

    Args:
        query: 用户查询文本（中文/英文均可）
        top_k: 返回结果数量，默认 5
        use_bm25: 是否启用 BM25 关键词检索
        use_hyde: 是否启用 HyDE 假设答案增强
        use_rerank: 是否启用 LLM 重排序

    Returns:
        JSON 字符串，包含检索到的文档片段及其元数据（来源、页码、章节、分数）
    """
    try:
        results = m06.retrieve(
            query, top_k=top_k,
            use_bm25=use_bm25, use_hyde=use_hyde, use_rerank=use_rerank,
        )
        # 精简输出：只保留检索 Agent 需要的关键字段
        simplified = []
        for r in results:
            meta = r.get('metadata', {})
            simplified.append({
                'rank': r.get('rank'),
                'text': r.get('text', '')[:500],
                'source': meta.get('source', ''),
                'page': meta.get('page', ''),
                'section': meta.get('section', ''),
                'score': round(r.get('score', 0), 4),
                'rrf_score': round(r.get('rrf_score', 0), 4) if r.get('rrf_score') else None,
                'rerank_score': r.get('rerank_score'),
            })
        return json.dumps(simplified, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'error': f'{type(e).__name__}: {str(e)[:300]}'}, ensure_ascii=False)


# Tool 2: graph_query — 知识图谱查询（阶段二占位）

@mcp.tool(
    name="graph_query",
    annotations={
        "title": "知识图谱查询",
        "readOnlyHint": True,
        "destructiveHint": False,
    },
)
def graph_query(
    query: str,
    max_hops: int = 2,
) -> str:
    """查询 Neo4j 知识图谱，支持实体关系多跳遍历。

    Args:
        query: Cypher 查询语句或自然语言实体名
        max_hops: 最大跳数（1-3），默认 2

    Returns:
        JSON 字符串，包含实体节点和关系边列表
    """
    # 阶段二实现：连接 Neo4j，执行 Cypher 查询
    return json.dumps({
        'status': 'not_implemented',
        'message': '知识图谱查询功能将在阶段二实现（Neo4j + 08-10 模块）',
        'query': query,
        'max_hops': max_hops,
    }, ensure_ascii=False)


# Tool 3: vlm_analysis — 图片理解（MiniMax-M3 多模态）

@mcp.tool(
    name="vlm_analysis",
    annotations={
        "title": "图片/图表分析",
        "readOnlyHint": True,
        "destructiveHint": False,
    },
)
def vlm_analysis(
    image_path: str,
    question: str = "请详细描述这张图片的内容",
) -> str:
    """使用 MiniMax-M3 多模态能力分析图片/图表/截图。

    Args:
        image_path: 图片文件路径（支持 PNG/JPG/WebP）
        question: 对图片的提问，默认描述图片内容

    Returns:
        MiniMax-M3 的图片分析结果文本
    """
    try:
        img_path = Path(image_path)
        if not img_path.exists():
            return json.dumps({'error': f'图片不存在: {image_path}'}, ensure_ascii=False)

        # 读取图片并 base64 编码
        suffix = img_path.suffix.lower().lstrip('.')
        mime_map = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                     'webp': 'image/webp', 'gif': 'image/gif', 'bmp': 'image/bmp'}
        mime_type = mime_map.get(suffix, 'image/png')

        with open(img_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        data_url = f"data:{mime_type};base64,{image_b64}"

        # 调用 MiniMax-M3 多模态 API（OpenAI 兼容格式）
        client = m05.create_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                    {'type': 'text', 'text': question},
                ],
            }],
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return json.dumps({'error': f'{type(e).__name__}: {str(e)[:300]}'}, ensure_ascii=False)


# Tool 4: web_search — 联网搜索（阶段三占位）

@mcp.tool(
    name="web_search",
    annotations={
        "title": "联网搜索",
        "readOnlyHint": True,
        "destructiveHint": False,
    },
)
def web_search(
    query: str,
    num_results: int = 5,
) -> str:
    """联网搜索最新信息，补充知识库未覆盖的内容。

    Args:
        query: 搜索查询文本
        num_results: 返回结果数量，默认 5

    Returns:
        JSON 字符串，包含搜索结果列表（标题、URL、摘要）
    """
    # 阶段三实现：接入搜索 API（Bing/Google/自建）
    return json.dumps({
        'status': 'not_implemented',
        'message': '联网搜索功能将在阶段三实现',
        'query': query,
        'num_results': num_results,
    }, ensure_ascii=False)


# 启动入口

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description='MCP 工具服务 — 启动 Server 或 CLI 直接查询',
        epilog='不加 --query 启动 MCP Server；加 --query 直接检索并输出结果',
    )
    ap.add_argument('--query', '-q', type=str, help='检索查询文本（CLI 模式）')
    ap.add_argument('--top-k', '-k', type=int, default=RETRIEVE_TOP_K, help=f'返回条数（默认 {RETRIEVE_TOP_K}）')
    ap.add_argument('--no-bm25', action='store_true', help='关闭 BM25')
    ap.add_argument('--no-hyde', action='store_true', help='关闭 HyDE')
    ap.add_argument('--no-rerank', action='store_true', help='关闭 LLM 重排序')
    ap.add_argument('--save', '-s', type=str, metavar='PATH', help='结果保存到 JSON 文件')
    ap.add_argument('--vlm', type=str, metavar='IMAGE', help='分析图片（CLI 模式）')
    ap.add_argument('--vlm-question', type=str, default='请详细描述这张图片的内容', help='图片分析提问')
    args = ap.parse_args()

    # ── CLI 模式：直接查询 ──
    if args.query or args.vlm:
        if args.vlm:
            result = vlm_analysis(args.vlm, args.vlm_question)
        else:
            result = vector_search(
                args.query,
                top_k=args.top_k,
                use_bm25=not args.no_bm25,
                use_hyde=not args.no_hyde,
                use_rerank=not args.no_rerank,
            )

        # 格式化输出
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                print(f"检索结果: {len(parsed)} 条")
                for item in parsed:
                    src = item.get('source', '?')
                    pg = item.get('page', '?')
                    sec = item.get('section', '?')
                    rk = item.get('rank', '?')
                    rs = item.get('rerank_score', '-')
                    print(f"  [{rk}] {src} p{pg} {sec[:30]} rerank={rs}")
                    print(f"      {item.get('text', '')[:100]}...")
                    print()
            else:
                print(result)
        except json.JSONDecodeError:
            print(result)

        # 保存到文件
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"结果已保存到: {save_path.resolve()}")
        sys.exit(0)

    # ── Server 模式 ──
    print(f"[MCP Server] 启动 MultiModal-RAG-Server (stdio 传输)")
    print(f"  已注册工具: vector_search, graph_query, vlm_analysis, web_search")
    print(f"  向量模型: bge-m3 (本地)")
    print(f"  LLM: {LLM_MODEL}")
    print(f"  CLI 用法: python3 07_mcp_server.py --query '你的问题' [--save result.json]")
    mcp.run(transport="stdio")
