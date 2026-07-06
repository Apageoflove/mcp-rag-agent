"""
04 embedder: 把文字chunk转成向量存进 chromadb，给06检索用
模型用本地 bge-m3 (1024维)，查询时加前缀提升召回
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    EMBEDDING_MODEL_PATH, EMBEDDING_DIMENSION,
    CHROMA_DIR, CHROMA_COLLECTION, RETRIEVE_TOP_K, BGE_QUERY_PREFIX,
)

# bge 官方推荐的查询前缀：编码 query 时拼到前面，文档不加
# 索引时不加是为了和文档向量处于同一分布，查询时加是为了拉近 query-doc 距离
_QPREFIX = "为这个句子生成表示以用于检索相关文章："

_model = None  # 模型只加载一次后面复用；第一次 query 会卡几秒，就是卡在这一步
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return _model


def _embed(texts, is_query=False):
    """批量编码，is_query=True 时加前缀"""
    if is_query:
        texts = [_QPREFIX + t for t in texts]
    return _get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def _get_coll():
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _cid(c):
    """chunk 的唯一id：source::page::chunk_index。天然防重复入库"""
    return f"{c.get('source','unknown')}::p{c.get('page',0)}::c{c.get('chunk_index',0)}"


def add_chunks(chunks, batch_size=64):
    """批量入库，已存在的id跳过。返回新增条数"""
    if not chunks:
        return 0
    coll = _get_coll()
    buf_ids, buf_txt, buf_meta = [], [], []
    n = 0

    def flush():
        nonlocal n
        if not buf_ids:
            return
        vecs = _embed(buf_txt, is_query=False)
        coll.add(ids=buf_ids, embeddings=vecs, documents=buf_txt, metadatas=buf_meta)
        n += len(buf_ids)
        buf_ids.clear(); buf_txt.clear(); buf_meta.clear()

    for c in chunks:
        cid = _cid(c)
        if coll.get(ids=[cid]).get("ids"):  # 已入库，跳过
            continue
        txt = c.get("text", "")
        if not txt.strip():
            continue
        buf_ids.append(cid)
        buf_txt.append(txt)
        buf_meta.append({
            "source": c.get("source", "unknown"),
            "page": int(c.get("page", 0)),
            "section": c.get("section", ""),
            "chunk_type": c.get("content_type", c.get("type", "text")),
            "chunk_index": int(c.get("chunk_index", 0)),
        })
        if len(buf_ids) >= batch_size:
            flush()
    flush()
    return n


def query(query_text, top_k=RETRIEVE_TOP_K, where=None):
    """向量检索 Top-K。返回 [{rank, id, text, metadata, score}, ...]"""
    coll = _get_coll()
    qvec = _embed([query_text], is_query=True)[0]
    res = coll.query(query_embeddings=[qvec], n_results=top_k, where=where)
    if not res or not res.get("ids"):
        return []
    # chroma 返回的距离是 cosine distance（越小越像），转成相似度（越大越像）
    return [{
        "rank": i + 1,
        "id": cid,
        "text": doc,
        "metadata": meta,
        "score": 1.0 - dist,
    } for i, (cid, doc, meta, dist) in enumerate(zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ))]


def count():
    return _get_coll().count()


if __name__ == "__main__":
    print("=== 04 Embedder 自检 ===")
    print(f"模型路径: {EMBEDDING_MODEL_PATH}")
    print(f"ChromaDB: {CHROMA_DIR}  集合: {CHROMA_COLLECTION}")

    v = _embed(["This is a test sentence.", "机器学习是人工智能的一个分支。"])
    print(f"嵌入维度: {len(v[0])} (期望 {EMBEDDING_DIMENSION})")
    print(f"前3维: {v[0][:3]}")
    print(f"查询向量维度: {len(_embed(['What is machine learning?'], is_query=True)[0])}")
    print(f"\n当前集合文档数: {count()}")
    print("用法:")
    print("  from embedder import add_chunks, query")
    print("  add_chunks(chunks_list)")
    print("  results = query('your question', top_k=5)")
