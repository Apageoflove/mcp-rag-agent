"""知识图谱构建模块。读取08的三元组，实体消歧+关系归一化+去重，存入Neo4j。

Neo4j 不可用时自动降级到内置内存图后端（_memory_graph.InMemoryGraph），
使全链路（建图/查询）在无 docker 环境也能跑通。两种后端对外接口一致。
"""
import json
import sys
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
)
from openai import OpenAI
from _memory_graph import InMemoryGraph, build_graph_from_triples as _build_mem_graph


# 关系同义词表：把近义关系归一到标准关系，避免同一语义散成多条边导致多跳查询漏路径。
RELATION_SYNONYMS = {
    'uses':         ['uses', 'use', 'utilize', 'utilizes', 'employ',
                     'employs', 'adopt', 'adopts', 'powered by', 'poweredby',
                     'based on', 'built on', 'accomplish', 'accomplishes'],
    'proposes':     ['proposes', 'propose', 'present', 'presents',
                     'introduce', 'introduces', 'develop', 'develops'],
    'supports':     ['supports', 'support', 'enable', 'enables',
                     'provide', 'provides', 'offer', 'offers'],
    'outperforms':  ['outperforms', 'outperform', 'surpass', 'surpasses',
                     'beat', 'beats', 'exceed', 'exceeds', 'superior to',
                     'comparable to', 'better than'],
    'compared_with':['compared with', 'compare with', 'compared to',
                     'compare to', 'versus', 'vs'],
    'trained_on':   ['trained on', 'train on', 'pretrained on',
                      'fine-tuned on', 'finetuned on'],
    'evaluated_on': ['evaluated on', 'evaluate on', 'tested on',
                      'test on', 'benchmark on', 'benchmarked on'],
    'achieves':     ['achieves', 'achieve', 'reach', 'reaches',
                      'obtain', 'obtains', 'attain', 'attains'],
    'consists_of':  ['consists of', 'consist of', 'compose of',
                      'comprises', 'comprise', 'contain', 'contains'],
    'part_of':      ['part of', 'component of', 'module of',
                      'subcategory of'],
    'is_a':         ['is a', 'is an', 'is', 'are', 'instance of'],
    'released_with':['released with', 'release with', 'comes with',
                      'launched with'],
}


def normalize_relation(relation):
    """把关系归一到标准类型。未匹配的保留原样。"""
    rel_lower = relation.strip().lower()
    for std_type, synonyms in RELATION_SYNONYMS.items():
        if rel_lower in synonyms:
            return std_type
    return relation.strip()


def load_triples(triples_path):
    """从08的JSON文件加载三元组列表。"""
    p = Path(triples_path)
    if not p.exists():
        print(f"[错误] 三元组文件不存在: {p}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        triples = json.load(f)
    print(f"  加载 {len(triples)} 个三元组 from {p.name}")
    return triples


def collect_entities(triples):
    """收集所有实体并统计频次（频次用于选标准名）。"""
    counter = Counter()
    for t in triples:
        counter[t["subject"].strip()] += 1
        counter[t["object"].strip()] += 1
    return counter


def is_substring(a, b):
    """判断 a 是否是 b 的子串（或反之），大小写不敏感。"""
    a_low, b_low = a.lower(), b.lower()
    return (a_low in b_low) or (b_low in a_low)


def string_similarity(a, b):
    """字符串相似度（SequenceMatcher，0~1）。"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def has_ambiguity(short, all_entities):
    """歧义守卫：短串若是多个长串的子串，说明它歧义，不合并。

    例：BGE 是 BGE-M3 和 BGE-Large 的子串 → BGE 歧义，不合并。
    """
    short_low = short.lower()
    longer_matches = []
    for e in all_entities:
        if e == short:
            continue
        if short_low in e.lower() and len(e) > len(short):
            longer_matches.append(e)
    return len(longer_matches) > 1


def _build_llm_client():
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def llm_judge_same_entity(a, b, client):
    """调LLM判断 a 和 b 是否指同一实体。返回 True/False/None(失败)。"""
    prompt = f"""判断下面两个实体名是否指同一个实体。只回答"是"或"否"。

实体A：{a}
实体B：{b}

判断标准：
- 大小写、空格、连字符差异不算不同（"MiniMax-M1" vs "MiniMax M1" → 是）
- 缩写与全称算同一实体（"M1" vs "MiniMax-M1" → 是；"GPT" vs "Generative Pre-trained Transformer" → 是）
- 不同型号/版本不算同一实体（"MiniMax-M1" vs "MiniMax-M2" → 否；"BGE-M3" vs "BGE-Large" → 否）

只输出"是"或"否"，不要其他内容。"""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        answer = resp.choices[0].message.content.strip()
        if "是" in answer and "否" not in answer:
            return True
        if "否" in answer:
            return False
        return None
    except Exception as e:
        print(f"  [警告] LLM消歧判断失败({a} vs {b}): {e}")
        return None


def pick_canonical_name(group, freq_counter):
    """从一组同义实体名里选标准名：频次优先，平局取最长。"""
    return max(group, key=lambda e: (freq_counter.get(e, 0), len(e)))


# def pick_canonical_name(group, freq_counter):
#     def get_key(e):
#         return (freq_counter.get(e, 0), len(e))
#     return max(group, key=get_key)



def disambiguate_entities(freq_counter, client=None):
    """实体消歧：三层过滤 + 歧义守卫。

    层1：子串关系（免API）+ 歧义守卫
    层2：相似度≥0.88 直接合并（免API）
    层3：相似度0.75-0.88 进 LLM 判断（最多20对）

    Returns:
        dict: {原实体名: 标准实体名}
    """
    if client is None:
        client = _build_llm_client()

    entities = list(freq_counter.keys())
    mapping = {e: e for e in entities} # 
    sorted_ents = sorted(entities,
                         key=lambda e: (freq_counter[e], len(e)),
                         reverse=True)

    llm_call_count = 0
    MAX_LLM_CALLS = 20  # 最多调 20 次 LLM 消歧，再多成本扛不住，剩下相似对靠 0.88 阈值兜着

    for i, a in enumerate(sorted_ents):
        if mapping.get(a, a) != a:
            continue
        for b in sorted_ents[i + 1:]:
            if mapping.get(b, b) != b:
                continue
            # 层1：子串关系（免API）+ 歧义守卫
            if is_substring(a, b):
                short = a if len(a) <= len(b) else b
                if has_ambiguity(short, entities):
                    continue
                long_one = b if short == a else a
                mapping[short] = long_one
                continue
            # 层2：相似度≥0.88 直接合并（免API）
            sim = string_similarity(a, b)
            if sim >= 0.88:
                canonical = pick_canonical_name([a, b], freq_counter)
                other = b if canonical == a else a
                mapping[other] = canonical
                continue
            # 层3：相似度0.75-0.88 进 LLM 判断
            if sim >= 0.75:
                if llm_call_count >= MAX_LLM_CALLS:
                    break
                llm_call_count += 1
                same = llm_judge_same_entity(a, b, client)
                if same is True:
                    canonical = pick_canonical_name([a, b], freq_counter)
                    other = b if canonical == a else a
                    mapping[other] = canonical

    # 路径压缩（并查集扁平化）
    def find_root(e):
        while mapping[e] != e:
            mapping[e] = mapping[mapping[e]]
            e = mapping[e]
        return e
    for e in entities:
        mapping[e] = find_root(e)

    merged_count = sum(1 for e in entities if mapping[e] != e)
    print(f"  实体消歧完成：{len(entities)} 个实体 → "
          f"{len(set(mapping.values()))} 个标准实体（合并 {merged_count} 个，"
          f"LLM调用 {llm_call_count} 次）")
    return mapping


# 引文型关系：来自参考文献解析（X authored paper Y），对「论文内容」型
# 知识图谱来说是噪声，建图时过滤掉。
_CITATION_RELATIONS = {
    "authored", "authors", "author of", "authored by", "affiliated with",
    "cited", "cites", "referenced by", "references",
}


def _is_citation_relation(rel: str) -> bool:
    return rel.strip().lower() in _CITATION_RELATIONS


def normalize_triples(triples, entity_mapping, drop_citations: bool = True):
    """对三元组做实体消歧 + 关系归一化 + 去重合并 + 引文噪声过滤。

    相同(subj,rel,obj)合并为一条，count累加。
    drop_citations=True 时丢弃引文型关系（authored/authors/...），它们来自
    参考文献解析，不算论文内容。
    """
    merged = {}
    dropped = 0
    for t in triples:
        subj = entity_mapping.get(t["subject"], t["subject"])
        rel = normalize_relation(t["relation"])
        obj = entity_mapping.get(t["object"], t["object"])
        if drop_citations and _is_citation_relation(rel):
            dropped += 1
            continue
        key = (subj, rel, obj)
        if key in merged:
            merged[key]["count"] += t.get("count", 1)
        else:
            merged[key] = {
                "subject": subj,
                "relation": rel,
                "object": obj,
                "count": t.get("count", 1),
                "source_page": t.get("source_page", 0),
                "source_section": t.get("source_section", ""),
            }
    result = list(merged.values())
    print(f"  三元组归一化去重：{len(triples)} → {len(result)} 条"
          f"（丢弃引文型 {dropped} 条）")
    return result


def get_neo4j_driver():
    """创建 Neo4j 连接 driver。连不上时抛异常，由调用方走内存图降级。"""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[信息] 未安装 neo4j 包，将使用内置内存图后端")
        raise ConnectionError("neo4j package not installed")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print(f"  Neo4j 连接成功: {NEO4J_URI}")
    except Exception as e:
        print(f"[信息] Neo4j 不可用（{type(e).__name__}），将使用内置内存图后端")
        raise
    return driver


def neo4j_available() -> bool:
    """探测 Neo4j 是否可用（连接成功返回 True）。"""
    try:
        d = get_neo4j_driver()
        d.close()
        return True
    except Exception:
        return False


# 内存图缓存：{triples_path: InMemoryGraph}
_MEM_GRAPH_CACHE: dict = {}


def get_memory_graph(triples_path: str = None) -> InMemoryGraph:
    """获取内置内存图。triples_path 缺省时扫描 output/kg_triples 下所有 JSON。"""
    if triples_path and triples_path in _MEM_GRAPH_CACHE:
        return _MEM_GRAPH_CACHE[triples_path]
    all_triples = []
    if triples_path:
        all_triples = load_triples(triples_path)
    else:
        kg_dir = Path(__file__).resolve().parent.parent.parent / "output" / "kg_triples"
        for jp in sorted(kg_dir.glob("*.json")):
            all_triples.extend(load_triples(str(jp)))
    g = _build_mem_graph(all_triples)
    if triples_path:
        _MEM_GRAPH_CACHE[triples_path] = g
    return g


def save_triples_to_memory(triples, graph: InMemoryGraph = None) -> InMemoryGraph:
    """把三元组写入内存图（用于 build_graph 流程的内存后端）。"""
    if graph is None:
        graph = InMemoryGraph()
    graph.add_triples(triples)
    return graph


def create_constraints_and_indexes(driver):
    """创建唯一约束和索引。"""
    cypher = (
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
    )
    with driver.session() as session:
        session.run(cypher)
    print("  唯一约束已就绪: Entity.name")


def save_triples_to_neo4j(triples, driver):
    """把归一化后的三元组写入 Neo4j。

    用统一边类型 RELATES_TO + relation 属性，避免关系类型清洗丢信息、
    同一对实体多种关系时丢边。
    """
    cypher = """
    MERGE (s:Entity {name: $subj})
    MERGE (o:Entity {name: $obj})
    MERGE (s)-[r:RELATES_TO {relation: $rel}]->(o)
    ON CREATE SET r.source_page = $page,
                  r.source_section = $section,
                  r.count = $count
    ON MATCH SET r.count = r.count + $count
    """
    with driver.session() as session:
        for t in triples:
            session.run(cypher, {
                "subj": t["subject"],
                "obj": t["object"],
                "rel": t["relation"],
                "page": t.get("source_page", 0),
                "section": t.get("source_section", ""),
                "count": t.get("count", 1),
            })
    print(f"  写入 Neo4j 完成：{len(triples)} 条三元组")


def get_graph_stats(driver):
    """查询图谱统计信息。"""
    with driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        edge_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        sample = session.run(
            "MATCH (n:Entity) RETURN n.name AS name LIMIT 10")
        sample_names = [r["name"] for r in sample]
        rel_dist = session.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN r.relation AS rel, "
            "count(r) AS cnt ORDER BY cnt DESC LIMIT 10")
        rel_distribution = {r["rel"]: r["cnt"] for r in rel_dist}
    print(f"  图谱统计：{node_count} 个节点, {edge_count} 条边")
    print(f"  节点抽样：{sample_names}")
    print(f"  关系分布(top10)：{rel_distribution}")
    return {"nodes": node_count, "edges": edge_count,
            "sample": sample_names, "rel_dist": rel_distribution}


def build_graph_from_triples(triples_path, driver=None, client=None):
    """完整建图流程：加载→消歧→归一化→写后端→统计验证。

    后端选择：传入 driver 用 Neo4j；否则探测 Neo4j，不可用则降级到内存图。
    """
    print("=" * 60)
    print("知识图谱构建开始")
    print("=" * 60)

    print("\n步骤1: 加载三元组...")
    triples = load_triples(triples_path)
    if not triples:
        print("[终止] 无三元组可建图")
        return None

    print("\n步骤2: 实体消歧...")
    freq = collect_entities(triples)
    entity_mapping = disambiguate_entities(freq, client=client)

    print("\n步骤3: 三元组归一化去重...")
    triples = normalize_triples(triples, entity_mapping)

    use_neo4j = driver is not None
    if driver is None:
        try:
            driver = get_neo4j_driver()
            use_neo4j = True
        except Exception:
            use_neo4j = False

    if use_neo4j:
        print("\n步骤4: 写入 Neo4j...")
        try:
            create_constraints_and_indexes(driver)
            save_triples_to_neo4j(triples, driver)
            print("\n步骤5: 统计验证...")
            stats = get_graph_stats(driver)
        finally:
            pass
    else:
        print("\n步骤4: 写入内存图后端...")
        g = save_triples_to_memory(triples)
        _MEM_GRAPH_CACHE[triples_path] = g
        print(f"  写入内存图完成：{len(triples)} 条三元组")
        print("\n步骤5: 统计验证...")
        stats = {
            "nodes": g.node_count(),
            "edges": g.edge_count(),
            "sample": g.sample_nodes(10),
            "rel_dist": dict(list(g.relation_distribution().items())[:10]),
            "backend": "memory",
        }
        print(f"  图谱统计：{stats['nodes']} 个节点, {stats['edges']} 条边 (内存图)")
        print(f"  节点抽样：{stats['sample']}")
        print(f"  关系分布(top10)：{stats['rel_dist']}")

    print("\n" + "=" * 60)
    print("知识图谱构建完成！")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 09_kg_builder.py <triples_json_path>")
        print("例:   python3 09_kg_builder.py "
              "output/kg_triples/MiniMax_M1_tech_report.pdf.json")
        sys.exit(1)
    build_graph_from_triples(sys.argv[1])
