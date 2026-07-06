"""知识图谱查询模块。自然语言→Cypher→执行查询，路径断了时向量检索兜底。"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, REFLECTION_THRESHOLD
from importlib.machinery import SourceFileLoader
from _memory_graph import (
    InMemoryGraph,
    execute_standardized_cypher as _mem_exec_triple,
    execute_path_cypher as _mem_exec_path,
)

m05 = SourceFileLoader('m05', str(Path(__file__).resolve().parent / '05_llm_client.py')).load_module()
m09 = SourceFileLoader('m09', str(Path(__file__).resolve().parent / '09_kg_builder.py')).load_module()


# 允许的关键字（只读）
_ALLOWED_KEYWORDS = {
    'MATCH', 'OPTIONAL', 'WHERE', 'RETURN', 'WITH', 'ORDER', 'BY',
    'LIMIT', 'SKIP', 'AS', 'AND', 'OR', 'NOT', 'IN', 'CONTAINS',
    'STARTS', 'ENDS', 'DISTINCT', 'UNION', 'ALL', 'EXISTS', 'IS',
    'NULL', 'TRUE', 'FALSE', 'COUNT', 'COLLECT', 'NODES', 'REL',
    'RELATIONSHIPS', 'TYPE', 'LABELS', 'PROPERTIES', 'ID', 'HEAD',
    'TAIL', 'LAST', 'SIZE', 'TOLOWER', 'TOUPPER', 'TRIM', 'SPLIT',
    'JOIN', 'SUBSTRING', 'COALESCE', 'CASE', 'WHEN', 'THEN', 'ELSE',
    'END', 'DESC', 'ASC',
}

# 禁止的关键字（写操作/危险操作）
_FORBIDDEN_KEYWORDS = {
    'CREATE', 'MERGE', 'DELETE', 'DETACH', 'SET', 'REMOVE', 'DROP',
    'CALL', 'YIELD', 'LOAD', 'CSV', 'PERIODIC', 'ITERATE', 'SHORTEST',
    'ALLSHORTEST', 'EXPLAIN', 'PROFILE', 'USE',
}


def validate_cypher(cypher: str) -> tuple[bool, str]:
    """校验 Cypher 语句是否安全（只读）。

    Returns:
        (is_safe, reason): is_safe=True 表示可执行
    """
    if not cypher or not cypher.strip():
        return False, "空查询"

    tokens = re.findall(r'\b[A-Z]+\b', cypher.upper())

    for tok in tokens:
        if tok in _FORBIDDEN_KEYWORDS:
            return False, f"包含禁止关键字: {tok}"

    stripped = cypher.strip().upper()
    if not (stripped.startswith('MATCH') or stripped.startswith('OPTIONAL')):
        return False, "查询必须以 MATCH 或 OPTIONAL MATCH 开头"

    if 'RETURN' not in stripped:
        return False, "查询必须包含 RETURN 子句"

    if ';' in cypher:
        return False, "查询包含分号（禁止多语句）"

    return True, "OK"


_TEXT2CYPHER_SYSTEM = """You are a Cypher query generator for a Neo4j knowledge graph.

Graph Schema:
- Nodes: (:Entity {name: string})
- Relationships: (:Entity)-[:RELATES_TO {relation: string, source_page: int, source_section: string, count: int}]->(:Entity)
- The relationship TYPE is always RELATES_TO. The actual relation name is stored in the `relation` property.
- To query a specific relation, use: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE r.relation = 'uses'

Available relation types (use these exact strings in WHERE r.relation = '...'):
{relations}

Rules:
1. Generate ONLY a Cypher query. No explanation, no markdown, no backticks.
2. Always use MATCH (read-only). Never use CREATE/MERGE/DELETE/SET.
3. For multi-hop queries, chain relationships: (a)-[:RELATES_TO*1..3]->(c)
4. To get the full path, return: nodes(p) AS entities, relationships(p) AS rels, [r.relation FOR r IN relationships(p)] AS relation_names
5. Limit results to 10 to avoid huge responses: add LIMIT 10 at the end.
6. Use case-insensitive matching for entity names: WHERE toLower(a.name) CONTAINS toLower('keyword')
7. For "what does X use" → MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE toLower(a.name) CONTAINS 'x' AND r.relation IN ['uses','utilizes','powered by'] RETURN b.name
8. For "X vs Y" or comparison → MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE r.relation IN ['compared_with','outperforms'] RETURN a.name, r.relation, b.name
9. CRITICAL - For "who proposes/developed/created X" (reverse query), match against b.name (the OBJECT), not a.name: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE toLower(b.name) CONTAINS 'x' AND r.relation = 'proposes' RETURN a.name AS proposer LIMIT 10. The question asks WHO did the action, so the unknown is the SUBJECT (a), and X is the OBJECT (b).
10. Always return ONLY the b.name (or a.name for reverse) as the answer entity. Do NOT return the query subject itself as a result. For "what does X use", X is the subject (a), return only b.name.

Examples:
Q: What technology does MiniMax-M1 use?
A: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE toLower(a.name) CONTAINS 'minimax-m1' AND r.relation = 'uses' RETURN b.name AS technology, r.count AS frequency, r.source_page AS page LIMIT 10

Q: What is the relationship between MiniMax-M1 and DeepSeek-R1?
A: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE (toLower(a.name) CONTAINS 'minimax-m1' AND toLower(b.name) CONTAINS 'deepseek') OR (toLower(a.name) CONTAINS 'deepseek' AND toLower(b.name) CONTAINS 'minimax-m1') RETURN a.name AS entity1, r.relation AS relation, b.name AS entity2 LIMIT 10

Q: What is CISPO?
# A: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE toLower(a.name) CONTAINS 'cispo' RETURN r.relation AS relation, b.name AS description LIMIT 10
A: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE toLower(a.name) CONTAINS 'cispo' RETURN b.name AS description LIMIT 10

Q: List all entities related to MiniMax-M1 within 2 hops.
A: MATCH p=(a:Entity)-[:RELATES_TO*1..2]->(b:Entity) WHERE toLower(a.name) CONTAINS 'minimax-m1' RETURN nodes(p) AS entities, [r.relation FOR r IN relationships(p)] AS relations LIMIT 10

Q: Who proposes MiniMax-M1?
A: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE toLower(b.name) CONTAINS 'minimax-m1' AND r.relation = 'proposes' RETURN a.name AS proposer LIMIT 10

Q: What does MiniMax-M1 outperform?
A: MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE toLower(a.name) CONTAINS 'minimax-m1' AND r.relation = 'outperforms' RETURN b.name AS outperformed_target LIMIT 10"""


def get_available_relations(driver) -> list[str]:
    """从图谱动态获取所有已有的关系类型（给 LLM 提示用）。"""
    try:
        if isinstance(driver, InMemoryGraph):
            return sorted(driver.relation_distribution().keys())
        with driver.session() as session:
            result = session.run(
                "MATCH ()-[r:RELATES_TO]->() "
                "RETURN DISTINCT r.relation AS rel ORDER BY rel"
            )
            return [r["rel"] for r in result]
    except Exception:
        return []


def nl_to_cypher(question: str, available_relations: list[str],
                 client=None) -> str:
    """将自然语言问题转为 Cypher 查询语句。

    Args:
        question: 用户问题
        available_relations: 图谱中已有的关系类型列表
        client: LLM 客户端（可选）

    Returns:
        Cypher 查询字符串
    """
    if client is None:
        client = m05.create_client()

    relations_str = ", ".join(f"'{r}'" for r in available_relations) if available_relations else "(图谱为空)"
    system = _TEXT2CYPHER_SYSTEM.replace("{relations}", relations_str)

    try:
        resp = client.chat.completions.create(
            model=m05.LLM_MODEL if hasattr(m05, 'LLM_MODEL') else 'MiniMax-M3',
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        cypher = resp.choices[0].message.content.strip()
        cypher = re.sub(r'<think>.*?</think>\s*', '', cypher, flags=re.DOTALL).strip()
        cypher = re.sub(r'^```(?:cypher)?\s*', '', cypher)
        cypher = re.sub(r'\s*```$', '', cypher)
        return cypher.strip()
    except Exception as e:
        print(f"  [警告] text2cypher 失败: {e}", file=sys.stderr)
        return ""


def execute_cypher(driver, cypher: str) -> list[dict]:
    """执行 Cypher 查询，返回结果列表。

    driver 可以是 Neo4j driver 或 InMemoryGraph（按类型分发）。
    """
    # 内存图后端：driver 是 InMemoryGraph 实例
    if isinstance(driver, InMemoryGraph):
        return _mem_exec_triple(cypher, driver)
    # Neo4j 后端
    with driver.session() as session:
        result = session.run(cypher)
        keys = result.keys()
        records = []
        for record in result:
            row = {}
            for key in keys:
                val = record[key]
                if hasattr(val, 'labels'):
                    row[key] = dict(val)
                    row[key]['_labels'] = list(val.labels)
                elif hasattr(val, 'type'):
                    row[key] = dict(val)
                    row[key]['_type'] = val.type
                elif isinstance(val, list):
                    converted = []
                    for item in val:
                        if hasattr(item, 'labels'):
                            converted.append({'name': dict(item).get('name', str(item)),
                                            'labels': list(item.labels)})
                        elif hasattr(item, 'type'):
                            converted.append({'relation': dict(item).get('relation', str(item)),
                                            'type': item.type})
                        else:
                            converted.append(item)
                    row[key] = converted
                else:
                    row[key] = val
            records.append(row)
        return records


def execute_cypher_path(driver, cypher: str) -> list[dict]:
    """执行多跳 path 模式 Cypher（内存图后端专用，Neo4j 直接走 execute_cypher）。"""
    if isinstance(driver, InMemoryGraph):
        return _mem_exec_path(cypher, driver)
    return execute_cypher(driver, cypher)


def _resolve_driver(driver=None, triples_path: str = None):
    """获取图后端：传入 driver 直接用；否则探测 Neo4j，不可用则降级内存图。

    返回 (driver, is_memory)。
    """
    if driver is not None:
        return driver, isinstance(driver, InMemoryGraph)
    try:
        d = m09.get_neo4j_driver()
        return d, False
    except Exception:
        g = m09.get_memory_graph(triples_path)
        return g, True


def format_graph_results(records: list[dict]) -> str:
    """将 Cypher 结果格式化为 LLM 可读的上下文文本。"""
    if not records:
        return "(图谱无匹配结果)"

    lines = []
    for i, rec in enumerate(records, 1):
        parts = []
        for key, val in rec.items():
            if isinstance(val, dict) and 'name' in val:
                parts.append(f"{key}: {val['name']}")
            elif isinstance(val, list):
                path_str = []
                for item in val:
                    if isinstance(item, dict):
                        if 'name' in item:
                            path_str.append(item['name'])
                        elif 'relation' in item:
                            path_str.append(f"-[{item['relation']}]->")
                parts.append(f"{key}: {' '.join(path_str)}")
            else:
                parts.append(f"{key}: {val}")
        lines.append(f"[{i}] " + ", ".join(parts))
    return "\n".join(lines)


_NODE_RE = re.compile(r'\(\s*(\w+)\s*:\s*Entity\b[^)]*\)')
_VARLEN_RE = re.compile(r'RELATES_TO\s*\*')
_REL_RE = re.compile(r'(<?)-\[\s*(\w*)\s*:\s*RELATES_TO\s*(?:\{[^}]*\})?\s*\]-(>?)')
_RETURN_SPLIT_RE = re.compile(r'\bRETURN\b', re.IGNORECASE)
_LIMIT_RE = re.compile(r'\bLIMIT\s+(\d+)', re.IGNORECASE)

# 别名.name 与字符串字面量比较（= / CONTAINS / STARTS WITH / ENDS WITH / IN，含 toLower 包裹）
_ALIAS_NAME_LIT_RE = re.compile(
    r"""(?:toLower\s*\(\s*)?
        (\w+)\.name
        \s*\)?\s*
        (?:=|CONTAINS|STARTS\s+WITH|ENDS\s+WITH|IN)
        \s*(?:toLower\s*\(\s*)?
        [\[]?\s*['"]
    """,
    re.IGNORECASE | re.VERBOSE,
)
# 反向：'literal' <cmp> 别名.name
_LIT_NAME_RE = re.compile(
    r"""['"]\s*\)?\s*
        (?:=|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)
        \s*(?:toLower\s*\(\s*)?
        (\w+)\.name
    """,
    re.IGNORECASE | re.VERBOSE,
)


def standardize_projection(cypher: str) -> tuple[str, str, dict]:
    """把 LLM 生成的 RETURN 重写为固定三元组投影，消除“选列”非确定性。

    只信任 LLM 的 MATCH+WHERE 定位子图，RETURN 由服务端固定成
    subject/predicate/object 三列，元数据列从源头不再出现。

    Returns:
        (std_cypher, mode, roles)
        mode:  'triple'（单跳可标准化）| 'path'（多跳/无法解析，原样执行）
        roles: {'subject_alias': str|None, 'object_alias': str|None}
    """
    if _VARLEN_RE.search(cypher):
        return cypher, 'path', {'subject_alias': None, 'object_alias': None}

    nodes = _NODE_RE.findall(cypher)
    rel = _REL_RE.search(cypher)
    parts = _RETURN_SPLIT_RE.split(cypher, maxsplit=1)
    if len(nodes) != 2 or rel is None or len(parts) != 2:
        return cypher, 'path', {'subject_alias': None, 'object_alias': None}

    left_arrow, rel_var, right_arrow = rel.group(1), rel.group(2), rel.group(3)
    n1, n2 = nodes[0], nodes[1]
    if right_arrow == '>':
        subj, obj = n1, n2
    elif left_arrow == '<':
        subj, obj = n2, n1
    else:
        subj, obj = n1, n2

    prefix = parts[0]
    if not rel_var:
        rel_var = 'r_std'
        prefix = _REL_RE.sub(
            lambda m: f'{m.group(1)}-[{rel_var}:RELATES_TO]-{m.group(3)}',
            prefix, count=1,
        )

    limit_m = _LIMIT_RE.search(parts[1])
    limit_clause = f' LIMIT {limit_m.group(1)}' if limit_m else ' LIMIT 50'
    std = (f'{prefix.rstrip()} RETURN '
           f'{subj}.name AS subject, {rel_var}.relation AS predicate, '
           f'{obj}.name AS object{limit_clause}')
    return std, 'triple', {'subject_alias': subj, 'object_alias': obj}


def infer_anchored_endpoints(cypher: str) -> set:
    """从 WHERE 推断被实体字面量锚定的别名集合（=已知量）。

    只认 别名.name 与字符串字面量的比较；r.relation 约束不算实体锚点。
    不看问题文本、只看查询结构，语料无关、语言无关。
    """
    parts = re.split(r'\bWHERE\b', cypher, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return set()
    where = re.split(r'\bRETURN\b', parts[1], maxsplit=1, flags=re.IGNORECASE)[0]
    anchors = set(m.group(1) for m in _ALIAS_NAME_LIT_RE.finditer(where))
    anchors |= set(m.group(1) for m in _LIT_NAME_RE.finditer(where))
    return anchors


def _collect_path_entities(val, out: set) -> None:
    """path 模式下收集实体名（节点 dict 的 name / 标量列），跳过 relation/元数据。"""
    if isinstance(val, dict):
        if 'name' in val:
            out.add(val['name'])
    elif isinstance(val, list):
        for item in val:
            _collect_path_entities(item, out)
    elif isinstance(val, str):
        out.add(val)


def project_answers(records: list, anchors: set, roles: dict, mode: str) -> list:
    """按锚点数投影出干净答案集，排序去重，确定性输出。

    - 锚点 0（探索型）：所有实体，不含谓词
    - 锚点 1（定向型）：未锚定端实体，不含谓词
    - 锚点 ≥2（关系型）：两端实体 + 谓词
    """
    entities: set = set()
    predicates: set = set()

    if mode == 'triple':
        subj_a = roles.get('subject_alias')
        obj_a = roles.get('object_alias')
        subj_anchored = subj_a in anchors
        obj_anchored = obj_a in anchors
        n_anchor = int(subj_anchored) + int(obj_anchored)

        for rec in records:
            s, p, o = rec.get('subject'), rec.get('predicate'), rec.get('object')
            if n_anchor >= 2:
                if s: entities.add(s)
                if o: entities.add(o)
                if p: predicates.add(p)
            elif n_anchor == 1:
                if subj_anchored and o:
                    entities.add(o)
                elif obj_anchored and s:
                    entities.add(s)
                else:
                    if s: entities.add(s)
                    if o: entities.add(o)
            else:
                if s: entities.add(s)
                if o: entities.add(o)
    else:
        for rec in records:
            for _key, val in rec.items():
                _collect_path_entities(val, entities)

    return sorted(entities | predicates)


def graph_search(question: str, driver=None, client=None,
                 fallback_to_vector: bool = True,
                 triples_path: str = None) -> dict:
    """图谱检索主流程。

    Args:
        question: 自然语言问题
        driver: Neo4j driver 或 InMemoryGraph（可选，自动解析后端）
        client: LLM 客户端（可选）
        fallback_to_vector: 路径断了时是否向量兜底
        triples_path: 内存图降级时加载的三元组文件（缺省扫描全部）

    Returns:
        {'question', 'cypher', 'graph_results', 'graph_context',
         'fallback_used', 'vector_results', 'source'}
    """
    own_driver = False
    driver, is_memory = _resolve_driver(driver, triples_path)
    # 内存图无需 close；Neo4j 由本函数负责 close（当它是这里创建的）
    own_driver = not is_memory and driver is not None

    if client is None:
        client = m05.create_client()

    try:
        relations = get_available_relations(driver)

        cypher = nl_to_cypher(question, relations, client)
        if not cypher:
            return {'question': question, 'cypher': '', 'std_cypher': '',
                    'answers': [], 'answer_mode': '', 'graph_results': [],
                    'graph_context': '', 'fallback_used': False,
                    'vector_results': [], 'source': 'empty'}

        is_safe, reason = validate_cypher(cypher)
        if not is_safe:
            print(f"  [安全拦截] Cypher 校验失败: {reason}", file=sys.stderr)
            print(f"  生成的 Cypher: {cypher}", file=sys.stderr)
            cypher = ''
        else:
            std_cypher, mode, roles = standardize_projection(cypher)
            anchors = infer_anchored_endpoints(std_cypher)
            is_safe2, _ = validate_cypher(std_cypher)
            exec_cypher = std_cypher if is_safe2 else cypher
            try:
                if is_memory and mode == 'path':
                    records = execute_cypher_path(driver, cypher)
                else:
                    records = execute_cypher(driver, exec_cypher)
            except Exception as e:
                print(f"  [查询执行失败] {e}", file=sys.stderr)
                records = []

            # 空结果重试：text2cypher 是 LLM 生成，偶发的 relation 别名/结构
            # 偏差会导致空结果。重试一次（带图谱已有 relation 列表强提示）。
            if not records:
                retry_cypher = nl_to_cypher(question, relations, client)
                if retry_cypher and retry_cypher.strip() != cypher.strip():
                    is_safe3, _ = validate_cypher(retry_cypher)
                    if is_safe3:
                        std2, mode2, roles2 = standardize_projection(retry_cypher)
                        anchors2 = infer_anchored_endpoints(std2)
                        is_safe4, _ = validate_cypher(std2)
                        exec2 = std2 if is_safe4 else retry_cypher
                        try:
                            if is_memory and mode2 == 'path':
                                records = execute_cypher_path(driver, retry_cypher)
                            else:
                                records = execute_cypher(driver, exec2)
                        except Exception:
                            records = []
                        if records:
                            cypher, std_cypher = retry_cypher, exec2
                            mode, roles, anchors = mode2, roles2, anchors2

            if records:
                ctx = format_graph_results(records)
                answers = project_answers(records, anchors, roles, mode)
                return {
                    'question': question,
                    'cypher': cypher,
                    'std_cypher': exec_cypher,
                    'answers': answers,
                    'answer_mode': mode,
                    'graph_results': records,
                    'graph_context': ctx,
                    'fallback_used': False,
                    'vector_results': [],
                    'source': 'graph',
                }

        if fallback_to_vector:
            print("  [信息] 图谱无结果，退回向量检索兜底", file=sys.stderr)
            try:
                m06 = SourceFileLoader(
                    'm06', str(Path(__file__).resolve().parent / '06_rag_query.py')
                ).load_module()
                vec_results = m06.retrieve(question, top_k=5)
                ctx = "\n".join(
                    f"[{i+1}] {r['text'][:200]}" for i, r in enumerate(vec_results)
                )
                return {
                    'question': question,
                    'cypher': cypher,
                    'std_cypher': '',
                    'answers': [],
                    'answer_mode': '',
                    'graph_results': [],
                    'graph_context': '',
                    'fallback_used': True,
                    'vector_results': vec_results,
                    'source': 'vector',
                }
            except Exception as e:
                print(f"  [向量兜底失败] {e}", file=sys.stderr)

        return {
            'question': question,
            'cypher': cypher,
            'std_cypher': '',
            'answers': [],
            'answer_mode': '',
            'graph_results': [],
            'graph_context': '',
            'fallback_used': False,
            'vector_results': [],
            'source': 'empty',
        }
    finally:
        if own_driver:
            driver.close()


def graph_query_tool(question: str) -> dict:
    """MCP 工具接口：知识图谱查询。

    Args:
        question: 用户的自然语言问题

    Returns:
        dict: 见 graph_search 返回格式
    """
    print(f"[graph_query_tool] 查询: {question}", file=sys.stderr)
    result = graph_search(question)
    print(f"[graph_query_tool] 来源: {result['source']}, "
          f"图谱结果: {len(result['graph_results'])} 条, "
          f"向量兜底: {len(result['vector_results'])} 条", file=sys.stderr)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description='知识图谱查询')
    ap.add_argument('question', nargs='*', help='自然语言问题')
    ap.add_argument('--no-fallback', action='store_true',
                    help='禁用向量兜底')
    args = ap.parse_args()

    if not args.question:
        print("用法: python3 10_kg_query.py 你的问题")
        print("示例: python3 10_kg_query.py what does MiniMax-M1 use")
        print("      python3 10_kg_query.py what is CISPO")
        print("      python3 10_kg_query.py relationship between MiniMax-M1 and DeepSeek-R1")
        sys.exit(1)

    q = ' '.join(args.question)
    print(f"\n=== Question ===\n{q}\n")

    result = graph_search(q, fallback_to_vector=not args.no_fallback)

    print(f"=== Source: {result['source']} ===")
    if result['cypher']:
        print(f"\n--- Generated Cypher ---\n{result['cypher']}")
    if result['graph_results']:
        print(f"\n--- Graph Results ({len(result['graph_results'])} 条) ---")
        print(result['graph_context'])
    if result['fallback_used'] and result['vector_results']:
        print(f"\n--- Vector Fallback ({len(result['vector_results'])} 条) ---")
        for i, r in enumerate(result['vector_results'], 1):
            meta = r.get('metadata', {})
            print(f"  [{i}] [{meta.get('source','')[:30]}] p{meta.get('page','')}")
            print(f"      {r['text'][:150]}...")
