"""内置内存图后端（Neo4j 不可用时的纯 Python 降级实现）。

设计目标：
- 不依赖 docker / Neo4j，让图谱构建/查询全链路在任何环境都能跑。
- 基于三元组集合 + 索引，按实体名(子串/相等)和关系(相等/IN)匹配，
  支持单跳与多跳路径遍历。
- 与 10_kg_query 的 LLM→Cypher→standardize_projection 流水线对接：
  接收标准化后的三元组投影查询，解析 WHERE 里「别名.name 与字符串字面量比较」
  和「r.relation = / IN ...」两类条件，执行等价语义匹配。
  （只解释查询的语义，不编造答案。）
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


class InMemoryGraph:
    """三元组内存图。

    边结构：每条边是 {subject, predicate, object, source_page, source_section, count}
    索引：
      _by_subj[entity_name] -> [edge,...]
      _by_obj[entity_name]  -> [edge,...]
      _by_rel[relation]     -> [edge,...]
      _entities             -> set(name)
    实体名匹配大小写不敏感（与 Cypher 的 toLower(...) CONTAINS 语义一致）。
    """

    def __init__(self):
        self._edges = []
        self._by_subj = defaultdict(list)
        self._by_obj = defaultdict(list)
        self._by_rel = defaultdict(list)
        self._entities = set()
        self._entity_lower = []  # [(lower_name, original_name]

    # ── 写 ─────────────────────────────────────────────
    def add_triples(self, triples: Iterable[dict]):
        for t in triples:
            edge = {
                "subject": t.get("subject", "").strip(),
                "predicate": t.get("relation", t.get("predicate", "")).strip(),
                "object": t.get("object", "").strip(),
                "source_page": int(t.get("source_page", 0) or 0),
                "source_section": t.get("source_section", ""),
                "count": int(t.get("count", 1) or 1),
            }
            if not edge["subject"] or not edge["object"] or not edge["predicate"]:
                continue
            self._edges.append(edge)
            self._by_subj[edge["subject"]].append(edge)
            self._by_obj[edge["object"]].append(edge)
            self._by_rel[edge["predicate"]].append(edge)
            for name in (edge["subject"], edge["object"]):
                if name not in self._entities:
                    self._entities.add(name)
                    self._entity_lower.append((name.lower(), name))
        return self

    # ── 统计 ────────────────────────────────────────────
    def node_count(self) -> int:
        return len(self._entities)

    def edge_count(self) -> int:
        return len(self._edges)

    def sample_nodes(self, k: int = 10) -> list[str]:
        return list(self._entities)[:k]

    def relation_distribution(self) -> dict:
        out = {}
        for e in self._edges:
            out[e["predicate"]] = out.get(e["predicate"], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ── 实体名匹配（toLower CONTAINS / = / STARTS WITH / ENDS WITH） ──
    def _name_matches(self, alias_name: str, op: str, literal: str) -> bool:
        a = alias_name.lower()
        lit = literal.lower()
        if op == "=":
            return a == lit
        if op == "contains":
            return lit in a
        if op == "starts with":
            return a.startswith(lit)
        if op == "ends with":
            return a.endswith(lit)
        return False

    # ── 单跳匹配 ───────────────────────────────────────
    def match_triples(self, subj_filter=None, rel_filter=None, obj_filter=None) -> list[dict]:
        """单跳三元组查询。每个 filter 为 None 或 (op, literal)。"""
        out = []
        for e in self._edges:
            if subj_filter and not self._name_matches(e["subject"], *subj_filter):
                continue
            if obj_filter and not self._name_matches(e["object"], *obj_filter):
                continue
            if rel_filter and e["predicate"] != rel_filter:
                continue
            out.append({
                "subject": e["subject"],
                "predicate": e["predicate"],
                "object": e["object"],
                "source_page": e["source_page"],
                "source_section": e["source_section"],
                "count": e["count"],
            })
        return out

    # ── 多跳路径 ───────────────────────────────────────
    def match_paths(self, start_filter, max_hops: int = 2, limit: int = 10) -> list[dict]:
        """从满足 start_filter 的实体出发，做 max_hops 跳的路径遍历。

        start_filter: (op, literal) 对起点实体的 name 约束。
        返回 [{entities:[...], relations:[...], edges:[...]}, ...]
        """
        op, lit = start_filter
        starts = [n for (low, n) in self._entity_lower
                  if self._name_matches(n, op, lit)]
        results = []
        seen_paths = set()

        def walk(current, path_entities, path_rels, path_edges, hops):
            if len(results) >= limit:
                return
            if hops > 0:
                key = tuple(path_entities)
                if key not in seen_paths:
                    seen_paths.add(key)
                    results.append({
                        "entities": list(path_entities),
                        "relations": list(path_rels),
                        "edges": list(path_edges),
                    })
            if hops >= max_hops:
                return
            for e in self._by_subj.get(current, []):
                if e["object"] in path_entities:
                    continue
                walk(e["object"], path_entities + [e["object"]],
                     path_rels + [e["predicate"]], path_edges + [e], hops + 1)

        for s in starts:
            walk(s, [s], [], [], 0)
        return results


# ── Cypher 标准化投影解释器 ───────────────────────────────
# 只解释 standardize_projection 产出的简单形式：
#   MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE <conds> RETURN ... LIMIT n
# <conds> 支持：别名.name (CONTAINS|=|STARTS WITH|ENDS WITH) 'lit'
#              r.relation (= 'x' | IN ['x','y'])
#              AND / OR / 括号（按 AND/OR 优先级近似求值）

_NAME_COND_RE = re.compile(
    r"""(?:toLower\s*\(\s*)?
        (?P<alias>\w+)\.name
        \s*\)?\s*
        (?P<op>=|CONTAINS|STARTS\s+WITH|ENDS\s+WITH|IN)
        \s*(?:toLower\s*\(\s*)?
        (?P<lit>[^)\s]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_REL_EQ_RE = re.compile(
    r"""(?P<alias>\w+)\.relation
        \s*(?P<op>=|IN)\s*
        (?:'(?P<lit>[^']*)'
          |\[(?P<list>[^\]]*)\])
    """,
    re.IGNORECASE | re.VERBOSE,
)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
_VARLEN_RE = re.compile(r"RELATES_TO\s*\*")
_NODE_RE = re.compile(r"\(\s*(\w+)\s*:\s*Entity\b[^)]*\)")
_REL_RE = re.compile(r"(<?)-\[\s*(\w*)\s*:\s*RELATES_TO\s*(?:\{[^}]*\})?\s*\]-(>?)")
_RETURN_SPLIT_RE = re.compile(r"\bRETURN\b", re.IGNORECASE)


def _parse_string_literal(token: str) -> str:
    """去掉引号/方括号，返回纯字符串。"""
    t = token.strip().strip("'\"")
    return t


def _parse_relation_list(group: str) -> list:
    """解析 IN [...] 里的字符串列表。"""
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", group)
    return [a or b for a, b in items]


def execute_standardized_cypher(cypher: str, graph: InMemoryGraph) -> list[dict]:
    """执行标准化后的 Cypher（单跳三元组投影）。

    返回 records: [{subject, predicate, object, source_page, source_section, count}, ...]
    多跳/无法解析时返回空，由调用方决定是否走 path 通道。
    """
    if _VARLEN_RE.search(cypher):
        return []  # 多跳由 path 通道处理

    nodes = _NODE_RE.findall(cypher)
    rel = _REL_RE.search(cypher)
    parts = _RETURN_SPLIT_RE.split(cypher, maxsplit=1)
    if len(nodes) != 2 or rel is None or len(parts) != 2:
        return []

    n1, n2 = nodes[0], nodes[1]
    left_arrow, _, right_arrow = rel.group(1), rel.group(2), rel.group(3)
    if right_arrow == ">":
        subj_alias, obj_alias = n1, n2
    elif left_arrow == "<":
        subj_alias, obj_alias = n2, n1
    else:
        subj_alias, obj_alias = n1, n2

    # WHERE
    where = ""
    parts2 = re.split(r"\bWHERE\b", parts[0], maxsplit=1, flags=re.IGNORECASE)
    if len(parts2) == 2:
        where = parts2[1]

    # 抽取别名条件
    subj_filter = None
    obj_filter = None
    for m in _NAME_COND_RE.finditer(where):
        alias = m.group("alias")
        op = m.group("op").upper().replace(" ", "")
        op_norm = {"=": "=", "CONTAINS": "contains",
                   "STARTSWITH": "starts with",
                   "ENDSWITH": "ends with"}.get(op, "contains")
        # IN 用于 name 时少见，跳过
        if op_norm == "IN" or m.group("op").upper() == "IN":
            continue
        lit = _parse_string_literal(m.group("lit"))
        if alias == subj_alias and subj_filter is None:
            subj_filter = (op_norm, lit)
        elif alias == obj_alias and obj_filter is None:
            obj_filter = (op_norm, lit)

    # 关系条件
    rel_filter = None
    for m in _REL_EQ_RE.finditer(where):
        if m.group("op").upper() == "=":
            rel_filter = m.group("lit")
            break
        if m.group("list") is not None:
            # IN：取第一个匹配的关系（若图里有多个候选，逐个试）
            candidates = _parse_relation_list(m.group("list"))
            for c in candidates:
                if graph._by_rel.get(c):
                    rel_filter = c
                    break
            break

    limit_m = _LIMIT_RE.search(parts[1])
    limit = int(limit_m.group(1)) if limit_m else 50

    records = graph.match_triples(subj_filter, rel_filter, obj_filter)
    return records[:limit]


def execute_path_cypher(cypher: str, graph: InMemoryGraph) -> list[dict]:
    """执行多跳 path 模式 Cypher。

    解析起点实体的 name 约束 + 变长跳数，返回 path 记录。
    返回 records: [{entities:[...], relations:[...]}]
    """
    # 跳数：RELATES_TO*1..N
    hop_m = re.search(r"RELATES_TO\s*\*\s*(\d+)\s*\.\.\s*(\d+)", cypher, re.IGNORECASE)
    max_hops = int(hop_m.group(2)) if hop_m else 2
    limit_m = _LIMIT_RE.search(cypher)
    limit = int(limit_m.group(1)) if limit_m else 10

    # 起点 name 约束
    start_filter = None
    for m in _NAME_COND_RE.finditer(cypher):
        op = m.group("op").upper().replace(" ", "")
        op_norm = {"=": "=", "CONTAINS": "contains",
                   "STARTSWITH": "starts with",
                   "ENDSWITH": "ends with"}.get(op, "contains")
        if op_norm == "IN":
            continue
        lit = _parse_string_literal(m.group("lit"))
        start_filter = (op_norm, lit)
        break

    if start_filter is None:
        return []
    paths = graph.match_paths(start_filter, max_hops=max_hops, limit=limit)
    return [{"entities": p["entities"], "relations": p["relations"]} for p in paths]


# ── 全局单例：按图谱 key 缓存加载 ────────────────────────
_GRAPH_CACHE = {}


def build_graph_from_triples(triples: list[dict]) -> InMemoryGraph:
    """从三元组列表构建内存图。"""
    g = InMemoryGraph()
    g.add_triples(triples)
    return g


def load_memory_graph_from_file(triples_path: str) -> InMemoryGraph:
    """从 JSON 文件加载三元组并建图（带缓存）。"""
    import json
    from pathlib import Path
    p = str(Path(triples_path).resolve())
    if p in _GRAPH_CACHE:
        return _GRAPH_CACHE[p]
    with open(p, "r", encoding="utf-8") as f:
        triples = json.load(f)
    g = build_graph_from_triples(triples)
    _GRAPH_CACHE[p] = g
    return g


def clear_cache():
    _GRAPH_CACHE.clear()
