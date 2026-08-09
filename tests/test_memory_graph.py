"""_memory_graph 内置图后端单测。

这个模块是纯标准库（re/collections/json），不带任何重依赖，所以这批测试
在最小 CI（只装 pytest）里也能真跑起来，不用 importorskip。

覆盖：三元组建图、实体名匹配、单跳/多跳查询、标准化 Cypher 解释器。
"""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "script"


def _load():
    spec = importlib.util.spec_from_file_location("mg", _SCRIPT / "_memory_graph.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mg = _load()


def _sample_graph():
    return mg.build_graph_from_triples([
        {"subject": "MiniMax-M1", "relation": "uses", "object": "MoE"},
        {"subject": "MiniMax-M1", "relation": "uses", "object": "CISPO"},
        {"subject": "MoE", "relation": "is a", "object": "architecture"},
        {"subject": "MiniMax-M1", "relation": "proposes", "object": "MiniMax"},
    ])


# ─────────────── 建图与统计 ───────────────

class TestBuildAndStats:
    def test_counts(self):
        g = _sample_graph()
        assert g.node_count() == 5
        assert g.edge_count() == 4

    def test_relation_distribution(self):
        g = _sample_graph()
        d = g.relation_distribution()
        assert d["uses"] == 2
        assert d["is a"] == 1

    def test_skips_incomplete_triples(self):
        # 缺主语/谓词/宾语的三元组直接丢掉
        g = mg.build_graph_from_triples([
            {"subject": "A", "relation": "", "object": "B"},
            {"subject": "A", "relation": "r", "object": "B"},
        ])
        assert g.edge_count() == 1


# ─────────────── 实体名匹配 ───────────────

class TestNameMatches:
    def test_operators_case_insensitive(self):
        g = _sample_graph()
        assert g._name_matches("MiniMax-M1", "contains", "minimax") is True
        assert g._name_matches("MiniMax-M1", "=", "minimax-m1") is True
        assert g._name_matches("MiniMax-M1", "starts with", "Mini") is True
        assert g._name_matches("MiniMax-M1", "ends with", "M1") is True

    def test_no_match(self):
        g = _sample_graph()
        assert g._name_matches("MoE", "contains", "xxx") is False


# ─────────────── 单跳三元组查询 ───────────────

class TestMatchTriples:
    def test_subject_filter(self):
        g = _sample_graph()
        out = g.match_triples(subj_filter=("contains", "minimax"))
        assert len(out) == 3

    def test_relation_filter(self):
        g = _sample_graph()
        out = g.match_triples(rel_filter="uses")
        assert sorted(r["object"] for r in out) == ["CISPO", "MoE"]


# ─────────────── 多跳路径 ───────────────

class TestMatchPaths:
    def test_two_hop_reaches_grandchild(self):
        g = _sample_graph()
        paths = g.match_paths(("contains", "minimax"), max_hops=2, limit=10)
        all_entities = [e for p in paths for e in p["entities"]]
        # 2 跳能到 architecture（MiniMax-M1 -> MoE -> architecture）
        assert "architecture" in all_entities

    def test_one_hop_only_direct_neighbors(self):
        g = _sample_graph()
        paths = g.match_paths(("contains", "minimax"), max_hops=1, limit=10)
        for p in paths:
            assert len(p["entities"]) == 2


# ─────────────── 标准化 Cypher 解释器 ───────────────

class TestExecuteStandardizedCypher:
    def test_subject_and_relation_filters(self):
        g = _sample_graph()
        cy = ("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
              "WHERE toLower(a.name) CONTAINS 'minimax-m1' AND r.relation = 'uses' "
              "RETURN a.name AS subject, r.relation AS predicate, b.name AS object LIMIT 10")
        recs = mg.execute_standardized_cypher(cy, g)
        assert sorted(r["object"] for r in recs) == ["CISPO", "MoE"]

    def test_varlen_deferred_to_path_channel(self):
        # 变长跳数查询不在单跳通道处理，返回空交给 path 通道
        g = _sample_graph()
        cy = ("MATCH p=(a:Entity)-[:RELATES_TO*1..2]->(b:Entity) "
              "WHERE toLower(a.name) CONTAINS 'x' RETURN nodes(p) LIMIT 10")
        assert mg.execute_standardized_cypher(cy, g) == []


class TestExecutePathCypher:
    def test_multi_hop_paths(self):
        g = _sample_graph()
        cy = ("MATCH p=(a:Entity)-[:RELATES_TO*1..2]->(b:Entity) "
              "WHERE toLower(a.name) CONTAINS 'minimax-m1' RETURN nodes(p) LIMIT 10")
        paths = mg.execute_path_cypher(cy, g)
        # 能拼出 MiniMax-M1 -> MoE -> architecture 这条 2 跳路径
        joined = [" ".join(p["entities"]) for p in paths]
        assert any("architecture" in j for j in joined)


# ─────────────── 辅助函数 ───────────────

class TestHelpers:
    def test_parse_string_literal(self):
        assert mg._parse_string_literal("'foo'") == "foo"
        assert mg._parse_string_literal('"bar"') == "bar"

    def test_parse_relation_list(self):
        assert mg._parse_relation_list("'a','b'") == ["a", "b"]
