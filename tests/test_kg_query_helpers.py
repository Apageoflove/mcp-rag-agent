"""10_kg_query 纯函数单测：Cypher 安全校验 + 投影标准化 + 锚点推断 + 答案投影。

这些是图谱查询的确定性核心——text2cypher 是 LLM 生成、有随机性，所以
服务端用这套规则把结果约束成稳定输出。逻辑比较绕，锁一下免得改崩。

10 import 会拉起 openai（05_llm_client），importorskip 跳过；10 还依赖
_memory_graph / 09，没装齐就整模块跳过。
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("openai")

_SCRIPT = Path(__file__).resolve().parent.parent / "script"


def _load():
    spec = importlib.util.spec_from_file_location("m10", _SCRIPT / "10_kg_query.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


try:
    m10 = _load()
except Exception:
    pytest.skip("10_kg_query 依赖未就绪，跳过", allow_module_level=True)


# ─────────────── validate_cypher ───────────────

class TestValidateCypher:
    def test_safe_match(self):
        assert m10.validate_cypher("MATCH (a:Entity) RETURN a")[0] is True

    def test_forbidden_create(self):
        ok, reason = m10.validate_cypher("CREATE (a:Entity)")
        assert ok is False
        assert "CREATE" in reason

    def test_semicolon_blocked(self):
        # 禁止多语句
        assert m10.validate_cypher("MATCH (a) RETURN a; DROP")[0] is False

    def test_must_start_with_match(self):
        assert m10.validate_cypher("WITH x RETURN x")[0] is False

    def test_must_have_return(self):
        assert m10.validate_cypher("MATCH (a:Entity)")[0] is False

    def test_empty(self):
        assert m10.validate_cypher("")[0] is False


# ─────────────── standardize_projection ───────────────

class TestStandardizeProjection:
    def test_triple_rewrite_keeps_limit(self):
        cy = ("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
              "WHERE toLower(a.name) CONTAINS 'minimax-m1' AND r.relation = 'uses' "
              "RETURN b.name LIMIT 10")
        std, mode, roles = m10.standardize_projection(cy)
        assert mode == "triple"
        assert "LIMIT 10" in std
        # RETURN 被改写成固定的 subject/predicate/object 三列
        assert "subject" in std and "predicate" in std and "object" in std
        assert roles == {"subject_alias": "a", "object_alias": "b"}

    def test_default_limit_when_missing(self):
        cy = ("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
              "WHERE a.name='x' RETURN b.name")
        std, mode, _ = m10.standardize_projection(cy)
        assert mode == "triple"
        assert "LIMIT 50" in std

    def test_varlen_goes_path(self):
        # 可变长跳数 *1..2 没法标准化成单跳三元组，走 path 模式原样执行
        cy = ("MATCH p=(a:Entity)-[:RELATES_TO*1..2]->(b:Entity) "
              "WHERE toLower(a.name) CONTAINS 'x' RETURN nodes(p) LIMIT 10")
        _, mode, _ = m10.standardize_projection(cy)
        assert mode == "path"

    def test_reverse_arrow_swaps_endpoints(self):
        cy = ("MATCH (a:Entity)<-[r:RELATES_TO]-(b:Entity) "
              "WHERE b.name='x' RETURN a.name")
        _, _, roles = m10.standardize_projection(cy)
        # 箭头从 b 指向 a → subject 是 b（关系发出方），object 是 a
        assert roles["subject_alias"] == "b"
        assert roles["object_alias"] == "a"


# ─────────────── infer_anchored_endpoints ───────────────

class TestInferAnchored:
    def test_one_anchor(self):
        cy = ("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
              "WHERE toLower(a.name) CONTAINS 'minimax-m1' AND r.relation = 'uses' "
              "RETURN b.name LIMIT 10")
        std = m10.standardize_projection(cy)[0]
        assert m10.infer_anchored_endpoints(std) == {"a"}

    def test_two_anchors(self):
        cy = ("MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
              "WHERE a.name='x' AND b.name='y' RETURN r.relation")
        std = m10.standardize_projection(cy)[0]
        assert m10.infer_anchored_endpoints(std) == {"a", "b"}

    def test_no_where_no_anchors(self):
        cy = "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN b.name"
        std = m10.standardize_projection(cy)[0]
        assert m10.infer_anchored_endpoints(std) == set()


# ─────────────── project_answers ───────────────

class TestProjectAnswers:
    ROLES = {"subject_alias": "a", "object_alias": "b"}

    def test_zero_anchor_keeps_both_ends(self):
        recs = [{"subject": "A", "predicate": "uses", "object": "B"}]
        out = m10.project_answers(recs, anchors=set(), roles=self.ROLES, mode="triple")
        assert out == ["A", "B"]

    def test_one_anchor_returns_unanchored_end(self):
        recs = [{"subject": "X", "predicate": "uses", "object": "Y"}]
        # subject 已锚定 → 只返回 object（未知端）
        out = m10.project_answers(recs, anchors={"a"}, roles=self.ROLES, mode="triple")
        assert out == ["Y"]

    def test_two_anchor_includes_predicate(self):
        recs = [{"subject": "A", "predicate": "uses", "object": "B"}]
        out = m10.project_answers(recs, anchors={"a", "b"}, roles=self.ROLES, mode="triple")
        # 两端都已知 → 关系本身才是答案
        assert out == ["A", "B", "uses"]

    def test_path_mode_extracts_entity_names(self):
        recs = [{"entities": [{"name": "A"}, {"name": "B"}],
                 "relations": [{"relation": "uses"}]}]
        out = m10.project_answers(recs, anchors=set(), roles={}, mode="path")
        assert out == ["A", "B"]
