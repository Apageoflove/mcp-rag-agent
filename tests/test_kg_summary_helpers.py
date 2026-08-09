"""09_kg_builder 与 11_summary_indexer 的纯函数单测。

这俩模块是 Graph RAG 建图 / 层次摘要的确定性核心，逻辑比较绕（实体消歧三层
过滤、关键句抽取式摘要），一直没测。只测不调 LLM / 不连 Neo4j 的纯逻辑。

两个模块 import 时都会 import openai（09 顶部、11 经 config），所以加了
importorskip，CI 没装 openai 时跳过，本地装了才跑。
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("openai")

_SCRIPT = Path(__file__).resolve().parent.parent / "script"


def _load(fn, name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT / fn)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m09 = _load("09_kg_builder.py", "m09_test")
m11 = _load("11_summary_indexer.py", "m11_test")


# ─────────────── 09: 关系归一化 ───────────────

class TestNormalizeRelation:
    def test_canonical_synonyms(self):
        assert m09.normalize_relation("uses") == "uses"
        assert m09.normalize_relation("utilize") == "uses"
        assert m09.normalize_relation("powered by") == "uses"
        assert m09.normalize_relation("develops") == "proposes"
        assert m09.normalize_relation("outperform") == "outperforms"

    def test_case_insensitive(self):
        assert m09.normalize_relation("Utilizes") == "uses"
        assert m09.normalize_relation("PROPOSES") == "proposes"

    def test_unknown_kept_as_is(self):
        assert m09.normalize_relation("some weird rel") == "some weird rel"


# ─────────────── 09: 实体相似度工具 ───────────────

class TestEntitySim:
    def test_substring_either_direction(self):
        assert m09.is_substring("M1", "MiniMax-M1") is True
        assert m09.is_substring("MiniMax-M1", "M1") is True  # 反向也算
        assert m09.is_substring("foo", "bar") is False

    def test_string_similarity(self):
        # 大小写/空格差异应高相似度
        assert m09.string_similarity("MiniMax-M1", "minimax m1") > 0.85
        # 完全不同则低
        assert m09.string_similarity("BERT", "ResNet") < 0.5


class TestHasAmbiguity:
    def test_ambiguous_short(self):
        # BGE 同时是 BGE-M3 和 BGE-Large 的子串 → 歧义
        ents = ["BGE", "BGE-M3", "BGE-Large", "ResNet"]
        assert m09.has_ambiguity("BGE", ents) is True

    def test_unambiguous(self):
        ents = ["BGE", "BGE-M3", "BGE-Large", "ResNet"]
        assert m09.has_ambiguity("ResNet", ents) is False


# ─────────────── 09: 标准名选取 ───────────────

class TestPickCanonical:
    def test_frequency_wins(self):
        freq = {"M1": 5, "MiniMax-M1": 2}
        assert m09.pick_canonical_name(["M1", "MiniMax-M1"], freq) == "M1"

    def test_tie_breaks_to_longest(self):
        freq = {"M1": 2, "MiniMax-M1": 2}
        assert m09.pick_canonical_name(["M1", "MiniMax-M1"], freq) == "MiniMax-M1"


# ─────────────── 09: 引文关系过滤 ───────────────

class TestCitationRelation:
    def test_citation_detected(self):
        for r in ["authored", "Authors", "cited", "references"]:
            assert m09._is_citation_relation(r) is True

    def test_non_citation(self):
        assert m09._is_citation_relation("uses") is False
        assert m09._is_citation_relation("proposes") is False


# ─────────────── 09: 三元组归一化去重 ───────────────

class TestNormalizeTriples:
    def test_dedup_accumulates_count(self):
        triples = [
            {"subject": "A", "relation": "uses", "object": "B", "count": 1, "source_page": 1},
            {"subject": "A", "relation": "uses", "object": "B", "count": 2, "source_page": 2},
        ]
        out = m09.normalize_triples(triples, {"A": "A", "B": "B"})
        assert len(out) == 1
        assert out[0]["count"] == 3

    def test_drops_citation_relations(self):
        triples = [
            {"subject": "A", "relation": "uses", "object": "B", "count": 1},
            {"subject": "X", "relation": "authored", "object": "Y", "count": 1},
        ]
        out = m09.normalize_triples(triples, {"A": "A", "B": "B", "X": "X", "Y": "Y"})
        assert len(out) == 1
        assert out[0]["subject"] == "A"

    def test_relation_synonyms_merged(self):
        # utilize → uses，所以和已有的 uses(A,B) 是不同三元组（object 不同）
        triples = [
            {"subject": "A", "relation": "utilize", "object": "C", "count": 1},
        ]
        out = m09.normalize_triples(triples, {"A": "A", "C": "C"})
        assert out[0]["relation"] == "uses"

    def test_entity_mapping_applied(self):
        # 消歧后 C → A，三条同 object 边应合并
        triples = [
            {"subject": "X", "relation": "uses", "object": "C", "count": 1},
            {"subject": "X", "relation": "uses", "object": "A", "count": 1},
        ]
        out = m09.normalize_triples(triples, {"X": "X", "C": "A", "A": "A"})
        assert len(out) == 1
        assert out[0]["count"] == 2


# ─────────────── 11: 句子切分 ───────────────

class TestSplitSentences:
    def test_basic_split(self):
        txt = ("First sentence is long enough to pass. "
               "Second one also passes the length threshold!")
        out = m11._split_sentences(txt)
        assert len(out) == 2

    def test_short_dropped(self):
        # 短于 25 字符的句子被丢
        txt = "Short. This one is definitely long enough to pass the threshold."
        out = m11._split_sentences(txt)
        assert all(len(s) > 25 for s in out)

    def test_empty(self):
        assert m11._split_sentences("") == []
        assert m11._split_sentences("   ") == []


# ─────────────── 11: 关键词抽取 ───────────────

class TestExtractKeywords:
    def test_proper_nouns_and_numbers(self):
        kw = m11._extract_keywords("BERT uses 512 H800 GPUs in 2024. The model achieves 95% accuracy.")
        assert "bert" in kw
        assert "512" in kw
        assert "achieves" in kw

    def test_year_excluded(self):
        kw = m11._extract_keywords("The paper from 2024 shows results.")
        # 4 位年份是引文噪声，被排除
        assert "2024" not in kw


# ─────────────── 11: 句子打分 ───────────────

class TestScoreSentences:
    def test_keyword_hits_raise_score(self):
        sents = [
            "The model is great and works well.",            # 少关键词
            "BERT uses transformers architecture heavily.",   # 多关键词
        ]
        scored = m11._score_sentences(sents, m11._extract_keywords("BERT uses transformers"))
        scored_dict = dict((s, sc) for sc, s in scored)
        # 含关键词的句子得分更高
        assert scored_dict["BERT uses transformers architecture heavily."] > \
               scored_dict["The model is great and works well."]


# ─────────────── 11: 抽取式摘要 ───────────────

class TestExtractiveSummary:
    def test_short_text_returned_as_is(self):
        text = "Just a couple sentences here that are under the limit."
        out = m11.extractive_summary(text, max_sentences=10)
        assert out == text

    def test_respects_max_sentences(self):
        text = ("BERT was proposed by Google. It uses transformers. "
                "The model has parameters. Training took days. "
                "It achieved SOTA. The dataset was large. "
                "Evaluation was thorough. Results were impressive. ") * 2
        out = m11.extractive_summary(text, max_sentences=3)
        assert len(m11._split_sentences(out)) <= 3 + 1  # 贪心 + 可能的稀有补捞
