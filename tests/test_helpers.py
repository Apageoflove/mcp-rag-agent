"""纯函数单元测试。

只覆盖不依赖本地模型 / API / PDF 数据的模块（_eval_helpers、03_chunker），
所以不用下 7G 的模型权重也能跑。直接 `pytest` 即可。
"""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "script"


def _load(filename, modname):
    """数字开头的文件没法直接 import，用 importlib 按路径加载。"""
    spec = importlib.util.spec_from_file_location(modname, _SCRIPT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eh = _load("_eval_helpers.py", "_eval_helpers")
ck = _load("03_chunker.py", "_chunker")


# ─────────────── _eval_helpers._normalize_text ───────────────

class TestNormalize:
    def test_empty(self):
        assert eh._normalize_text("") == ""
        assert eh._normalize_text(None) == ""

    def test_lowercase_and_punct(self):
        assert eh._normalize_text("BERT, GPT!") == "bert gpt"

    def test_strip_accents(self):
        assert eh._normalize_text("café Héllo") == "cafe hello"

    def test_digit_commas_removed(self):
        # 534,700 -> 534700，方便数字对齐
        assert eh._normalize_text("$534,700") == "534700"

    def test_inline_hyphen_linebreak(self):
        # 跨行连字符 en-\ncoder -> encoder
        assert eh._normalize_text("en-\ncoder") == "encoder"

    def test_letter_hyphens_collapsed(self):
        assert eh._normalize_text("Mixture-of-Experts") == "mixtureofexperts"

    def test_whitespace_fold(self):
        assert eh._normalize_text("  a   b  ") == "a b"


# ─────────────── keyword_hit_ratio ───────────────

class TestKeywordHit:
    def test_case_insensitive(self):
        hit, total, missed = eh.keyword_hit_ratio(["BERT", "GPT"], "bert and gpt")
        assert (hit, total, missed) == (2, 2, [])

    def test_accent_insensitive(self):
        hit, _, _ = eh.keyword_hit_ratio(["café"], "I love CAFE")
        assert hit == 1

    def test_missed_returned(self):
        hit, total, missed = eh.keyword_hit_ratio(["xyz", "abc"], "only abc here")
        assert hit == 1
        assert total == 2
        assert missed == ["xyz"]

    def test_empty_keyword_skipped(self):
        # 空关键词不计入总数，也不会被当成命中
        hit, total, _ = eh.keyword_hit_ratio(["", "bert"], "bert")
        assert (hit, total) == (1, 1)


# ─────────────── compute_f1 / aggregate_f1 ───────────────

class TestF1:
    def test_perfect(self):
        assert eh.compute_f1({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)

    def test_both_empty(self):
        assert eh.compute_f1(set(), set()) == (1.0, 1.0, 1.0)

    def test_disjoint(self):
        assert eh.compute_f1({"a"}, {"b"}) == (0.0, 0.0, 0.0)

    def test_partial(self):
        p, r, f1 = eh.compute_f1({"a", "b"}, {"a", "b", "c"})
        assert p == 1.0
        assert abs(r - 2 / 3) < 1e-6
        assert abs(f1 - 0.8) < 1e-6

    def test_aggregate_empty(self):
        assert eh.aggregate_f1([]) == {"avg_f1": 0.0, "pass_rate": 0.0, "n": 0}

    def test_aggregate_stats(self):
        r = eh.aggregate_f1([1.0, 0.5, 0.0])
        assert r["n"] == 3
        assert abs(r["avg_f1"] - 0.5) < 1e-6
        assert abs(r["pass_rate"] - 1 / 3) < 1e-6


# ─────────────── 03_chunker.smart_chunk ───────────────

class TestSmartChunk:
    def test_empty(self):
        assert ck.smart_chunk("") == []
        assert ck.smart_chunk("   ") == []

    def test_single_sentence(self):
        assert ck.smart_chunk("短句。") == ["短句。"]

    def test_overlap_between_chunks(self):
        chunks = ck.smart_chunk("第一句。第二句。", max_size=5, overlap=2)
        assert len(chunks) == 2
        # 第二块开头得带上一块尾部 overlap
        assert chunks[1].startswith(chunks[0][-2:])

    def test_no_terminator_no_split(self):
        # 没句末标点的长文本不会被硬切，这是当前行为，记一下免得以后误改
        long = "a" * 600
        assert ck.smart_chunk(long, max_size=100) == [long]


# ─────────────── 03_chunker 表格相关 ───────────────

class TestTable:
    def test_code_table_detected(self):
        data = [["def x", "import y"], ["return z", "# comment"]]
        assert ck.is_code_table(data) is True

    def test_normal_table_not_code(self):
        data = [["Name", "Age"], ["Bob", "30"]]
        assert ck.is_code_table(data) is False

    def test_table_to_chunk_vertical(self):
        c = ck.table_to_chunk([["A", "B"], ["1", "2"]], 3)
        assert c["page"] == 3
        assert c["type"] == "table"
        assert "A: 1" in c["text"] and "B: 2" in c["text"]

    def test_table_to_chunk_too_short(self):
        assert ck.table_to_chunk([["A"]], 1) is None

    def test_table_to_chunk_empty_rows(self):
        assert ck.table_to_chunk([["A", "B"], ["", " "]], 1) is None


# ─────────────── 03_chunker.fix_english_spacing ───────────────

class TestSpacing:
    def test_camel_split(self):
        assert ck.fix_english_spacing("DataCuration") == "Data Curation"

    def test_hyphen_preserved(self):
        assert ck.fix_english_spacing("self-attention") == "self-attention"

    def test_short_unchanged(self):
        assert ck.fix_english_spacing("Hi") == "Hi"

    def test_no_letter_unchanged(self):
        assert ck.fix_english_spacing("123") == "123"


# ─────────────── 03_chunker._fallback_section ───────────────

class TestFallbackSection:
    def test_abstract(self):
        assert ck._fallback_section("Abstract\nWe study...") == "Abstract"

    def test_introduction(self):
        assert ck._fallback_section("1. Introduction\nFoo") == "1. Introduction"

    def test_numbered_section(self):
        assert ck._fallback_section("2. Method\nbody") == "Method"

    def test_long_first_line(self):
        text = "Some very long title that is definitely over forty chars"
        assert ck._fallback_section(text) == "Abstract / 论文摘要"

    def test_short_first_line(self):
        assert ck._fallback_section("Short title") == "Short title"
