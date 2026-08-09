"""02_pdf_parser 纯函数单测。

02 import 时会 import pdfplumber，所以加了 importorskip：本地装了 pdfplumber
才跑，CI 没装就跳过。

只测不打开 PDF 的纯函数：表格真伪判定、公式分离、标题识别、页眉页脚去除、
区域合并、字符→文本拼接。
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

_SCRIPT = Path(__file__).resolve().parent.parent / "script"


def _load():
    spec = importlib.util.spec_from_file_location("p2", _SCRIPT / "02_pdf_parser.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p2 = _load()


# ─────────────── is_real_table ───────────────

class TestIsRealTable:
    def test_real(self):
        assert p2.is_real_table([["A", "B"], ["1", "2"]]) is True

    def test_one_row(self):
        assert p2.is_real_table([["A", "B"]]) is False

    def test_all_empty(self):
        assert p2.is_real_table([["", ""], ["", ""]]) is False

    def test_too_sparse(self):
        # 4 格空 3 个 = 75% 空 → 不是真表格
        assert p2.is_real_table([["A", "B"], ["", ""]]) is False

    def test_acceptably_sparse(self):
        # 25% 空 → 还算真表格
        assert p2.is_real_table([["A", "B"], ["", "2"]]) is True


# ─────────────── _fix_title_spacing ───────────────

class TestFixTitleSpacing:
    def test_camel_split(self):
        assert p2._fix_title_spacing("DataCuration") == "Data Curation"

    def test_hyphen_preserved(self):
        # 连字符分段处理，分段内再做 camel 拆分，但 "self"/"Attention" 段内无 camel
        assert p2._fix_title_spacing("self-Attention") == "self-Attention"
        assert p2._fix_title_spacing("Mixture-of-Experts") == "Mixture-of-Experts"


# ─────────────── detect_formulas ───────────────

class TestDetectFormulas:
    def test_separates_formulas(self):
        text = "This is normal text.\n(cid:123)\n𝛼 + 𝛽 = 𝛾\nshort\n"
        clean, forms = p2.detect_formulas(text)
        assert "This is normal text." in clean
        assert "short" in clean
        assert "(cid:123)" in forms
        assert "𝛼 + 𝛽 = 𝛾" in forms

    def test_normal_text_unchanged(self):
        clean, forms = p2.detect_formulas("Just a normal sentence about models.")
        assert forms == []
        assert "normal sentence" in clean


# ─────────────── merge_regions ───────────────

class TestMergeRegions:
    def test_merges_same_page(self):
        figs = [
            {"page": 1, "type": "矢量图", "bbox": (0.0, 0.0, 10.0, 10.0)},
            {"page": 1, "type": "位图", "bbox": (5.0, 5.0, 20.0, 20.0)},
        ]
        out = p2.merge_regions(figs)
        assert len(out) == 1
        assert out[0]["page"] == 1
        # 合并后的 bbox 是两个区域的外接矩形
        assert out[0]["bbox"] == (0.0, 0.0, 20.0, 20.0)

    def test_keeps_no_bbox_fig(self):
        figs = [{"page": 2, "type": "矢量图", "bbox": None}]
        out = p2.merge_regions(figs)
        assert len(out) == 1
        assert out[0]["bbox"] is None


# ─────────────── remove_header_footer ───────────────

class TestRemoveHeaderFooter:
    def test_strips_repeated_header_footer(self):
        pages = [
            "Header\nbody one here\nFooter 1",
            "Header\nbody two here\nFooter 2",
            "Header\nbody three here\nFooter 3",
        ]
        out = p2.remove_header_footer(pages)
        assert out == ["body one here", "body two here", "body three here"]

    def test_short_doc_unchanged(self):
        # 不到 3 页不去页眉页脚
        assert p2.remove_header_footer(["only one page"]) == ["only one page"]


# ─────────────── _line_matches_pattern ───────────────

class TestLineMatchesPattern:
    def test_dynamic_page_number(self):
        assert p2._line_matches_pattern("Page 5", "Page {N}") is True
        assert p2._line_matches_pattern("Page 12", "Page {N}") is True

    def test_pure_number(self):
        assert p2._line_matches_pattern("12", "{N}") is True

    def test_mismatch(self):
        assert p2._line_matches_pattern("Foo", "Page {N}") is False

    def test_empty_pattern(self):
        assert p2._line_matches_pattern("anything", "") is False


# ─────────────── extract_headers ───────────────

class TestExtractHeaders:
    def test_h1(self):
        out = p2.extract_headers(["1 Introduction\nsome body text"])
        assert out == [{"level": "h1", "title": "1 Introduction", "page": 1, "line": 0}]

    def test_h2(self):
        out = p2.extract_headers(["2.1 Attention Mechanism\nbody"])
        assert len(out) == 1
        assert out[0]["level"] == "h2"
        assert out[0]["title"] == "2.1 Attention Mechanism"

    def test_references_not_header(self):
        assert p2.extract_headers(["References\n[1] foo bar"]) == []

    def test_high_numbered_ref_filtered(self):
        # 编号 > 10 的通常是参考文献条目，不当标题
        assert p2.extract_headers(["11. Some Paper Title Here"]) == []


# ─────────────── _chars_to_text ───────────────

class TestCharsToText:
    def test_gap_inserts_space(self):
        chars = [
            {"top": 100.0, "x0": 50.0, "x1": 60.0, "text": "A"},
            {"top": 100.0, "x0": 80.0, "x1": 90.0, "text": "B"},
        ]
        # 字符间距 > 1.8pt → 中间补空格
        assert p2._chars_to_text(chars) == "A B"

    def test_multiline(self):
        chars = [
            {"top": 100.0, "x0": 50.0, "x1": 60.0, "text": "A"},
            {"top": 120.0, "x0": 50.0, "x1": 60.0, "text": "C"},
        ]
        assert p2._chars_to_text(chars) == "A\nC"

    def test_empty(self):
        assert p2._chars_to_text([]) == ""
