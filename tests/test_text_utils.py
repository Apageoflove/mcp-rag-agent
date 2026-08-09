"""_text_utils.strip_think 单测（纯标准库，最小 CI 也能跑）。"""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "script"


def _load():
    spec = importlib.util.spec_from_file_location("tu", _SCRIPT / "_text_utils.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tu = _load()


class TestStripThink:
    def test_no_think_tag_unchanged(self):
        assert tu.strip_think("just a normal answer") == "just a normal answer"

    def test_closed_think_removed(self):
        text = "<think>let me reason</think>final answer"
        assert tu.strip_think(text) == "final answer"

    def test_multiline_closed(self):
        text = "<think>\nline1\nline2\n</think>\nanswer"
        assert tu.strip_think(text) == "answer"

    def test_unclosed_think_removed(self):
        # max_tokens 截断时的情形：裸 <think> 一直写到结尾
        text = "prefix<think>and then it got cut off mid reasoning"
        assert tu.strip_think(text) == "prefix"

    def test_think_at_start(self):
        assert tu.strip_think("<think>x</think>hello") == "hello"

    def test_empty(self):
        assert tu.strip_think("") == ""

    def test_only_think(self):
        assert tu.strip_think("<think>only reasoning</think>") == ""


class TestChunkFilenameToPdf:
    def test_pdf_chunk(self):
        assert tu.chunk_filename_to_pdf("bge_paper.pdf.json") == "bge_paper.pdf"
        assert tu.chunk_filename_to_pdf("resnet.pdf.json") == "resnet.pdf"

    def test_non_pdf_chunk_not_silently_broken(self):
        # 不含 .pdf 的分块文件名，原样返回，不再错加 .pdf
        assert tu.chunk_filename_to_pdf("notes.json") == "notes"
        assert tu.chunk_filename_to_pdf("data.json") == "data"

    def test_no_json_extension(self):
        assert tu.chunk_filename_to_pdf("bge_paper.pdf") == "bge_paper.pdf"


class TestParseRerankScores:
    def test_normal(self):
        import json
        resp = json.dumps([{"index": 0, "score": 5}, {"index": 2, "score": 9}])
        assert tu.parse_rerank_scores(resp, n_candidates=10) == {0: 5, 2: 9}

    def test_duplicate_index_takes_highest(self):
        # LLM 对同一 index 返回两次评分 → 取最高，不重复
        import json
        resp = json.dumps([{"index": 2, "score": 9}, {"index": 2, "score": 8}])
        assert tu.parse_rerank_scores(resp, n_candidates=10) == {2: 9}

    def test_out_of_range_filtered(self):
        import json
        resp = json.dumps([{"index": 0, "score": 5}, {"index": 99, "score": 9}])
        # 越界 index 被丢
        assert tu.parse_rerank_scores(resp, n_candidates=10) == {0: 5}

    def test_think_block_stripped(self):
        resp = '<think>reasoning here</think>[{"index":1,"score":7}]'
        assert tu.parse_rerank_scores(resp, n_candidates=10) == {1: 7}

    def test_fallback_regex_when_not_json_array(self):
        # 整体不是合法 JSON 数组，但能逐个提到 {index,score}
        resp = "I rate them: {\"index\": 0, \"score\": 8} and {\"index\": 3, \"score\": 6}"
        assert tu.parse_rerank_scores(resp, n_candidates=10) == {0: 8, 3: 6}

    def test_no_valid_scores_returns_empty(self):
        assert tu.parse_rerank_scores("totally unparseable", n_candidates=10) == {}
