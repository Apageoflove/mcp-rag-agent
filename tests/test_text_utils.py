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
