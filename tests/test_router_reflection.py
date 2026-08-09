"""路由 Agent / 反思 Agent 的纯逻辑单测。

这俩模块 import 时会顺带 import openai（05_llm_client），所以加了 importorskip：
本地装了 openai 就跑，CI 只装 pytest 时自动跳过，保持绿灯。

只测不调 LLM / 不碰模型的确定性逻辑：
- 12_router_agent：规则意图分类 + 工具/跳数映射
- 15_reflection_agent：断言拆分、证据词抽取、忠实度判定
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("openai")

_SCRIPT = Path(__file__).resolve().parent.parent / "script"


def _load(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, _SCRIPT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load("12_router_agent.py", "rt_router")
rf = _load("15_reflection_agent.py", "rf_reflection")


# ─────────────── 路由：规则意图分类 ───────────────

class TestRuleClassify:
    def test_summary(self):
        assert rt._rule_classify("Summarize this paper") == ("summary", 0.95)
        assert rt._rule_classify("give me the overview") == ("summary", 0.95)

    def test_quantity_is_fact(self):
        # how many 这类归到 fact（细节题）
        assert rt._rule_classify("How many layers in ResNet") == ("fact", 0.95)

    def test_relation(self):
        assert rt._rule_classify("compare bert and gpt") == ("relation", 0.95)

    def test_default_fact_low_confidence(self):
        # 没命中任何规则 → fact，置信度低（会走 LLM 仲裁）
        t, c = rt._rule_classify("what is attention")
        assert t == "fact"
        assert c < 0.9


# ─────────────── 路由：工具 / 跳数映射 ───────────────

class TestRouteMapping:
    def test_summary_tools(self):
        r = rt.route("Summarize this paper", use_llm_fallback=False)
        assert r["type"] == "summary"
        assert r["tools"] == ["summary_index", "vector"]
        assert r["max_hops"] == 1

    def test_fact_tools(self):
        r = rt.route("How many layers", use_llm_fallback=False)
        assert r["type"] == "fact"
        assert r["tools"] == ["vector", "summary_index"]

    def test_relation_uses_graph_and_multi_hop(self):
        r = rt.route("relationship between", use_llm_fallback=False)
        assert r["type"] == "relation"
        assert "graph" in r["tools"]
        assert r["max_hops"] == 3

    def test_image_question_adds_vlm(self):
        # 问句里带 figure/chart 这类词，工具里要加 vlm
        r = rt.route("What does the figure show?", use_llm_fallback=False)
        assert "vlm" in r["tools"]


# ─────────────── 反思：断言拆分 ───────────────

class TestSplitClaims:
    def test_basic_split(self):
        ans = "BERT was proposed by Google. It uses 340M parameters."
        assert rf._split_claims(ans) == [
            "BERT was proposed by Google.",
            "It uses 340M parameters.",
        ]

    def test_strips_supporting_evidence_suffix(self):
        # 14 加的 Supporting evidence / Key entities 附录要被剥掉，不参与校验
        ans = ("Some claim here is long enough.\n"
               "Supporting evidence\n[Passage 1] verbatim quote stuff.")
        claims = rf._split_claims(ans)
        assert len(claims) == 1
        assert "Supporting evidence" not in claims[0]

    def test_citation_markers_removed(self):
        ans = "The model was released in 2018 by Google [Passage 2]."
        claims = rf._split_claims(ans)
        assert "[Passage" not in claims[0]


# ─────────────── 反思：单断言支撑判定 ───────────────

class TestClaimSupported:
    def test_supported(self):
        ok, score, missing = rf._claim_supported(
            "BERT was proposed by Google.", "google proposed bert")
        assert ok is True
        assert missing == []

    def test_wrong_number_unsupported(self):
        # 数字对不上 → 无依据
        ok, score, missing = rf._claim_supported(
            "ResNet has 152 layers.", "resnet has 50 layers")
        assert ok is False
        assert "152" in missing

    def test_wrong_amount_unsupported(self):
        ok, _, missing = rf._claim_supported(
            "It costs $534,700.", "the cost was 100000 dollars")
        assert ok is False
        assert "534,700" in missing


# ─────────────── 反思：verify_answer ───────────────

class TestVerifyAnswer:
    def test_fully_supported(self):
        v = rf.verify_answer(
            "BERT was made by Google. It has 340M parameters.",
            [{"text": "google made bert with 340m parameters"}])
        assert v["faithfulness"] == 1.0
        assert v["n_claims"] == 2
        assert v["unsupported_claims"] == []

    def test_empty_answer_short_circuits(self):
        v = rf.verify_answer("", [{"text": "x"}])
        assert v["faithfulness"] == 1.0
        assert v["n_claims"] == 0
        assert v["needs_retry"] is False


# ─────────────── 反思：证据词抽取顺序 ───────────────

class TestEvidenceTerms:
    def test_order_by_appearance(self):
        # 主语在前：MiniMax-M1 先于 MiniMax（按出现位置）
        terms = rf._evidence_terms("MiniMax-M1 was proposed by MiniMax.")
        assert terms == ["minimax-m1", "minimax"]
