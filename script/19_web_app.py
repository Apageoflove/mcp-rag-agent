"""Gradio Web 界面：多模态 RAG 智能体的浏览器交互入口。

左侧：选 PDF + 提问；右侧：答案 + 置信度 + 来源溯源 + 推理链路；
底部：知识图谱可视化（21_kg_visualizer 渲染的 HTML，iframe 内嵌）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m16 = SourceFileLoader("m16", str(Path(__file__).resolve().parent / "16_agent_orchestrator.py")).load_module()
m21 = SourceFileLoader("m21", str(Path(__file__).resolve().parent / "21_kg_visualizer.py")).load_module()

import gradio as gr

CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "chunks"
KG_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "kg_triples"


def list_sources():
    if not CHUNKS_DIR.exists():
        return ["(无)"]
    out = sorted(p.stem.replace(".pdf", "") + ".pdf" for p in CHUNKS_DIR.glob("*.json"))
    return out or ["(无)"]


def list_triples():
    if not KG_DIR.exists():
        return []
    return sorted(p.name for p in KG_DIR.glob("*.json"))


def ask(question: str, source: str, top_k: int):
    if not question.strip():
        return "请输入问题。", "", ""
    sf = None if source in ("(无)", "全部", "") else source
    try:
        r = m16.answer(question, source_filter=sf, top_k=int(top_k))
    except Exception as e:
        return f"[错误] {type(e).__name__}: {e}", "", ""
    src_lines = []
    for i, p in enumerate(r["passages"], 1):
        meta = p.get("metadata", {})
        src_lines.append(f"[{i}] [{meta.get('source','')[:28]}] p{meta.get('page','')} "
                         f"{meta.get('section','')[:28]}")
        src_lines.append(f"    {p.get('text','')[:140]}...")
    src_text = "\n".join(src_lines) if src_lines else "(无来源)"
    trace_lines = [f"- 路由: type={r['route']['type']}, tools={r['route']['tools']}, "
                   f"hops={r['route']['max_hops']}"]
    for step in r["trace"]:
        trace_lines.append(f"- {step['agent']}: " +
                           ", ".join(f"{k}={v}" for k, v in step.items()
                                     if k != "agent"))
    header = (f"**置信度**: {r['confidence']}  |  **来源**: {r['source']}  |  "
              f"**重试**: {r['retries']}  |  **耗时**: {r['time']}s\n\n")
    return header + r["answer"], src_text, "\n".join(trace_lines)


def render_kg(triples_name: str):
    if not triples_name:
        return None
    try:
        return m21.render_graph(str(KG_DIR / triples_name), max_nodes=50)
    except Exception as e:
        return None


def build_app():
    with gr.Blocks(title="多模态 RAG 智能体", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 多模态 RAG 增强智能体系统\n"
                    "GraphRAG + 自反思 + 自一致性推理。左侧提问，右侧看答案/来源/推理链。")
        with gr.Row():
            with gr.Column(scale=1):
                inp_q = gr.Textbox(label="问题", placeholder="例: What does MiniMax-M1 use?",
                                   lines=2)
                inp_src = gr.Dropdown(choices=list_sources(), label="限定 PDF",
                                      value=None, interactive=True)
                inp_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                btn = gr.Button("提问", variant="primary")
            with gr.Column(scale=2):
                out_ans = gr.Markdown(label="答案")
                with gr.Accordion("来源溯源", open=False):
                    out_src = gr.Textbox(label="来源", lines=10, interactive=False)
                with gr.Accordion("推理链路", open=False):
                    out_trace = gr.Textbox(label="Agent 链", lines=10, interactive=False)
        btn.click(ask, [inp_q, inp_src, inp_k], [out_ans, out_src, out_trace])

        gr.Markdown("---\n### 知识图谱可视化")
        with gr.Row():
            kg_dd = gr.Dropdown(choices=list_triples(), label="选择三元组",
                                interactive=True)
            kg_btn = gr.Button("渲染图谱")
        kg_html = gr.HTML("（点击渲染图谱）")
        kg_btn.click(render_kg, [kg_dd], [kg_html])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860, share=False)
