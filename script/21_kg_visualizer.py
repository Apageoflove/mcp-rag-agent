"""知识图谱可视化模块：从内存图/Neo4j 读取，用 pyvis 生成交互式 HTML。

节点按「关系连接数(度)」着色与定大小——度越大越突出，一眼看出核心实体。
边显示 relation 类型。输出独立 HTML 文件，可嵌入 Gradio 或直接浏览器打开。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m09 = SourceFileLoader("m09", str(Path(__file__).resolve().parent / "09_kg_builder.py")).load_module()

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "kg_visualize"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_pyvis_graph(graph, max_nodes: int = 60, title: str = "Knowledge Graph"):
    """从 InMemoryGraph 或 Neo4j 构造 pyvis Network。"""
    from pyvis.network import Network
    import collections

    net = Network(height="640px", width="100%", notebook=False, directed=True,
                  bgcolor="#ffffff", font_color="#1f2937")
    net.heading = title

    # 收集边（兼容内存图；Neo4j 需先取数，这里只支持内存图对象）
    edges = []
    if hasattr(graph, "_edges"):  # InMemoryGraph
        for e in graph._edges:
            edges.append((e["subject"], e["predicate"], e["object"]))
    degree = collections.Counter()
    for s, _, o in edges:
        degree[s] += 1
        degree[o] += 1

    # 取度最高的 max_nodes 个实体
    top = {n for n, _ in degree.most_common(max_nodes)}
    sub_edges = [(s, p, o) for s, p, o in edges if s in top and o in top]

    # 节点配色：按度分档
    def color_for(d: int) -> str:
        if d >= 6:
            return "#ef4444"   # 红：核心
        if d >= 3:
            return "#f59e0b"   # 橙：重要
        return "#3b82f6"       # 蓝：普通

    for n in top:
        d = degree[n]
        net.add_node(n, label=n, title=f"degree={d}",
                     size=10 + min(d, 10) * 2, color=color_for(d))

    for s, p, o in sub_edges:
        net.add_edge(s, o, label=p, title=p, arrows="to",
                     color="#94a3b8")

    net.set_options(json_options())
    return net


def json_options() -> str:
    return '{"physics":{"barnesHut":{"gravitationalConstant":-8000,"springLength":180}},' \
           '"edges":{"smooth":{"type":"continuous","roundness":0.15}}}'


def render_graph(triples_path: str = None, output_html: str = None,
                 max_nodes: int = 60) -> str:
    """读三元组建内存图 → 渲染 HTML。返回 HTML 路径。"""
    triples_path = triples_path or str(
        Path(__file__).resolve().parent.parent.parent
        / "output" / "kg_triples" / "MiniMax_M1_tech_report_test.json")
    graph = m09.get_memory_graph(triples_path)
    name = Path(triples_path).stem
    net = build_pyvis_graph(graph, max_nodes=max_nodes,
                            title=f"Knowledge Graph — {name}")
    output_html = output_html or str(OUT_DIR / f"{name}.graph.html")
    net.write_html(output_html, notebook=False)
    return output_html


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="知识图谱可视化")
    ap.add_argument("--triples", help="三元组 JSON 路径")
    ap.add_argument("--max-nodes", type=int, default=60)
    args = ap.parse_args()
    out = render_graph(triples_path=args.triples, max_nodes=args.max_nodes)
    print(f"已生成: {out}")
