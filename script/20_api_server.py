"""FastAPI 后端：提供 /query /upload /graph /status 接口。

封装 16 编排器与 21 可视化，对外 REST。可用 uvicorn 启动：
    uvicorn 20_api_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.machinery import SourceFileLoader

m16 = SourceFileLoader("m16", str(Path(__file__).resolve().parent / "16_agent_orchestrator.py")).load_module()
m21 = SourceFileLoader("m21", str(Path(__file__).resolve().parent / "21_kg_visualizer.py")).load_module()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="多模态 RAG 智能体 API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class QueryReq(BaseModel):
    question: str
    source: str | None = None
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryReq):
    """端到端问答：路由→检索→推理(自一致性)→反思。"""
    try:
        r = m16.answer(req.question, source_filter=req.source, top_k=req.top_k)
        return {
            "answer": r["answer"],
            "confidence": r["confidence"],
            "source": r["source"],
            "retries": r["retries"],
            "time": r["time"],
            "route": {"type": r["route"]["type"], "tools": r["route"]["tools"]},
            "n_passages": len(r["passages"]),
            "passages": [{"text": p.get("text", "")[:300],
                          "source": p.get("metadata", {}).get("source", ""),
                          "page": p.get("metadata", {}).get("page", "")}
                         for p in r["passages"]],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[:400]}


@app.post("/upload")
def upload(filename: str):
    """声明已入库的文档（实际上传/解析/入库由 02-04 流水线离线完成）。"""
    p = Path(__file__).resolve().parent.parent.parent / "output" / "chunks" / f"{filename}.json"
    if not p.exists():
        return {"error": f"未找到 {filename} 的分块，请先跑解析入库流水线"}
    import json
    chunks = json.load(open(p, encoding="utf-8"))
    return {"status": "ok", "source": filename, "n_chunks": len(chunks)}


@app.get("/graph")
def graph(triples: str = "MiniMax_M1_tech_report_test.json", max_nodes: int = 60):
    """渲染知识图谱 HTML（data URI 返回，便于前端 iframe 内嵌）。"""
    try:
        kg_dir = Path(__file__).resolve().parent.parent.parent / "output" / "kg_triples"
        html = m21.render_graph(str(kg_dir / triples), max_nodes=max_nodes)
        return {"status": "ok", "html_path": html}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/status")
def status():
    """服务自检：报告可用后端与模块。"""
    from _memory_graph import InMemoryGraph
    return {
        "status": "ok",
        "modules": ["router", "retriever", "reasoner(self-consistency)",
                    "reflection", "orchestrator", "eval", "kg_visualizer"],
        "memory_graph_backend": InMemoryGraph.__name__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
