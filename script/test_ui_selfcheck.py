"""UI/接口类脚本启动自检：19_web_app / 20_api_server / 21_kg_visualizer。

验收方式（用户已确认）：导入成功 + 关键接口可调用 + 不报错即通过。
不跑实际推理（推理准确度由 11-17 各自的 F1 测试覆盖）。
"""
import os
# API host 在 NO_PROXY 里，SOCKS 代理(httpx 需 socksio)对本测试无必要，先清掉
for k in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader

print("=" * 60)
print("UI/接口类脚本启动自检")
print("=" * 60)

results = []

# ── 19_web_app：Gradio 应用构造 ────────────────────────
try:
    m19 = SourceFileLoader("m19", str(Path(__file__).resolve().parent / "19_web_app.py")).load_module()
    app = m19.build_app()
    assert app is not None, "build_app 返回 None"
    # ask("","(无)",5) 应返回提示而非抛异常
    ans, src, trace = m19.ask("", "(无)", 5)
    assert "请输入" in ans
    results.append(("19_web_app (Gradio 构造 + ask 防空)", True, ""))
    print("  ✅ 19_web_app: build_app() 成功, ask() 防空处理正常")
except Exception as e:
    results.append(("19_web_app", False, f"{type(e).__name__}: {e}"))
    print(f"  ❌ 19_web_app: {type(e).__name__}: {e}")

# ── 20_api_server：FastAPI 路由注册 + /health ──────────
try:
    m20 = SourceFileLoader("m20", str(Path(__file__).resolve().parent / "20_api_server.py")).load_module()
    app = m20.app
    routes = {r.path for r in app.routes}
    for must in ("/health", "/query", "/upload", "/graph", "/status"):
        assert must in routes, f"缺少路由 {must}"
    # 直接调 status()（不经过 HTTP）
    st = m20.status()
    assert st["status"] == "ok"
    results.append(("20_api_server (FastAPI 路由 + /status)", True, ""))
    print(f"  ✅ 20_api_server: 路由 {sorted(routes)} 齐全, /status 返回 ok")
except Exception as e:
    results.append(("20_api_server", False, f"{type(e).__name__}: {e}"))
    print(f"  ❌ 20_api_server: {type(e).__name__}: {e}")

# ── 21_kg_visualizer：渲染一个 HTML ────────────────────
try:
    m21 = SourceFileLoader("m21", str(Path(__file__).resolve().parent / "21_kg_visualizer.py")).load_module()
    kg_dir = Path(__file__).resolve().parent.parent.parent / "output" / "kg_triples"
    triples = str(kg_dir / "MiniMax_M1_tech_report_test.json")
    html_path = m21.render_graph(triples, max_nodes=20)
    assert Path(html_path).exists() and Path(html_path).stat().st_size > 0
    results.append(("21_kg_visualizer (pyvis 渲染)", True, ""))
    print(f"  ✅ 21_kg_visualizer: 渲染 HTML ({Path(html_path).name}, "
          f"{Path(html_path).stat().st_size} 字节)")
except Exception as e:
    results.append(("21_kg_visualizer", False, f"{type(e).__name__}: {e}"))
    print(f"  ❌ 21_kg_visualizer: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
n_ok = sum(1 for _, ok, _ in results if ok)
n_all = len(results)
print(f"通过: {n_ok}/{n_all}")
for name, ok, err in results:
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  — {err}" if err else ""))
if n_ok == n_all:
    print("✅ UI/接口类三脚本全部启动自检 PASS")
else:
    print("❌ 未达标")
print("=" * 60)
sys.exit(0 if n_ok == n_all else 1)
