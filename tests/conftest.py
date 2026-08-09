"""pytest 共享配置：把 script/ 加进 sys.path，方便测试 import。

另外，被测的纯函数（路由分类、忠实度判定、Cypher 校验等）本身不调
OpenAI / pdfplumber，只是 import 链会拉起它们。这里在缺失时注入轻量
stub，让这些测试在最小 CI（只装 pytest）里也能真跑，不用 importorskip
干跳过——回归覆盖面从 ~65 提升到 ~140。
"""
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "script"
for _p in (str(_ROOT), str(_SCRIPT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _stub_if_missing(name: str, **attrs):
    """依赖没装时往 sys.modules 塞个最小 stub，让 import 链不断。

    只 stub 掉模块级别的 class/function 引用（如 openai.OpenAI），
    被测的纯函数不会真正调用它们。
    """
    if name in sys.modules:
        return  # 真的装了，别覆盖
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


# openai: 05_llm_client 顶部 from openai import OpenAI
class _StubOpenAI:
    def __init__(self, *a, **kw):
        pass


_stub_if_missing("openai", OpenAI=_StubOpenAI)

# pdfplumber: 02_pdf_parser 顶部 import pdfplumber
_stub_if_missing("pdfplumber")

# dotenv: 01_config 顶部 from dotenv import load_dotenv
# CI 没装 python-dotenv，load_dotenv 在测试里不需要真加载 .env
_stub_if_missing("dotenv", load_dotenv=lambda *a, **kw: False)
