"""pytest 共享配置：把 script/ 加进 sys.path，方便测试 import。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "script"
for _p in (str(_ROOT), str(_SCRIPT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
