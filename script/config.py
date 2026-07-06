"""让 01_config.py 能被 'from config import xxx' 这样用。

01_config.py 文件名以数字开头，Python 不允许直接 import 数字开头的模块名，
所以这里用 importlib 把它加载进来，再把里面的变量透传出来。
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_config_inner", Path(__file__).resolve().parent / "01_config.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# 把 01_config 里所有公开符号搬到本模块的命名空间
for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)
