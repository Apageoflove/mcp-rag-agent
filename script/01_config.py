"""全局配置：路径、模型、检索参数都集中放这儿，别的脚本统一 import。"""

import os
from pathlib import Path
from dotenv import load_dotenv


# 算一下项目根目录：本文件在 script/ 下，再往上两层就是项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

# 几个常用目录。data 放原始 PDF，output 放中间产物和向量库
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "Xorbits" / "bge-m3"
CHROMA_DIR = PROJECT_ROOT / "output" / "chromadb"
NEO4J_DIR = PROJECT_ROOT / "output" / "neo4j"
LOG_DIR = PROJECT_ROOT / "output" / "logs"

# 目录不存在就建一下，免得到后面写文件才报错
for d in [DATA_DIR, CHROMA_DIR, NEO4J_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---- LLM（走 MiniMax 的 OpenAI 兼容接口）----
# 密钥只从 .env 读，没配就让它直接报错，不在代码里留 fallback
LLM_API_KEY = os.getenv("MINIMAX_API_KEY")
LLM_API_BASE = os.getenv("MINIMAX_API_HOST", "https://api.minimaxi.com/v1")
LLM_MODEL = "MiniMax-M3"
LLM_BASE_URL = f"{LLM_API_BASE}/v1"

# ---- Embedding（bge-m3，本地加载）----
EMBEDDING_MODEL_PATH = str(MODEL_DIR)
EMBEDDING_DIMENSION = 1024  # bge-m3 输出固定 1024 维
# 官方推荐的中文 query 前缀，加上之后检索质量会好一些
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

CHROMA_COLLECTION = "documents"

# 分块：500 字符一块、重叠 50，这是试下来召回和噪声比较平衡的一组
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVE_TOP_K = 5

# Neo4j 连接（本地 docker 起的）
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

# Agent 相关：置信度低于 0.7 就让 reflection 触发一次重检索，最多再试 2 轮防止死循环
REFLECTION_THRESHOLD = 0.7
MAX_RETRY = 2
