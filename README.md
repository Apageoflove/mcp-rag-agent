# 多模态 RAG 增强智能体系统

> 把论文变成可问答、可溯源知识库的检索增强生成（RAG）智能体：混合检索 + 知识图谱多跳推理 + 多 Agent 编排，按 MCP 协议对外提供工具，并用 RAGAS 做标准评估。

![Python](https://img.shields.io/badge/Python-3.12-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![RAG](https://img.shields.io/badge/RAG-GraphRAG-orange)

---

## 这是什么

输入一篇 PDF 论文，系统能：

- 解析论文成结构化文本（图表 / 公式 / 标题分离）
- 建向量库 + 知识图谱 + 层次摘要三层索引
- 回答关于论文的问题（事实 / 关系 / 总结），答案带来源页码和章节
- 多跳推理（"A 和 B 是什么关系"）
- 理解论文里的图表（多模态）
- 自检答案可信度（反思 + token 级幻觉检测）

和"直接调 LLM"的区别：这个系统不靠 LLM 硬背，而是**先从论文里检索证据再生成**，答案可溯源、幻觉可控。

---

## 效果

在 4 篇不同领域的公开论文上跑端到端问答，用金标准关键词命中率算 F1：

| 论文 | 领域 | F1 |
|---|---|---|
| bge_paper | Embedding 模型 | 91 ~ 100% |
| MiniMax-M1 技术报告 | LLM | 83% |
| Attention Is All You Need | NLP 经典 | 100% |
| ResNet | CV 经典 | 94% |

用 ragas 0.2.15 原版（LLM-as-judge，不是自定义指标）评 RAGAS 标准指标：

| 指标 | 含义 | 总体 |
|---|---|---|
| **Context Recall** | 金标准事实被检索回来的比例 | **97.9%** |
| **Faithfulness** | 答案断言被证据支撑的比例（越高幻觉越低）| **95.2%** |

![检索效果](images/01_retrieval_accuracy.png)
![RAGAS 指标](images/02_ragas_metrics.png)
![4 篇论文 F1](images/03_paper_f1.png)

---

## 系统架构

![系统架构](images/04_architecture.png)

系统分五层：

- **用户层**：Gradio Web 界面 / FastAPI / 知识图谱可视化
- **MCP 协议层**：FastMCP 把能力封装成标准工具，可被 Claude 等 MCP 客户端直接调用
- **Agent 编排层**：路由 → 检索 → 推理 → 反思，Corrective RAG 范式
- **检索层**：向量 + 知识图谱 + 层次摘要三路召回，Cross-Encoder 两阶段精排
- **数据层**：ChromaDB / 内存图（可切 Neo4j）/ bge-m3 / bge-reranker

---

## 核心技术

### 1. 混合检索：向量 + BM25 + RRF + HyDE

纯向量检索有个老问题：对专有名词和数字不敏感。查"用了多少 H800 GPU"，向量可能召回一堆讲 GPU 的片段，漏掉那句"用了 512 张 H800"。

解法是逐步叠加策略：

```
纯向量（语义召回）→ +BM25（精确匹配）+RRF 融合 → +HyDE（假设答案检索）→ +Cross-Encoder 精排
```

- **RRF**（Reciprocal Rank Fusion）：向量分是 cosine（0~1），BM25 分是非归一化的，量纲不同没法直接加。RRF 只看排名不看绝对分：`score(d) = Σ 1/(k + rank)`，k 取 60。
- **HyDE**（Hypothetical Document Embeddings）：query 太短、chunk 太长，直接算相似度吃亏。先让 LLM 生成一个假设答案，用假设答案的向量去检索——它和 chunk 都是"答案形态"，语义更接近。

### 2. Cross-Encoder 两阶段精排

双塔（Bi-Encoder）：query 和 doc 分别编码算 cosine，快但精度有上限，两个向量没交互。

Cross-Encoder：query 和 doc 拼一起喂入 transformer 做交叉注意力，输出相关性分。精度高一个量级，但慢（每次都要过模型）。

所以走两阶段：

```
双塔粗召回 top-N  →  Cross-Encoder（bge-reranker-v2-m3）精排 top-5
   （快，可预计算）        （准，实时打分）
```

### 3. GraphRAG：让 RAG 会多跳推理

向量检索只能找"相似片段"，但"A 的作者还用过什么技术"这种问题答案分散在多个段落，靠相似度找不到。

解法是把论文里的实体关系抽成图，用图遍历做多跳推理：

```
论文文本 → LLM 抽三元组 → 实体消歧 → 建图 → text2cypher 查询
```

**实体消歧**是最难的环节（"MiniMax-M1"、"M1"、"the model" 可能是同一个实体），用了三层策略：子串包含 → 相似度阈值 → LLM 仲裁。

**text2cypher**（类比 text2SQL）：LLM 把自然语言翻译成图查询语句。Neo4j 连不上时，内置了一个内存图后端降级，本地也能跑。

### 4. 多 Agent 编排（Corrective RAG）

```
路由器(12) → 检索器(13) → 推理器(14) → 反思器(15)
```

- **路由器**：判定问题类型（事实 / 关系 / 总结）、该用哪些工具（向量 / 图谱 / 摘要）、跳数。规则优先，规则拿不准再请 LLM 仲裁。
- **检索器**：按路由策略多路召回 → RRF 融合 → MMR 多样性去重 → cross-encoder 精排。
- **推理器**：extraction-then-generation——先让 LLM 枚举片段中所有相关术语，再据此写答案，关键术语逐字保留。多次采样取忠实度最高的（自一致性）。
- **反思器**：把答案拆成原子断言，每条回检索片段找证据。有个"主语共指宽容"细节：答案里"它用了 lightning attention"和片段里"MiniMax-M1 用了 lightning attention"，"它"=MiniMax-M1，不该判幻觉。

### 5. MCP 协议工具层

把检索 / 图谱 / 多模态 / 联网搜索封装成标准 MCP（Model Context Protocol）工具。任何支持 MCP 的客户端（Claude Desktop、其他 Agent）都能直接调用，不用手写 API 集成。

---

## 项目结构

```
.
├── script/                      # 全部代码（按流水线顺序编号）
│   ├── 01_config.py             # 全局配置
│   ├── 02_pdf_parser.py         # PDF 解析（多栏 / 图表 / 公式 / 标题）
│   ├── 03_chunker.py            # 文本分块 + 章节归属
│   ├── 04_embedder.py           # bge-m3 向量化 + ChromaDB
│   ├── 05_llm_client.py         # MiniMax-M3 客户端封装
│   ├── 06_rag_query.py          # 混合检索（向量+BM25+RRF+HyDE+rerank）
│   ├── 07_mcp_server.py         # MCP 工具服务
│   ├── 08_kg_extractor.py       # LLM 三元组抽取
│   ├── 09_kg_builder.py         # 实体消歧 + 建图
│   ├── 10_kg_query.py           # text2cypher + 图查询
│   ├── 11_summary_indexer.py    # 层次摘要（document/section/paragraph）
│   ├── 12_router_agent.py       # 路由 Agent
│   ├── 13_retriever_agent.py    # 检索 Agent（多路+MMR+rerank）
│   ├── 14_reasoning_agent.py    # 推理 Agent（自一致性）
│   ├── 15_reflection_agent.py   # 反思 Agent（幻觉检测）
│   ├── 16_agent_orchestrator.py # Agent 编排器
│   ├── 17_eval_framework.py     # RAGAS 评估框架
│   ├── 17b_ragas_official.py    # ragas 原版（LLM-as-judge 对照）
│   ├── 18_eval_dataset.json     # 评测数据集
│   ├── 19_web_app.py            # Gradio Web 界面
│   ├── 20_api_server.py         # FastAPI
│   ├── 21_kg_visualizer.py      # 知识图谱可视化
│   ├── _reranker.py             # Cross-Encoder 重排模块
│   ├── _memory_graph.py         # 内存图后端（Neo4j 降级）
│   ├── _kg_gold.py              # 测试金标准数据
│   ├── _eval_helpers.py         # 评测工具函数
│   ├── run_embed_all.py         # 一键入库脚本
│   └── test_*.py                # 各模块准确率测试
├── data/                        # 4 篇测试论文 PDF
├── images/                      # 架构图与结果图
├── requirements.txt
├── docker-compose.yml           # Neo4j（可选）
├── LICENSE
└── README.md
```

---

## 核心模块速查

| 模块 | 关键函数 | 作用 |
|---|---|---|
| `02_pdf_parser` | `parse_pdf`, `detect_all_figures`, `is_real_table` | 三层图表检测（位图/矢量/标注）+ 假表格过滤 |
| `06_rag_query` | `retrieve`, `rrf_fuse`, `generate_hyde`, `rerank_with_llm` | 混合检索主流程 |
| `09_kg_builder` | `disambiguate_entities`, `normalize_triples` | 三层实体消歧 + 关系归一化 + 引文过滤 |
| `10_kg_query` | `nl_to_cypher`, `standardize_projection`, `graph_search` | 自然语言 → Cypher → 执行 → 答案投影 |
| `11_summary_indexer` | `extractive_summary`, `retrieve_hierarchical` | 关键句抽取式摘要 + 三层 RRF 检索 |
| `13_retriever_agent` | `retrieve`, `mmr_rerank` | 多路召回 + MMR 多样性重排 |
| `15_reflection_agent` | `verify_answer`, `_claim_supported` | token 级幻觉检测（命名实体/数字从严）|
| `16_agent_orchestrator` | `answer` | 端到端编排：路由→检索→推理→反思 |

---

## 快速开始

### 1. 环境

Python 3.12，建议建虚拟环境：

```bash
python3.12 -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

（requirements.txt 顶部注释里也有阿里云 / 华为云 / 豆瓣源，任选。）

### 3. 下载模型

两个模型要从 HuggingFace 下载（体积大，所以没放进仓库）：

```bash
# Embedding 模型 bge-m3
git clone https://huggingface.co/BAAI/bge-m3 models/Xorbits/bge-m3

# 重排序模型 bge-reranker-v2-m3
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3 models/bge-reranker-v2-m3
```

国内访问 HF 慢的话，用 ModelScope：

```bash
pip install modelscope
modelscope download --model BAAI/bge-m3 --local_dir models/Xorbits/bge-m3
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir models/bge-reranker-v2-m3
```

### 4. 配置 API Key

在项目根目录建一个 `.env` 文件（已在 `.gitignore` 里，不会被上传）：

```ini
MINIMAX_API_KEY=你的_MiniMax_API_Key
MINIMAX_API_HOST=https://api.minimaxi.com/v1
```

MiniMax API Key 在 https://platform.minimaxi.com 注册获取。代码里的 LLM 调用走 MiniMax-M3 的 OpenAI 兼容接口。

### 5. （可选）启动 Neo4j

知识图谱用内置内存图就能跑，不装 Neo4j 也行。想要 Neo4j 的话：

```bash
docker-compose up -d
```

默认连接 `bolt://localhost:7687`，账号密码在 `docker-compose.yml` 和 `.env` 里改。

### 6. 跑起来

```bash
# ① 入库：解析 data/ 下所有 PDF → 切块 → 向量化存进 ChromaDB
python script/run_embed_all.py

# ② 建知识图谱（以 bge_paper 为例）
python script/08_kg_extractor.py data/bge_paper.pdf
python script/09_kg_builder.py output/kg_triples/bge_paper.pdf.json

# ③ 直接 RAG 问答
python script/06_rag_query.py "what is bge-m3"

# ④ Agent 编排问答（推荐，走完整 路由→检索→推理→反思 链路）
python script/16_agent_orchestrator.py "what does MiniMax-M1 use"

# ⑤ 跑评估
python script/17_eval_framework.py

# ⑥ 起 Web 界面
python script/19_web_app.py
```

各脚本都支持 `--help` 看参数。每个 `test_XX_accuracy.py` 可单独验证对应模块的正确率。

---

## 技术栈

| 组件 | 选型 | 用在哪 |
|---|---|---|
| LLM | MiniMax-M3 | 推理 / 三元组抽取 / text2cypher |
| Embedding | bge-m3 | 文本向量化 |
| Reranker | bge-reranker-v2-m3 | Cross-Encoder 精排 |
| 向量库 | ChromaDB | 语义检索 |
| 知识图谱 | NetworkX（内存图）/ Neo4j | 多跳推理 |
| 工具协议 | FastMCP | 标准化工具调用 |
| 界面 | Gradio / FastAPI | Web / REST API |
| 评估 | ragas 0.2.15 | RAGAS 标准指标 |

---

## 贡献者

- **Apageoflove** — 独立设计与实现

## License

[MIT](LICENSE)
