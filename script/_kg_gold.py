"""知识图谱金标准数据（08/09/10 测试共享）。

基于 MiniMax-M1 论文 Abstract 人工核实。数据来源：
output/chunks/MiniMax_M1_tech_report.pdf.json 的 Abstract 章节。
"""

# 金标准三元组（17条，归一化后的标准形式）
GOLD_TRIPLES = {
    # 论文: "MiniMax-M1 ... is developed based on our previous MiniMax-Text-01 model"
    ("MiniMax-M1", "based_on", "MiniMax-Text-01"),
    # 论文: "We introduce MiniMax-M1" (主体 MiniMax 提出 M1)
    ("MiniMax", "proposes", "MiniMax-M1"),
    # 论文: "powered by a hybrid Mixture-of-Experts (MoE) architecture"
    ("MiniMax-M1", "uses", "Mixture-of-Experts"),
    # 论文: "combined with a lightning attention mechanism"
    ("MiniMax-M1", "uses", "lightning attention mechanism"),
    # 论文: "compared to DeepSeek R1, M1 consumes 25% of the FLOPs"
    ("MiniMax-M1", "compared_with", "DeepSeek-R1"),
    # 论文: "comparable or superior to ... DeepSeek-R1"
    ("MiniMax-M1", "outperforms", "DeepSeek-R1"),
    # 论文: "trained using large-scale reinforcement learning (RL)"
    ("MiniMax-M1", "uses", "reinforcement learning"),
    # 论文: "we propose CISPO"
    ("MiniMax-M1", "proposes", "CISPO"),
    # 论文: "CISPO, a novel RL algorithm"
    ("CISPO", "is_a", "novel RL algorithm"),
    # 论文: "CISPO clips importance sampling weights rather than token updates"
    ("CISPO", "clips", "importance sampling weights"),
    # 论文: "outperforming other competitive RL variants"
    ("CISPO", "outperforms", "RL variants"),
    # 论文: "Combining hybrid-attention and CISPO"
    ("MiniMax-M1", "uses", "CISPO"),
    # 论文: "Combining hybrid-attention and CISPO"
    ("MiniMax-M1", "uses", "hybrid-attention"),
    # 论文: "full RL training on 512 H800 GPUs"
    ("MiniMax-M1", "trained_on", "H800"),
    # 论文: "We release two versions ... with 40K and 80K thinking budgets"
    ("MiniMax-M1", "released_with", "80K thinking budget"),
    # 论文: "comparable or superior to ... Qwen3-235B"
    ("MiniMax-M1", "compared_with", "Qwen3-235B"),
    # 论文: "comparable or superior to ... Qwen3-235B"
    ("MiniMax-M1", "outperforms", "Qwen3-235B"),
}


# 金标准实体映射（08原始实体→09归一化标准名）
GOLD_ENTITY_MAPPING = {
    # 需要合并的（指向同一实体的不同写法）
    "M1": "MiniMax-M1",
    "DeepSeek R1": "DeepSeek-R1",
    "Lightning Attention": "lightning attention mechanism",
    "lightning attention": "lightning attention mechanism",
    # 保持独立的（已经是标准名）
    "MiniMax-M1": "MiniMax-M1",
    "MiniMax": "MiniMax",
    "MiniMax-Text-01": "MiniMax-Text-01",
    "Mixture-of-Experts": "Mixture-of-Experts",
    "lightning attention mechanism": "lightning attention mechanism",
    "reinforcement learning": "reinforcement learning",
    "CISPO": "CISPO",
    "novel RL algorithm": "novel RL algorithm",
    "importance sampling weights": "importance sampling weights",
    "RL variants": "RL variants",
    "hybrid-attention": "hybrid-attention",
    "H800": "H800",
    "80K thinking budget": "80K thinking budget",
    "DeepSeek-R1": "DeepSeek-R1",
    "Qwen3-235B": "Qwen3-235B",
}


def compute_f1(predicted: set, gold: set) -> tuple[float, float, float]:
    """计算 precision/recall/F1。返回 (precision, recall, f1)。"""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


RAW_TRIPLES_PATH = "output/kg_triples/MiniMax_M1_tech_report_test.json"


# ───────────────────────────────────────────────────────────────
# bge_paper.pdf 的金标准三元组（人工从 Abstract/Intro 核实）
# 用于 09/10 在 bge 上的两-PDF 验证。归一化后的标准形式。
# ───────────────────────────────────────────────────────────────
BGE_TRIPLES = {
    # 三种检索功能
    ("M3-Embedding", "uses", "dense retrieval"),
    ("M3-Embedding", "uses", "multi-vector retrieval"),
    ("M3-Embedding", "uses", "sparse retrieval"),
    # 多语言
    ("M3-Embedding", "supports", "Multi-Linguality"),
    ("M3-Embedding", "supports", "Multi-Functionality"),
    ("M3-Embedding", "supports", "Multi-Granularity"),
    # 自知识蒸馏（核心训练方法）
    ("M3-Embedding", "uses", "Self-Knowledge Distillation"),
    # 100+ 语言
    ("M3-Embedding", "supports", "100+ languages"),
    # 作者机构
    ("M3-Embedding", "proposes", "BAAI"),
}

# bge 关系归一化期望（原始 relation → 标准化）
BGE_RELATION_NORMALIZE = {
    "use": "uses",
    "uses": "uses",
    "support": "supports",
    "supports": "supports",
    "accomplishes": "uses",      # "accomplishes dense retrieval" → uses
    "propose": "proposes",
    "proposes": "proposes",
    "is a": "is_a",
    "is_a": "is_a",
}
