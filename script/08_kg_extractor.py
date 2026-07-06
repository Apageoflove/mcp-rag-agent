"""知识图谱三元组抽取模块。调LLM从文本抽取(S, R, O)三元组，结构性规则校验。"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from openai import OpenAI

# RELATION_TYPES = {
#     '任职': ['任职', '担任', '出任', '就职', 'CEO', 'CTO', '总裁'],
#     '创建': ['创建', '创立', '创办', '成立', '建立', '发起'],
#     '投资': ['投资', '融资', '注资', '领投', '参投'],
#     '合作': ['合作', '协作', '联合', '伙伴'],
#     '收购': ['收购', '并购', '兼并'],
#     '产品': ['产品', '发布', '推出', '研发', '开发'],
#     '总部': ['总部', '位于', '设在'],
#     '隶属': ['隶属', '属于', '旗下', '子公司'],
#     '人员': ['创始人', '员工', '团队', '成员'],
#     '技术': ['基于', '采用', '使用', '技术', '架构'],
# }

# 关键词全小写，normalize_relation() 匹配前会把关系转小写再比对。
RELATION_TYPES = {
    'proposes':     ['propose', 'present', 'introduce', 'develop'],
    'based_on':     ['based on', 'built on', 'extend', 'derived from'],
    'uses':         ['use', 'utilize', 'employ', 'adopt', 'leverage'],
    'outperforms':  ['outperform', 'surpass', 'beat', 'exceed'],
    'trained_on':   ['train', 'pretrain', 'fine-tune', 'finetune'],
    'evaluated_on': ['evaluate', 'test', 'benchmark', 'assess'],
    'achieves':     ['achieve', 'reach', 'obtain', 'attain'],
    'consists_of':  ['consist of', 'compose', 'comprise', 'contain'],
    'compared_with':['compare', 'versus', 'against'],
    'part_of':      ['part of', 'component', 'module', 'submodule'],
}

MAX_RELATION_LEN = 15

EXTRACT_PROMPT = """你是一个知识图谱抽取专家。从下面这段文本中抽取实体关系三元组。

要求：
1. 每个三元组格式为 (主体, 关系, 客体)
2. 只抽取文本中明确提到的事实，不要推断或编造
# 3. 关系用简短的中文词表示（如"创建""投资""任职""产品"）
3. 关系用简短的英文动词表示（如 "proposes" "uses" "outperforms" "based on"）
4. 实体必须是专有名词（人名、公司名、产品名、技术名等）
5. 如果文本中没有明确的关系，返回空列表

文本：
{chunk_text}

请只输出JSON格式，不要其他内容：
[
  {{"subject": "主体", "relation": "关系", "object": "客体"}}
]
如果没有三元组，输出 []

"""
def _build_client():
    """初始化LLM客户端（MiniMax兼容OpenAI接口）"""
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

def call_llm_extract(client, chunk_text):
    """调LLM抽取三元组，返回原始列表（未验证）"""
    if len(chunk_text.strip()) < 20:
        return []

    prompt = EXTRACT_PROMPT.format(chunk_text=chunk_text[:2000])

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content.strip()

        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        if content.startswith('```'):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        json_match = re.search(r'\[.*\]', content, flags=re.DOTALL)
        if json_match:
            content = json_match.group(0)

        return json.loads(content)

    except json.JSONDecodeError as e:
        print(f"  [警告] LLM输出不是合法JSON: {e}")
        return []
    except Exception as e:
        print(f"  [警告] LLM调用失败: {e}")
        return []

def is_valid_entity(entity, chunk_text):
    """验证实体是否有效"""
    if not entity or not isinstance(entity, str):
        return False
    entity = entity.strip()

    if len(entity) < 2:
        return False
    if entity.isdigit():
        return False
    if not re.search(r'[\u4e00-\u9fffa-zA-Z0-9]', entity):
        return False
    # 实体必须出现在原文中，防止LLM编造
    if entity not in chunk_text:
        return False
    return True

def is_valid_relation(relation):
    """验证关系是否有效"""
    if not relation or not isinstance(relation, str):
        return False
    relation = relation.strip()
    if len(relation) > MAX_RELATION_LEN:
        return False
    if len(relation) < 1:
        return False
    return True

def normalize_relation(relation):
    """把关系标准化到预定义类型"""
    relation = relation.strip()
    # rel_lower = relation
    # for std_type, keywords in RELATION_TYPES.items():
    #     for kw in keywords:
    #         if kw in relation:
    #             return std_type
    rel_lower = relation.lower()
    for std_type, keywords in RELATION_TYPES.items():
        for kw in keywords:
            if kw in rel_lower:
                return std_type
    return relation

def validate_triple(triple, chunk_text):
    """用结构性规则验证单个三元组，无效返回None"""
    if not isinstance(triple, dict):
        return None
    subj = triple.get('subject', '')
    rel = triple.get('relation', '')
    obj = triple.get('object', '')

    if subj.strip() == obj.strip():
        return None
    if not is_valid_entity(subj, chunk_text):
        return None
    if not is_valid_entity(obj, chunk_text):
        return None
    if not is_valid_relation(rel):
        return None
    rel = normalize_relation(rel)

    return {"subject": subj.strip(), "relation": rel, "object": obj.strip()}

def extract_triples_two_pass(client, chunk_text):
    """两次抽取取并集+频次过滤。

    两次都抽到=高置信(2分)，只抽到一次=低置信(1分)仍保留，交给09消歧+10查询兜底。
    """
    pass1 = call_llm_extract(client, chunk_text)
    pass2 = call_llm_extract(client, chunk_text)

    valid1 = [v for v in (validate_triple(t, chunk_text) for t in pass1) if v]
    valid2 = [v for v in (validate_triple(t, chunk_text) for t in pass2) if v]

    # # 取交集：用标准化后的 (subject, relation, object) 作为key
    # # 这样"创立"和"创建"标准化后都是"创建"，能正确匹配
    # seen2 = {(t['subject'], t['relation'], t['object']) for t in valid2}
    # kept = [t for t in valid1 if (t['subject'], t['relation'], t['object']) in seen2]
    # return kept

    freq = {}
    for t in valid1 + valid2:
        key = (t['subject'], t['relation'], t['object'])
        if key not in freq:
            freq[key] = {**t, 'count': 0}
        freq[key]['count'] += 1
    kept = sorted(freq.values(), key=lambda x: (-x['count'], x['subject']))
    return kept

def extract_triples_from_chunks(chunks, client=None):
    """批量处理所有chunk，抽取三元组并全局去重"""
    if client is None:
        client = _build_client()

    all_triples = []
    seen = set()

    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        if not text:
            continue
        print(f" [{i+1}/{len(chunks)}]抽取中...(页{chunk.get('page', '?')})")

        triples = extract_triples_two_pass(client, text)

        for t in triples:
            key = (t['subject'], t['relation'], t['object'])
            if key not in seen:
                seen.add(key)
                t['source_page'] = chunk.get('page', 0)
                t['source_section'] = chunk.get('section', '')
                all_triples.append(t)

    # 旧代码：直接返回带变体的原始三元组（如24条含M1/MiniMax-M1等变体），下游需自行归一化
    # return all_triples
    # 新代码：返回前调09归一化（实体消歧+关系归一化+去重），输出干净三元组（如17条）
    from importlib.machinery import SourceFileLoader as _SFL
    _m09 = _SFL('m09', str(Path(__file__).resolve().parent / '09_kg_builder.py')).load_module()
    _freq = _m09.collect_entities(all_triples)
    _mapping = _m09.disambiguate_entities(_freq, client=client)
    all_triples = _m09.normalize_triples(all_triples, _mapping)
    return all_triples

def save_triples(triples, pdf_name, output_dir=None):
    """保存三元组到JSON文件"""
    if output_dir is None:
      output_dir =Path(__file__).resolve().parent.parent.parent / 'output' / 'kg_triples'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = output_dir / f"{pdf_name}.json"
    with open(out, 'w', encoding='utf-8') as f:
      json.dump(triples, f, ensure_ascii=False, indent=2)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 08_kg_extractor.py <pdf_path>")
        sys.exit(1)

    from importlib import import_module
    sys.path.insert(0, 'script')

    pdf_path = sys.argv[1]
    pdf_name = pdf_path.split('/')[-1]

    print("步骤1: 解析PDF...")
    parsed = import_module('02_pdf_parser').parse_pdf(pdf_path)

    print("步骤2: 分块...")
    chunks = import_module('03_chunker').chunk_document(parsed, source_name=pdf_name)
    print(f"  共 {len(chunks)} 个chunk")

    print("步骤3: 抽取三元组（每个chunk调2次LLM取交集）...")
    triples = extract_triples_from_chunks(chunks)

    print(f"\n抽取完成：{len(triples)} 个三元组")

    for i, t in enumerate(triples[:20]):
        print(f"({t['subject']}, {t['relation']}, {t['object']})  [页{t.get('source_page', '?')}]")
    if len(triples) > 20:
        print(f"  ...还有 {len(triples) - 20} 个")

    out_path = save_triples(triples, pdf_name)
    print(f"\n已保存到: {out_path}")
