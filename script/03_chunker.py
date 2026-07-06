"""
03 chunker: 把 02 解析出的长文本切成 ~500 字的小块，每块贴上章节元数据。
表格单独成块（竖排展示），代码块过滤掉（02 偶尔会把代码当表格，踩过坑）。
overlap 一开始设的 0，后来发现跨块的问题经常检索不到，才加上。
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# 句子边界切分：用 findall 不用 split，split 遇到连续标点会错位
_SENT_RE = re.compile(r'[^。！？；\n]*[。！？；\n]+|[^。！？；\n]+')


def smart_chunk(text, max_size=500, overlap=50):
    if not text or not text.strip():
        return []

    sents = [s for s in _SENT_RE.findall(text) if s.strip()]
    if not sents:
        return [text.strip()]

    out = []
    cur = ''
    for s in sents:
        if len(cur) + len(s) > max_size and cur:
            out.append(cur)
            # 上一块尾部 overlap 个字拼到下一块开头，保证上下文连续
            cur = (cur[-overlap:] if len(cur) > overlap else cur) + s
        else:
            cur += s
    if cur.strip():
        out.append(cur)
    return out


# 02偶尔会把代码块识别成表格，这里过滤
# 列表里的关键字是踩一个坑加一个：gradient_checkpointing / torch 是 ResNet 那篇带进来的
_CODE_HINTS = ['def ', 'import ', 'return ', 'for ', '# ', '()',
               'self.', 'lambda', 'enumerate', 'gradient_checkpointing', 'torch']

def is_code_table(table_data):
    """表格内容里如果命中>=3个代码关键字，就当它是代码"""
    txt = ' '.join(str(c) for row in table_data for c in row if c).lower()
    return sum(1 for k in _CODE_HINTS if k in txt) >= 3


def table_to_chunk(table_data, page):
    """表格转竖排 chunk：'列名: 值 | 列名: 值' 一行展示，避免横向溢出"""
    if not table_data or len(table_data) < 2:
        return None
    header = [str(c) if c else '' for c in table_data[0]]
    rows = [[str(c) if c else '' for c in row]
            for row in table_data[1:]
            if any(c and str(c).strip() for c in row)]
    if not rows:
        return None

    lines = []
    for row in rows:
        parts = [f"{header[i]}: {c}" for i, c in enumerate(row)
                 if c.strip() and i < len(header)]
        if parts:
            lines.append(' | '.join(parts))
    return {"text": "表格内容：\n" + "\n".join(lines), "page": page, "type": "table"}


# 章节归属：根据02给出的headers，给每个chunk找到它属于哪一节
def _fallback_section(page_text):
    """02没识别到标题时的兜底：从首页文本里猜"""
    if re.search(r'^abstract\b', page_text, re.I | re.M):
        return 'Abstract'
    if re.search(r'^1\.?\s*introduction', page_text, re.I | re.M):
        return '1. Introduction'
    m = re.search(r'^(\d+)\.\s*(.+)$', page_text, re.M)
    if m:
        return m.group(2).strip()[:60]
    first = page_text.split('\n')[0].strip()
    # 首页第一行如果太长，是论文标题，用"摘要"代替
    return 'Abstract / 论文摘要' if len(first) > 40 else first[:80]


def attach_section(chunks, headers):
    if not headers:
        for c in chunks:
            c['section'] = ''
        return chunks

    headers_sorted = sorted(headers, key=lambda h: (h.get('page', 0), h.get('line', 0)))
    cache = {}  # 同一页的所有chunk共享同一个section，按页缓存避免重复算

    for c in chunks:
        p = c.get('page', 1)
        if p in cache:
            c['section'] = cache[p]
            continue

        h1, h2 = '', ''
        for h in headers_sorted:
            if h.get('page', 0) > p:  # 只看到当前页为止
                break
            if h.get('level') == 'h1':
                h1 = fix_english_spacing(h.get('title', '')[:80])
                h2 = ''
            elif h.get('level') == 'h2':
                h2 = fix_english_spacing(h.get('title', '')[:80])

        if h1 and h2:
            sec = f"{h1} > {h2}"
        elif h1:
            sec = h1
        elif h2:
            sec = h2
        elif p <= 2:
            sec = 'Abstract / 摘要'
        else:
            sec = '未分类'
        cache[p] = sec
        c['section'] = sec
    return chunks


# 英文空格修复：主要给章节标题用（"DataCuration" -> "Data Curation"）
# 注：正文不修，文本质量是02的活，03不要碰
_COMMON_WORDS = ['the', 'and', 'for', 'are', 'was', 'were', 'with',
                 'from', 'that', 'this', 'have', 'not', 'but', 'which',
                 'can', 'will', 'would', 'their', 'they', 'been', 'also',
                 'where', 'when', 'what', 'each', 'both', 'into', 'such']

def split_lowercase_join(token):
    """全小写连拼串(如"thedatacuration")尝试用常见词拆开"""
    if len(token) < 15 or not token.islower():
        return token
    out = token
    for w in sorted(_COMMON_WORDS, key=len, reverse=True):
        out = re.sub(r'(?<=[a-z])' + w + r'(?=[a-z])', ' ' + w + ' ', out)
    return re.sub(r'\s+', ' ', out).strip()


def fix_english_spacing(text):
    def fix_token(t):
        if len(t) < 10 or not re.search(r'[a-zA-Z]', t):
            return t
        # 含 - 或 _ ：分段处理（保护 hyphen 命名）
        if '-' in t or '_' in t:
            sep = '-' if '-' in t else '_'
            return sep.join(
                re.sub(r'([a-zA-Z])(\d)', r'\1 \2',
                       re.sub(r'(\d)([a-zA-Z])', r'\1 \2',
                              re.sub(r'([a-z])([A-Z])', r'\1 \2', p)))
                if len(p) > 3 else p
                for p in t.split(sep)
            )
        # 普通token：大小写切换位置补空格
        t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        t = re.sub(r'(\d)([A-Z][a-z])', r'\1 \2', t)
        t = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', t)
        return split_lowercase_join(t)

    return ''.join(
        p if p.isspace() else fix_token(p)
        for p in re.split(r'(\s+)', text)
    )


def chunk_document(parsed_pdf, source_name='', max_size=500, overlap=50):
    pages = parsed_pdf.get('text', [])
    tables = parsed_pdf.get('tables', [])
    headers = parsed_pdf.get('headers', [])

    chunks = []
    idx = 0

    # 1) 表格先处理（每表一块，过滤代码表）
    for t in tables:
        if not isinstance(t, dict):
            continue
        data = t.get('data', [])
        if not data or is_code_table(data):
            continue
        tc = table_to_chunk(data, t.get('page', 1))
        if tc:
            tc.update(source=source_name, chunk_index=idx, content_type='table')
            chunks.append(tc)
            idx += 1

    # 2) 正文按页切
    for pidx, ptext in enumerate(pages):
        if not ptext or not ptext.strip():
            continue
        for tc in smart_chunk(ptext, max_size, overlap):
            if len(tc.strip()) < 20:
                continue
            chunks.append({
                "text": tc.strip(),
                "source": source_name,
                "page": pidx + 1,
                "chunk_index": idx,
                "content_type": "text",
            })
            idx += 1

    # 3) 关联章节
    return attach_section(chunks, headers)


def save_chunks_json(chunks, pdf_name, output_dir=None):
    """将 chunk 列表保存为 JSON，供 06_rag_query 构建 BM25 索引"""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / 'output' / 'chunks'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{pdf_name}.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 03_chunker.py <pdf_path> [--save]")
        sys.exit(1)

    from importlib import import_module
    sys.path.insert(0, 'script')
    parsed = import_module('02_pdf_parser').parse_pdf(sys.argv[1])
    pdf_name = sys.argv[1].split('/')[-1]
    chunks = chunk_document(parsed, source_name=pdf_name)

    # 简单诊断
    types, sec_ok = {}, 0
    for c in chunks:
        types[c.get('content_type', '?')] = types.get(c.get('content_type', '?'), 0) + 1
        sec_ok += bool(c.get('section'))

    print(f"分块完成：{len(chunks)} 个chunk")
    print(f"  类型: {types}")
    print(f"  有章节: {sec_ok}/{len(chunks)}")
    print()
    for i, c in enumerate(chunks[:6]):
        prev = c.get('text', '')[:120].replace('\n', ' ')
        print(f"[{i}] 页{c.get('page', '?')} | {c.get('content_type', '')} | 章节: {c.get('section', '')[:40]}")
        print(f"    内容前120字: {prev}（后面还有内容）")
    if len(chunks) > 6:
        print(f"\n（只显示前6个，还有{len(chunks)-6}个chunk未展示）")

    if '--save' in sys.argv:
        out_path = save_chunks_json(chunks, pdf_name)
        print(f"\nchunk 已保存到: {out_path}")
