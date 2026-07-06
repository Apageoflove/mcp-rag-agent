"""PDF 解析：把论文拆成干净的结构化文本（文本 / 表格 / 图表 / 公式 / 标题）。

最早这文件只做了纯文本提取，后来在 MiniMax 论文上一跑发现图表全是矢量图、
公式满天飞、还有扫描件，于是一路往上加：多栏检测 → 图表区域剔除 → 公式过滤 → OCR 降级。
每个函数里写了为什么这么干，就不在这儿罗列了。
"""

import os
import re
# from config import 
# from PIL import Image
import pdfplumber
from collections import Counter
try:
    import fitz  # PyMuPDF，文本提取质量比pdfplumber好（特别是分栏处理）
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# def parse_pdf(pdf_path: str) -> dict:
#     """解析PDF文件，返回结构化结果"""
#     pass

# 检测页面中的所有图表 （位图+矢量图+文本标注）!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!新增的
def detect_all_figures(page,page_idx,fitz_page=None):
    figures = []

    # 1.位图检测（原有的）
    for img in page.images:
        if img["width"] > 20 and img["height"] > 20:
            figures.append({
                "page": page_idx + 1,
                "type": "位图",
                "bbox": (round(img["x0"],1), round(img["top"],1),
                         round(img["x1"],1), round(img["bottom"],1))
            })
    
    # 2.矢量图检测：曲线+矩形总数超过15个，说明有图表
    # （这一层是后面才加的——一开始只查 page.images，结果论文里的图基本全漏了，
    #  学术论文 90% 的图都是矢量画的，不是嵌入的 png/jpg）
    n_curves = len(page.curves)
    n_rects = len(page.rects)
    if n_curves + n_rects > 15:
        # 用 find_tables 的的假表格区域作为图表位置
        found = page.find_tables()
        chart_found = False
        for t in found:
            rows = t.extract()
            total = sum(len(r) for r in rows) if rows else 0
            empty = sum(1 for r in rows for c in r
                        if not c or not str(c).strip()) if rows else 0 
            if total > 0 and (empty / total) > 0.4:
                bbox = t.bbox
                figures.append({
                    "page": page_idx + 1,
                    "type": "矢量图",
                    "bbox": (round(bbox[0],1), round(bbox[1],1),
                             round(bbox[2],1), round(bbox[3],1))
                })
                chart_found = True
        # 如果find_tables没找到区域，但确实有很多绘图元素
        if not chart_found:
            figures.append({
                "page": page_idx + 1,
                "type": "矢量图",
                "bbox": None,
                "info": f"{n_curves}条曲线，{n_rects}个矩形"
            })
        
    # 3.文本标注检测（用PyMuPDF文本，空格更好）
    # text = page.extract_text() or ""  # 旧：pdfplumber提取，BGE无空格
    text = fitz_page.get_text('text') if fitz_page else (page.extract_text() or "")
    fig_captions = []
    for m in re.finditer(r'Figure\s*\d+\s*[|:][^\n]{0,200}', text):
        cap_text = m.group()
        if len(cap_text) > 0 and cap_text.count(' ') / len(cap_text) < 0.05:
            cap_text = _fix_title_spacing(cap_text)
        search_text = cap_text[:20]
        pos = page.search(search_text)
        bbox = None
        if pos:
            bbox = (round(pos[0]['x0'],1), round(pos[0]['top'],1),
                    round(pos[0]['x1'],1), round(pos[0]['bottom'],1))
        fig_captions.append({"name": cap_text, "bbox": bbox})

    table_captions = []
    for m in re.finditer(r'Table\s*\d+\s*[|:][^\n]{0,200}', text):
        cap_text = m.group()
        if len(cap_text) > 0 and cap_text.count(' ') / len(cap_text) < 0.05:
            cap_text = _fix_title_spacing(cap_text)
        search_text = cap_text[:20]
        pos = page.search(search_text)
        bbox = None
        if pos:
            bbox = (round(pos[0]['x0'],1), round(pos[0]['top'],1),
                    round(pos[0]['x1'],1), round(pos[0]['bottom'],1))
        table_captions.append({"name": cap_text, "bbox": bbox})


    return figures, fig_captions, table_captions


# 假表格验证：pdfplumber 的 find_tables 会把图表的网格线也当成表格，得另外判一下
def is_real_table(table):
    if not table or len(table) < 2:
        return False
    total_cells = 0
    empty_cells = 0
    for row in table:
        for cell in row:
            total_cells += 1
            if not cell or not str(cell).strip():
                empty_cells += 1
    if total_cells == 0:
        return False
    empty_ratio = empty_cells / total_cells
    # 0.4 是反复试出来的：再高会把图表网格漏进来，再低又误杀稀疏的真表格
    return empty_ratio < 0.4

# 页眉页脚过滤（支持动态页码）！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
def _normalize_line(text: str) -> str:
    return re.sub(r'\d+', '{N}', text.strip())

def _line_matches_pattern(line: str, pattern: str) -> bool:
    if not pattern:
        return False
    regex = re.escape(pattern).replace(r'\{N\}', r'\d+')
    return bool(re.fullmatch(regex, line.strip()))

def remove_header_footer(pages_text: list[str]) -> list[str]:
    n = len(pages_text)
    if n < 3:
        return pages_text

    top_pattern_counter = Counter()
    bottom_pattern_counter = Counter()

    for page in pages_text:
        lines = [l for l in page.split('\n') if l.strip()]
        if len(lines) >= 1:
            top_pattern_counter[_normalize_line(lines[0])] += 1
        if len(lines) >= 2:
            bottom_pattern_counter[_normalize_line(lines[-1])] += 1

    threshold = n * 0.5
    noise_top = {p for p, c in top_pattern_counter.items() if c > threshold and p}
    noise_bottom = {p for p, c in bottom_pattern_counter.items() if c > threshold and p}

    cleaned = []
    for page in pages_text:
        lines = page.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]

        result_lines = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                result_lines.append(line)
                continue

            is_noise = False

            if non_empty_lines and stripped == non_empty_lines[0]:
                for pattern in noise_top:
                    if _line_matches_pattern(stripped, pattern):
                        is_noise = True
                        break

            if not is_noise and non_empty_lines and stripped == non_empty_lines[-1]:
                for pattern in noise_bottom:
                    if _line_matches_pattern(stripped, pattern):
                        is_noise = True
                        break

            if not is_noise:
                result_lines.append(line)

        cleaned.append('\n'.join(result_lines))

    return cleaned


# 多栏排版 -> 文字顺序错乱
def _chars_to_text(char_list: list) -> str:
    if not char_list:
        return ""
    sorted_chars = sorted(char_list, key=lambda c: (round(c['top'], 1), c['x0']))
    lines = []
    current_line = []
    current_y = None
    Y_TOLERANCE = 3
    SPACE_THRESHOLD = 1.8  # 字符间距>1.8pt时在两个字符之间插入空格

    for char in sorted_chars:
        char_y = round(char['top'], 1)
        if current_y is None:
            current_y = char_y
            current_line.append(char)
        elif abs(char_y - current_y) <= Y_TOLERANCE:
            current_line.append(char)
        else:
            # 行结束，按间距补空格后拼接
            line_text = ''
            for i, c in enumerate(current_line):
                if i > 0:
                    prev = current_line[i - 1]
                    gap = c['x0'] - prev.get('x1', prev.get('x0', 0))
                    if gap > SPACE_THRESHOLD:
                        line_text += ' '
                line_text += c['text']
            lines.append(line_text)
            current_line = [char]
            current_y = char_y

    if current_line:
        line_text = ''
        for i, c in enumerate(current_line):
            if i > 0:
                prev = current_line[i - 1]
                gap = c['x0'] - prev.get('x1', prev.get('x0', 0))
                if gap > SPACE_THRESHOLD:
                    line_text += ' '
            line_text += c['text']
        lines.append(line_text)

    return '\n'.join(lines)


def detect_columns(page, chars=None) -> int:
    """检测页面栏数：1/2/3"""
    if chars is None:
        chars = page.chars
    if not chars:
        return 1
    # 排除页眉/页脚区域的字符（y<80或y>page.height-80），避免全页宽字符干扰栏数检测
    body_chars = [c for c in chars if 80 < c['top'] < page.height - 80]
    if len(body_chars) < 30:
        return 1
    x_positions = [c['x0'] for c in body_chars]
    page_width = page.width

    # 双栏检测
    mid = page_width / 2
    left_count = sum(1 for x in x_positions if x < mid - 10)
    right_count = sum(1 for x in x_positions if x > mid + 10)
    gap_count = sum(1 for x in x_positions if mid - 10 <= x <= mid + 10)

    if gap_count < len(x_positions) * 0.05 and left_count > 100 and right_count > 100:
        # 三栏检测
        third = page_width / 3
        left_l = sum(1 for x in x_positions if x < third - 5)
        mid_l = sum(1 for x in x_positions if third + 5 < x < 2 * third - 5)
        right_l = sum(1 for x in x_positions if x > 2 * third + 5)
        if left_l > 100 and mid_l > 100 and right_l > 100:
            # 验证3栏：必须有两个间隙（1/3和2/3处字符都很少）
            gap1 = sum(1 for x in x_positions if abs(x - third) < 8)
            gap2 = sum(1 for x in x_positions if abs(x - 2 * third) < 8)
            total = len(x_positions)
            if gap1 < total * 0.02 and gap2 < total * 0.02:
                return 3
            return 2  # 只有一个间隙 → 2栏
        return 2
    return 1


def extract_page_text(page, exclude_bboxes=None, fitz_page=None) -> str:
    # 优先用PyMuPDF（分栏处理好、英文有空格）
    if fitz_page is not None:
        if exclude_bboxes:
            # 按文本块过滤图表区域
            blocks = fitz_page.get_text('blocks')
            result = []
            for b in blocks:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                in_chart = any(
                    bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]
                    for bbox in exclude_bboxes if bbox
                )
                if not in_chart and b[4].strip():
                    result.append(b[4].strip())
            return '\n'.join(result)
        return fitz_page.get_text('text')
    # 回退：用pdfplumber（fitz不可用时）
    if exclude_bboxes:
        chars = [c for c in page.chars
                 if not any(bbox[0] <= c['x0'] <= bbox[2] and bbox[1] <= c['top'] <= bbox[3]
                           for bbox in exclude_bboxes if bbox)]
        return _chars_to_text(chars)
    return page.extract_text() or ""

'''
# 旧版多栏提取逻辑（保留供参考，原生extract_text在MiniMax论文上效果更好）
def extract_page_text_old(page, exclude_bboxes=None) -> str:
    if exclude_bboxes:
        chars = [c for c in page.chars
                 if not any(bbox[0] <= c['x0'] <= bbox[2] and bbox[1] <= c['top'] <= bbox[3]
                           for bbox in exclude_bboxes if bbox)]
    else:
        chars = page.chars
    n_cols = detect_columns(page, chars)
    if n_cols == 1:
        return page.extract_text() or ""
    page_width = page.width
    if n_cols == 2:
        mid = page_width / 2
        left_chars = [c for c in chars if c['x0'] < mid - 8]
        right_chars = [c for c in chars if c['x0'] >= mid - 8]
        return _chars_to_text(left_chars) + '\n' + _chars_to_text(right_chars)
    else:
        third = page_width / 3
        col1 = [c for c in chars if c['x0'] < third]
        col2 = [c for c in chars if third <= c['x0'] < 2 * third]
        col3 = [c for c in chars if c['x0'] >= 2 * third]
        return _chars_to_text(col1) + '\n' + _chars_to_text(col2) + '\n' + _chars_to_text(col3)
'''

# 标题层级识别（优化）
TITLE_PATTERNS = [
    # 一级标题（句号可选，要求有空格："1 Introduction"或"1. Introduction"）
    (re.compile(r'^第[一二三四五六七八九十\d]+章\s'), 'h1'),
    (re.compile(r'^[一二三四五六七八九十]、'), 'h1'),
    (re.compile(r'^\d+\.?\s+[\u4e00-\u9fffa-zA-Z]'), 'h1'),
    # 二级标题（要求后面至少2个连续字母/汉字）
    (re.compile(r'^\d+\.\d+\.?\s*[\u4e00-\u9fffa-zA-Z]{2,}'), 'h2'),
]


def _fix_title_spacing(title: str) -> str:
    """修复标题中缺失的空格（如DataCuration→Data Curation）"""
    # 含连字符：分段处理
    if '-' in title:
        parts = title.split('-')
        fixed = []
        for p in parts:
            p = re.sub(r'([a-z])([A-Z])', r'\1 \2', p)
            fixed.append(p)
        return '-'.join(fixed)
    # 无连字符：直接补
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', title)


def extract_headers(pages_text: list[str]) -> list[dict]:
    """
    识别标题层级，返回标题列表。
    每个标题: {"level": "h1"|"h2", "title": "...", "page": 3, "line": 5}
    """
    headers = []
    for page_idx, page_text in enumerate(pages_text):
        lines = page_text.split('\n')
        for line_idx, line_text in enumerate(lines):
            stripped = line_text.strip()
            if not stripped:
                continue
            # PyMuPDF可能把编号和标题分两行：只合并1-9的编号
            if re.match(r'^[1-9]\.?\d{0,1}$', stripped) and line_idx + 1 < len(lines):
                next_line = lines[line_idx + 1].strip()
                if (next_line and next_line[0].isalpha() and next_line[0].isupper()
                        and not next_line.startswith(('Table', 'Figure', 'Accuracy', 'FLOPs', 'Loss'))):
                    stripped = f"{stripped} {next_line}"
            for pattern, level in TITLE_PATTERNS:
                if pattern.match(stripped):
                    # 过滤参考文献条目（编号>10的通常是引用文献）
                    num_match = re.match(r'^(\d+)\.', stripped)
                    if num_match and int(num_match.group(1)) > 10:
                        break
                    # 过滤非章节编号（0.5、1.0等坐标轴标签）
                    sec_num = re.match(r'^(\d+)\.(\d+)', stripped)
                    if sec_num:
                        major, minor = int(sec_num.group(1)), int(sec_num.group(2))
                        if major == 0 or minor == 0:
                            break
                    # 过滤URL（含http/github/huggingface等）
                    if re.search(r'https?://|github\.com|huggingface\.co|\.org|\.com', stripped):
                        break
                    # 过滤过长的行（标题通常<120字符）
                    if len(stripped) > 120:
                        break
                    # 过滤正文小数（如"1.2billion"）：标题英文首字母应大写
                    title_text = re.sub(r'^[\d.]+\s*', '', stripped)
                    if title_text and title_text[0].isalpha() and title_text[0].islower():
                        break
                    # 过滤含逗号的行（通常是参考文献或脚注）
                    if ',' in stripped:
                        break
                    # 过滤正文句子（以This/These/The/Our/In/It开头）
                    sentence_starts = ('This', 'These', 'The ', 'Our ', 'In ', 'It ', 'For ', 'As ')
                    if any(title_text.startswith(s) for s in sentence_starts):
                        break
                    # 过滤含4位以上数字的行（如"8192"、"2024"等参考文献）
                    if re.search(r'\d{4,}', stripped):
                        break
                    # 过滤含数字词的行（如"1.2 Billion"是正文不是标题）
                    NUMBER_WORDS = {'Billion', 'Million', 'Thousand', 'Trillion'}
                    if any(w in stripped.split() for w in NUMBER_WORDS):
                        break
                    # 过滤模型名误判（如"4 Opus"是Claude模型名不是标题）
                    MODEL_NAMES = {'Opus', 'GPT', 'Claude', 'Gemini', 'LLaMA', 'BERT', 'ChatGPT', 'Dolly'}
                    if any(w in stripped.split() for w in MODEL_NAMES):
                        break
                    # 修复标题空格（DataCuration→Data Curation）
                    fixed_title = _fix_title_spacing(stripped)
                    # 截断正文残留
                    words = fixed_title.split()
                    has_long_word = any(len(w) > 25 for w in words[1:])
                    if len(words) > 9 or has_long_word:
                        BODY_START = {'are', 'is', 'was', 'were', 'also', 'used',
                                      'similar', 'however', 'which', 'that', 'these',
                                      'this', 'their', 'they', 'same', 'even', 'more',
                                      'most', 'some', 'such', 'than', 'then', 'when',
                                      'where', 'how', 'shows', 'shown', 'based'}
                        for wi, w in enumerate(words[2:], 2):
                            if w.lower() in BODY_START or len(w) > 25:
                                fixed_title = ' '.join(words[:wi])
                                break
                    # 标题中间有句号（非章节号）→ 截断
                    period_idx = fixed_title.find('.', 5)
                    if 5 < period_idx < len(fixed_title) - 3:
                        fixed_title = fixed_title[:period_idx].strip()
                    headers.append({
                        "level": level,
                        "title": fixed_title,
                        "page": page_idx + 1,
                        "line": line_idx,
                    })
                    break
    return headers



# 公式处理
def detect_formulas(text):
    """检测并分离公式，返回（干净文本，公式列表）"""
    # 数学斜体变量
    math_vars = re.compile(r'[𝑖𝑡𝑗𝑘𝑛𝑚𝑠𝑟𝐴𝐽𝐺𝑀𝑅𝑜𝑞𝜃𝜋𝜖𝛽𝛾𝛼𝜆𝜇𝜎]')
    # 数学符号（求和、积分等）
    math_symbols = re.compile(r'[∑∫∂∇√𝔼ˆ]')
    formula_lines = []
    body_lines = []

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue

        is_formula = False

        # 特征1：含(cid:)标记（pdfplumber无法识别的数学符号）
        if '(cid:' in stripped:
            is_formula = True

        # 特征2：短行（≤15字符）且含数学变量或符号
        # （长行是正文中的行内变量引用，保留在正文里）
        if len(stripped) <= 15 and (math_vars.search(stripped) or math_symbols.search(stripped)):
            is_formula = True

        if is_formula:
            formula_lines.append(stripped)
        else:
            body_lines.append(line)

    cleaned_text = '\n'.join(body_lines)
    return cleaned_text, formula_lines


# 合并函数：合并：矢量图 + figure
def merge_regions(figures):
    """合并同一页面的碎片区域为一个整体"""
    by_page = {}
    for fig in figures:
        p = fig["page"]
        by_page.setdefault(p, []).append(fig)

    merged = []
    for page_num, figs in by_page.items():
        bboxes = [f["bbox"] for f in figs if f.get("bbox")]
        if len(bboxes) <= 1:
            merged.extend(figs)
            continue
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        merged.append({
            "page":page_num,
            "type":"图表",
            "bbox":(round(x0,1), round(y0,1), round(x1,1), round(y1,1))
        })
    return merged



# 图片PDF降级：扫描件提取不出文字时，转图片调M3多模态OCR
# 这块是后期补的——遇到一份扫描版 PDF，extract_text 直接返回空，整个链路就断了
def ocr_page_with_m3(pdf_path, page_idx):
    """把页面渲染成图片丢给 M3 多模态识别，慢但兜底用"""
    import base64
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
        from openai import OpenAI
    except ImportError:
        return ""

    # 用pdfplumber将页面转为图片
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        img = page.to_image(resolution=200)
        img.save('/tmp/opencode/page_ocr.png')

    # 读取图片并转base64
    with open('/tmp/opencode/page_ocr.png', 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode()

    # 调用M3多模态API
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请识别图片中的所有文字内容，保持原有的排版格式。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        }],
        max_tokens=4000,
    )
    return response.choices[0].message.content


# 加表格提取和图片位置
def parse_pdf(pdf_path):
    all_tables = []
    # all_images = []
    all_figures = []
    all_captions = []
    all_formulas = [] # 公式
    formula_counts = {} # 公式个数统计
    global_eq_seen = set() # 全局已见公式编号（去重，同一编号只计一次）
    with pdfplumber.open(pdf_path) as pdf:
        doc_fitz = fitz.open(pdf_path) if HAS_FITZ else None
        pages = []
        for page_idx, page in enumerate(pdf.pages):
            # 1. 先检测图表（需要bbox来剔除图表区域文字）
            fitz_page = doc_fitz[page_idx] if doc_fitz else None
            figures, fig_caps, tab_caps = detect_all_figures(page, page_idx, fitz_page=fitz_page)
            all_figures.extend(figures)
            for cap in fig_caps:
                all_captions.append({"page": page_idx + 1, "type": "Figure",
                                     "name": cap["name"], "bbox": cap.get("bbox")})
            for cap in tab_caps:
                all_captions.append({"page": page_idx + 1, "type": "Table",
                                     "name": cap["name"], "bbox": cap.get("bbox")})

            # 2. 收集本页图表区域bbox
            chart_bboxes = [f['bbox'] for f in figures if f.get('bbox')]

            # 3. 文本提取（用PyMuPDF，分栏处理好+有空格）
            raw_text = extract_page_text(page, exclude_bboxes=chart_bboxes, fitz_page=fitz_page)

            # 4. 图片PDF降级：如果提取为空，转图片调M3 OCR
            if not raw_text.strip():
                raw_text = ocr_page_with_m3(pdf_path, page_idx)

            # 5. 公式过滤
            cleaned_text, formulas = detect_formulas(raw_text)
            pages.append(cleaned_text)
            for f in formulas:
                all_formulas.append({"page": page_idx + 1, "content": f})

            # 统计公式个数（用原始文本，fitz或pdfplumber）
            orig_text = fitz_page.get_text('text') if fitz_page else (page.extract_text() or "")
            eq_nums = set()
            for line in orig_text.split('\n'):
                stripped = line.strip()
                search_area = stripped[-25:] if len(stripped) > 25 else stripped
                m = re.search(r'(?<![\w])\((\d+)\)', search_area)
                if m:
                    num = int(m.group(1))
                    if 1 <= num <= 20:
                        eq_nums.add(num)
            # 只统计本页新出现的编号
            new_eqs = eq_nums - global_eq_seen
            global_eq_seen.update(eq_nums)
            if new_eqs:
                formula_counts[page_idx + 1] = len(new_eqs)

            # 表格提取（带假表格过滤）
            tables = page.extract_tables()
            # for table in tables:
            #     all_tables.append({
            #         "page": page_idx + 1,
            #         "data": table
            #     })
            # 加入假表格验证
            for table in tables:
                if is_real_table(table):
                    all_tables.append({
                        "page": page_idx + 1,
                        "data": table
                    })


            # for img in page.images:
            #     all_images.append({
            #         "page": page_idx + 1,
            #         "x0": round(img["x0"], 1),
            #         "y0": round(img["top"], 1),
            #         "width": round(img["width"], 1),
            #         "height": round(img["height"], 1)
            #     })

    #         # 太窄或太小的不是真图片
    #         for img in page.images:
    #             if img["width"] > 20 and img["height"] > 20:
    #                 all_images.append({
    #                     "page": page_idx + 1,
    #                     "x0": round(img["x0"], 1),
    #                     "y0": round(img["top"], 1),
    #                     "width": round(img["width"], 1),
    #                     "height": round(img["height"], 1)
    #                 })    
    # # return {"text": pages, "tables": [], "images": []}
    # return {"text": pages, "tables": all_tables, "images": all_images}  # 解析完成：22 页文本，12 个表格

    # 合并碎片区域
    all_figures = merge_regions(all_figures)

    # 关闭fitz文档
    if doc_fitz:
        doc_fitz.close()

    # 页眉页脚过滤（在所有页面收集完后执行）
    pages = remove_header_footer(pages)

    # 标题层级识别
    all_headers = extract_headers(pages)

    # 把图表区域和标注匹配，合并到一起
    figure_with_cap = []
    for fig in all_figures:
        matched = ""
        cap_bbox = None
        for cap in all_captions:
            if cap["page"] == fig["page"]:
                matched = cap["name"][:120]
                cap_bbox = cap.get("bbox")
                break
        bbox = fig.get("bbox") or cap_bbox  # 图表区域优先，无区域则用标注位置回退
        figure_with_cap.append({
            "page": fig["page"],
            "type": fig["type"],
            "bbox": bbox,
            "caption": matched
        })
    # 没有区域的标注（无边框表格）也加进来
    fig_pages = [f["page"] for f in all_figures]
    for cap in all_captions:
        if cap["page"] not in fig_pages:
            figure_with_cap.append({
                "page": cap["page"],
                "type": "无边框表格" if cap["type"] == "Table" else "图表",
                "bbox": cap.get("bbox"),
                "caption": cap["name"][:120],
            })

    return {"text": pages, "tables": all_tables, "figures": figure_with_cap,
            "formulas": all_formulas, "formula_counts": formula_counts,
            "headers": all_headers}




'''
以上的过滤无法检测矢量图（用线条，曲线，矩形画的图表），page.images只能检测嵌入的位图（jpeg,png）。学术论文里90%的图表都是矢量图

1、位图 -> page.images(现有的，只有2张)
2、矢量图 -> page.curves + page.rects + page.lines  密集区域（图表都是用这些画的）
3、文本标注 -> 文本里的 “figure 1”, "table 2" 字样
'''




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = parse_pdf(sys.argv[1])
        # print(f"解析完成：{len(result.get('text', []))} 页文本，{len(result.get('tables', []))} 个表格")
        total_eq = sum(result.get('formula_counts', {}).values())
        print(f"解析完成： {len(result.get('text', []))}页文本， {len(result.get('tables', []))}个表格， {len(result.get('figures', []))}个图表, {total_eq}个公式")
        
        
        print()

        # for t in result.get('tables', []):
        #     print(f"表格：第{t['page']}页，{len(t['data'])}行")
        # for img in result.get('images', []):
        #     print(f"图片：第{img['page']}页，位置({img['x0']}, {img['y0']}), 大小{img['width']}x{img['height']}")

        # for fig in result.get('figures', []):
        #     bbox_str = fig['bbox'] if fig ['bbox'] else fig.get('info', '')
        #     print(f"[{fig['type']}] 第{fig['page']}页， 区域：{bbox_str}")
        # print()
        # for cap in result.get('captions', []):
        #     print(f"[{cap['type']}] 第{cap['page']}页：{cap['name']}")

        for fig in result.get('figures', []):
            bbox_str = fig['bbox'] if fig['bbox'] else "无位置"
            cap_str = fig['caption'] if fig['caption'] else "无标注"
            print(f"  [{fig['type']}]  第{fig['page']}页  {cap_str}")
            print(f"         区域：{bbox_str}")

        # 公式处理
        print()
        # for f in result.get('formulas', [])[:10]:
        #     print(f"[公式]第{f['page']}页: {f['content'][:120]}")
        # if len(result.get('formulas', [])) > 10:
        #     print(f"...还有{len(result.get('formulas', [])) - 10}行公式")
        
        # from collections import Counter
        # formula_pages = Counter(f['page'] for f in result.get('formulas', []))
        # for page_num in sorted(formula_pages.keys()):
        #     count = formula_pages[page_num]
        #     samples = [f['content'][:30] for f in result.get('formulas', []) if f['page'] == page_num][:2]
        #     sample_str = ' | '.join(samples)
        #     print(f"[公式]第{page_num}页：{count}行，示例：{sample_str}")

        total_eq = sum(result.get('formula_counts', {}).values())
        for page_num, count in sorted(result.get('formula_counts', {}).items()):
            print(f"  [公式]第{page_num}页:{count}个公式")
        if total_eq == 0:
            print("无公式")

        # 标题层级
        print()
        headers = result.get('headers', [])
        if headers:
            for h in headers:
                print(f"  [{h['level']}]  第{h['page']}页  {h['title'][:120]}")
        else:
            print("  未检测到标题")

    else:
        print("用法：python3 02_pdf_parser.py /data/MiniMax_M1_tech_report.pdf")
