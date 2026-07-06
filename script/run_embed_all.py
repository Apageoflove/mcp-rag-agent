"""
批量入库脚本：一键解析 data/ 下所有PDF -> 切分chunk -> 入库ChromaDB
- 中间chunk落盘到 output/chunks/，方便调试和被其他脚本消费
- 增量入库：已存在的 chunk_id 跳过（依赖 04 的去重机制）
- 单PDF失败不影响其他PDF继续跑

用法:
    env/bin/python3 script/run_embed_all.py
    env/bin/python3 script/run_embed_all.py --reset   # 清空库重建
    env/bin/python3 script/run_embed_all.py data/xxx.pdf  # 只跑指定文件
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib.machinery import SourceFileLoader
m02 = SourceFileLoader('m02', str(Path(__file__).resolve().parent / '02_pdf_parser.py')).load_module()
m03 = SourceFileLoader('m03', str(Path(__file__).resolve().parent / '03_chunker.py')).load_module()
m04 = SourceFileLoader('m04', str(Path(__file__).resolve().parent / '04_embedder.py')).load_module()

from config import DATA_DIR, CHROMA_DIR


CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / 'output' / 'chunks'
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def _save_chunks(pdf_name: str, chunks: list[dict]) -> Path:
    out = CHUNKS_DIR / f"{pdf_name}.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return out


def _load_chunks(pdf_name: str) -> list[dict] | None:
    p = CHUNKS_DIR / f"{pdf_name}.json"
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def _process_one(pdf_path: Path, force_rechunk: bool = False) -> dict:
    pdf_name = pdf_path.name
    t0 = time.time()

    if not force_rechunk:
        cached = _load_chunks(pdf_name)
        if cached is not None:
            chunks = cached
            t_parse = 0.0
            t_chunk = 0.0
        else:
            t1 = time.time()
            parsed = m02.parse_pdf(str(pdf_path))
            t_parse = time.time() - t1
            t2 = time.time()
            chunks = m03.chunk_document(parsed, source_name=pdf_name)
            t_chunk = time.time() - t2
            _save_chunks(pdf_name, chunks)
    else:
        t1 = time.time()
        parsed = m02.parse_pdf(str(pdf_path))
        t_parse = time.time() - t1
        t2 = time.time()
        chunks = m03.chunk_document(parsed, source_name=pdf_name)
        t_chunk = time.time() - t2
        _save_chunks(pdf_name, chunks)

    for i, c in enumerate(chunks):
        c['chunk_index'] = i
        c.setdefault('source', pdf_name)

    t3 = time.time()
    added = m04.add_chunks(chunks)
    t_embed = time.time() - t3

    return {
        'pdf': pdf_name,
        'chunks': len(chunks),
        'added': added,
        'skipped': len(chunks) - added,
        'time_parse': round(t_parse, 1),
        'time_chunk': round(t_chunk, 1),
        'time_embed': round(t_embed, 1),
        'time_total': round(time.time() - t0, 1),
    }


def _reset_db():
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if CHUNKS_DIR.exists():
        import shutil
        shutil.rmtree(CHUNKS_DIR)
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[RESET] 已清空 {CHROMA_DIR} 和 {CHUNKS_DIR}")


def main():
    ap = argparse.ArgumentParser(description='批量解析+切分+入库')
    ap.add_argument('pdfs', nargs='*', help='指定PDF文件（不传则处理 data/ 下所有）')
    ap.add_argument('--reset', action='store_true', help='清空库和chunk缓存后重建')
    ap.add_argument('--rechunk', action='store_true', help='忽略chunk缓存，重新跑02+03')
    args = ap.parse_args()

    if args.reset:
        _reset_db()

    if args.pdfs:
        pdf_paths = [Path(p) for p in args.pdfs]
    else:
        pdf_paths = sorted(DATA_DIR.glob('*.pdf'))

    if not pdf_paths:
        print(f"在 {DATA_DIR} 下没找到 PDF 文件")
        return

    print(f"待处理: {len(pdf_paths)} 个PDF")
    print(f"  - chunk缓存目录: {CHUNKS_DIR}")
    print(f"  - ChromaDB目录:  {CHROMA_DIR}")
    print(f"  - 模型:          bge-m3 (本地, CPU)")
    print()

    results = []
    for pdf in pdf_paths:
        if not pdf.exists():
            print(f"[SKIP] {pdf} 不存在")
            continue
        print(f"--- {pdf.name} ---")
        try:
            r = _process_one(pdf, force_rechunk=args.rechunk)
            results.append(r)
            print(f"  chunk: {r['chunks']} (新增 {r['added']} / 跳过 {r['skipped']})")
            print(f"  耗时: 解析 {r['time_parse']}s + 切分 {r['time_chunk']}s + 入库 {r['time_embed']}s = 总 {r['time_total']}s")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {str(e)[:200]}")
            results.append({'pdf': pdf.name, 'error': str(e)[:200]})
        print()

    print("=" * 60)
    print(f"处理完成: {len([r for r in results if 'error' not in r])}/{len(results)} 成功")
    print(f"ChromaDB 当前文档数: {m04.count()}")
    print(f"chunk缓存目录: {CHUNKS_DIR}")


if __name__ == '__main__':
    main()
