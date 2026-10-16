"""摄入流水线：文档 → 切块 → 入库（新覆盖老） → 重建向量/BM25 索引。

策略说明：局部数据量小，摄入后对 FAISS + BM25 做全量轻量重建，简单且可靠；
大库场景可改为增量（本代码用序列重建，注释即扩展点）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from config import config

from ..retrieval.bm25_index import BM25Index
from ..retrieval.vector_index import VectorIndex
from ..wiki.models import Chunk
from ..wiki.store import WikiStore

SUPPORTED_EXTS = {".md", ".txt", ".markdown", ".pdf", ".docx", ".doc", ".html", ".htm"}


class IngestError(Exception):
    pass


def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".markdown", ".txt", ".html", ".htm"}:
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext in {".pdf", ".docx", ".doc"}:
        raise IngestError(
            f"{ext} 解析依赖 pypdf/python-docx，当前版本仅支持 markdown/txt。"
        )
    raise IngestError(f"不支持的文件类型：{ext}")


def chunk_text(text: str) -> List[str]:
    """切块：优先用 langchain 的 RecursiveCharacterTextSplitter；缺依赖时按简单按字符切。"""
    try:
        sum_len = sum(len(x) for x in text.splitlines())
        if sum_len < 2:
            return [t for t in text.splitlines() if t.strip()] or [text]
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception:
        # 兜底：简单按固定大小切片
        size = config.chunk_size
        chunks: List[str] = []
        for i in range(0, len(text), size):
            chunks.append(text[i : i + size])
        return [c for c in chunks if c.strip()]


def ingest_file(
    store: WikiStore,
    vector: VectorIndex,
    bm25: BM25Index,
    path: Path,
    space: str,
    doc_id: str,
    title: Optional[str] = None,
) -> dict:
    """摄入单文件。返回摄入统计。"""
    path = Path(path)
    if not path.exists():
        raise IngestError(f"文件不存在：{path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise IngestError(f"不支持的文件类型：{path.suffix}")

    title = title or path.stem
    text = read_text(path)
    if not text.strip():
        raise IngestError("文档内容为空")

    pieces = chunk_text(text)

    version = store.begin_ingest(space, doc_id, title)
    chunks = [
        Chunk(
            space=space,
            doc_id=doc_id,
            version=version,
            seq=i,
            text=p,
            meta={
                "title": title,
                "source": str(path.name),
                "seq": i,
                "total": len(pieces),
            },
        )
        for i, p in enumerate(pieces)
    ]

    try:
        store.save_chunks(chunks, version)
        vector.rebuild()
        bm25.rebuild()
    except Exception as e:
        store.rollback_ingest(space, doc_id, version)
        raise IngestError(f"摄入失败，已回滚到上一版本：{e}") from e

    return {
        "space": space,
        "doc_id": doc_id,
        "title": title,
        "version": version,
        "chunks": len(chunks),
        "file": str(path),
    }