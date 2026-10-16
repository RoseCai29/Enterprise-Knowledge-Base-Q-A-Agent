"""Wiki 层数据模型（dataclass，无 ORM 依赖）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """文档身份：空间 + doc_id 唯一标识一篇知识文档。"""

    space: str
    doc_id: str
    title: str
    latest_version: int
    is_latest: bool = True
    updated_at: str = ""


@dataclass
class DocVersion:
    """文档的一个版本（历史留档，供审计/回滚）。"""

    doc_id: str
    version: int
    created_at: str
    source: str = ""
    active: bool = True


@dataclass
class Chunk:
    """文档切块：作为检索的最小单位。"""

    space: str
    doc_id: str
    version: int
    seq: int
    text: str
    meta: dict = field(default_factory=dict)

    # 检索命中过滤依据
    active: bool = True
    is_latest: bool = True

    def key(self) -> str:
        """块唯一键，用于 FAISS/BM25 与 DB 对齐。"""
        return f"{self.space}::{self.doc_id}::{self.version}::{self.seq}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")