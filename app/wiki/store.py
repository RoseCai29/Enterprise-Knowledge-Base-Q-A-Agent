"""SQLite 存储 + 版本「新覆盖老」逻辑。

核心语义：
- 文档唯一身份 = (space, doc_id)。
- 重新摄入同一文档 → version+1；旧版本 chunks 与旧 version 置为 inactive，
  新版本 chunks 置为 active + is_latest。
- 检索只读 is_latest=1 且 active=1 的 chunk（即“新覆盖老”后被保留的当前内容）。
- 旧版本通过 doc_versions 表保留，可审计、可回滚。

面试讲点：软删 + is_latest 标志 + 版本号，而非物理删除旧数据。
"""
from __future__ import annotations

import sqlite3
from typing import List, Optional, Sequence

from config import config

from .models import Chunk, DocVersion, Document, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    space         TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    title         TEXT NOT NULL,
    latest_version INTEGER NOT NULL DEFAULT 1,
    is_latest     INTEGER NOT NULL DEFAULT 1,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (space, doc_id)
);

CREATE TABLE IF NOT EXISTS doc_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    space       TEXT NOT NULL,
    version     INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    source      TEXT DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1,
    UNIQUE (doc_id, version)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    space       TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    seq         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    meta        TEXT DEFAULT '{}',
    active      INTEGER NOT NULL DEFAULT 1,
    is_latest   INTEGER NOT NULL DEFAULT 1,
    UNIQUE (doc_id, version, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunks_latest ON chunks(space, doc_id, is_latest, active);
CREATE INDEX IF NOT EXISTS idx_versions_doc   ON doc_versions(doc_id);
"""


class WikiStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(config.db_path)
        config.ensure_dirs()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- 版本「新覆盖老」 ----------

    def begin_ingest(self, space: str, doc_id: str, title: str) -> int:
        """开始摄入一篇文档，返回新版本号。旧版本内容被同步标记为“已被覆盖”。"""
        existing = self.conn.execute(
            "SELECT * FROM documents WHERE space=? AND doc_id=?", (space, doc_id)
        ).fetchone()

        if existing is None:
            first_version = 1
            self.conn.execute(
                "INSERT INTO documents(space, doc_id, title, latest_version, is_latest, updated_at) "
                "VALUES (?,?,?,?,1,?)",
                (space, doc_id, title, first_version, now_iso()),
            )
        else:
            first_version = int(existing["latest_version"]) + 1
            # 旧版本 doc 置为 非当前
            self.conn.execute(
                "UPDATE documents SET title=?, latest_version=?, is_latest=0, updated_at=? "
                "WHERE space=? AND doc_id=?",
                (title, first_version, now_iso(), space, doc_id),
            )
            # 旧 version 记录置为 inactive
            self.conn.execute(
                "UPDATE doc_versions SET active=0 WHERE doc_id=?", (doc_id,)
            )
            # 旧 chunks 全部置为 非 active + 非 latest
            self.conn.execute(
                "UPDATE chunks SET active=0, is_latest=0 WHERE space=? AND doc_id=?",
                (space, doc_id),
            )

        self.conn.execute(
            "INSERT INTO doc_versions(doc_id, space, version, created_at, source, active) "
            "VALUES (?,?,?,?,?,1)",
            (doc_id, space, first_version, now_iso(), ""),
        )
        self.conn.commit()
        return first_version

    def _mark_version_active(self, doc_id: str, version: int) -> None:
        self.conn.execute(
            "UPDATE doc_versions SET active=1 WHERE doc_id=? AND version=?", (doc_id, version)
        )

    def rollback_ingest(self, space: str, doc_id: str, version: int) -> None:
        """摄入失败时回滚到上一版本仍可用。"""
        self.conn.execute(
            "DELETE FROM chunks WHERE space=? AND doc_id=? AND version=?",
            (space, doc_id, version),
        )
        self.conn.execute(
            "DELETE FROM doc_versions WHERE doc_id=? AND version=?", (doc_id, version)
        )
        prev = self.conn.execute(
            "SELECT version FROM doc_versions WHERE doc_id=? AND active=1 ORDER BY version DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
        if prev is None:
            # 无更早版本，则删除文档壳
            self.conn.execute("DELETE FROM documents WHERE space=? AND doc_id=?", (space, doc_id))
        else:
            self.conn.execute(
                "UPDATE documents SET latest_version=?, is_latest=1 WHERE space=? AND doc_id=?",
                (prev["version"], space, doc_id),
            )
            self.conn.execute(
                "UPDATE chunks SET active=1, is_latest=1 WHERE space=? AND doc_id=? AND version=?",
                (space, doc_id, prev["version"]),
            )
        self.conn.commit()

    def save_chunks(self, chunks: Sequence[Chunk], version: int) -> None:
        if not chunks:
            self.conn.commit()
            return
        for c in chunks:
            import json

            self.conn.execute(
                "INSERT INTO chunks(space, doc_id, version, seq, text, meta, active, is_latest) "
                "VALUES (?,?,?,?,?,?,1,1)",
                (
                    c.space,
                    c.doc_id,
                    c.version,
                    c.seq,
                    c.text,
                    json.dumps(c.meta, ensure_ascii=False),
                ),
            )
        # 本轮新版本已成为当前生效版本
        if chunks:
            first = chunks[0]
            self.conn.execute(
                "UPDATE documents SET is_latest=1, latest_version=? "
                "WHERE space=? AND doc_id=?",
                (first.version, first.space, first.doc_id),
            )
        self._mark_version_active(chunks[0].doc_id, version)
        self.conn.commit()

    # ---------- 查询 ----------

    def list_spaces(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT space FROM documents WHERE is_latest=1 ORDER BY space"
        ).fetchall()
        return [r["space"] for r in rows]

    def list_documents(self, space: Optional[str] = None) -> List[Document]:
        q = "SELECT * FROM documents WHERE is_latest=1"
        args: tuple = ()
        if space:
            q += " AND space=?"
            args = (space,)
        q += " ORDER BY space, doc_id"
        return [self._row_to_doc(r) for r in self.conn.execute(q, args).fetchall()]

    def get_document(self, space: str, doc_id: str) -> Optional[Document]:
        r = self.conn.execute(
            "SELECT * FROM documents WHERE space=? AND doc_id=? AND is_latest=1",
            (space, doc_id),
        ).fetchone()
        return self._row_to_doc(r) if r else None

    def versions(self, doc_id: str) -> List[DocVersion]:
        rows = self.conn.execute(
            "SELECT * FROM doc_versions WHERE doc_id=? ORDER BY version", (doc_id,)
        ).fetchall()
        return [
            DocVersion(
                doc_id=r["doc_id"],
                version=r["version"],
                created_at=r["created_at"],
                source=r["source"],
                active=bool(r["active"]),
            )
            for r in rows
        ]

    def latest_active_chunks(self, space: Optional[str] = None, doc_id: Optional[str] = None) -> List[Chunk]:
        """取当前生效（is_latest + active）的 chunks —— 检索的数据源。"""
        q = "SELECT * FROM chunks WHERE is_latest=1 AND active=1"
        args: list = []
        if space:
            q += " AND space=?"
            args.append(space)
        if doc_id:
            q += " AND doc_id=?"
            args.append(doc_id)
        return [self._row_to_chunk(r) for r in self.conn.execute(q, args).fetchall()]

    def get_chunk(self, space: str, doc_id: str, version: int, seq: int) -> Optional[Chunk]:
        r = self.conn.execute(
            "SELECT * FROM chunks WHERE space=? AND doc_id=? AND version=? AND seq=?",
            (space, doc_id, version, seq),
        ).fetchone()
        return self._row_to_chunk(r) if r else None

    def is_chunk_active_latest(self, space: str, doc_id: str, version: int, seq: int) -> bool:
        r = self.conn.execute(
            "SELECT active, is_latest FROM chunks "
            "WHERE space=? AND doc_id=? AND version=? AND seq=?",
            (space, doc_id, version, seq),
        ).fetchone()
        return bool(r and r["active"] and r["is_latest"])

    # ---------- helpers ----------

    @staticmethod
    def _row_to_doc(r: sqlite3.Row) -> Document:
        return Document(
            space=r["space"],
            doc_id=r["doc_id"],
            title=r["title"],
            latest_version=r["latest_version"],
            is_latest=bool(r["is_latest"]),
            updated_at=r["updated_at"],
        )

    @staticmethod
    def _row_to_chunk(r: sqlite3.Row) -> Chunk:
        import json

        try:
            meta = json.loads(r["meta"])
        except Exception:
            meta = {}
        return Chunk(
            space=r["space"],
            doc_id=r["doc_id"],
            version=r["version"],
            seq=r["seq"],
            text=r["text"],
            meta=meta,
            active=bool(r["active"]),
            is_latest=bool(r["is_latest"]),
        )