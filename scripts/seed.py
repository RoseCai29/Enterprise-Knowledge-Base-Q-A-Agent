"""一键摄入演示数据（含同一文档 v1→v2 覆盖演示）。

用法：python scripts/seed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config  # noqa: E402
from app.core.log import get_logger  # noqa: E402
from app.ingest.pipeline import ingest_file  # noqa: E402
from app.retrieval.bm25_index import BM25Index  # noqa: E402
from app.retrieval.vector_index import VectorIndex  # noqa: E402
from app.wiki.store import WikiStore  # noqa: E402

log = get_logger("seed")

SEED = [
    ("hr", "emp-handbook", "员工手册_v1.md"),
    ("hr", "emp-handbook", "员工手册_v2.md"),  # 覆盖 v1
    ("it", "it-ops", "IT运维手册.md"),
]


def main() -> None:
    config.ensure_dirs()
    store = WikiStore()
    vector = VectorIndex(store)
    bm25 = BM25Index(store)
    base = Path(__file__).resolve().parent / "seed_docs"
    for space, doc_id, fname in SEED:
        result = ingest_file(store, vector, bm25, base / fname, space=space, doc_id=doc_id)
        log.info("ok %s/%s v%s chunks=%s source=%s",
                 space, doc_id, result["version"], result["chunks"], fname)

    log.info("当前生效文档：")
    for d in store.list_documents():
        log.info("  %s/%s v%s  %s", d.space, d.doc_id, d.latest_version, d.title)
    log.info("emp-handbook 版本历史：")
    for v in store.versions("emp-handbook"):
        log.info("  v%s active=%s %s", v.version, v.active, v.created_at)


if __name__ == "__main__":
    main()