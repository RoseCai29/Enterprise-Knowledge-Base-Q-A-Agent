"""Smoke 测试：版本「新覆盖老」、混合检索、is_latest 过滤。

注意：FAISS/检索相关用例不依赖外部 API（EMBED_MODE 默认 debug）。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402

config.embed_mode = "debug"
config.ensure_dirs()

from app.retrieval.bm25_index import BM25Index  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.vector_index import VectorIndex  # noqa: E402
from app.wiki.store import WikiStore  # noqa: E402


@pytest.fixture()
def trio(tmp_path):
    """隔离的 store+indexes，避免污染真实 data。"""
    config.db_path = tmp_path / "knowledge.db"
    config.faiss_path = tmp_path / "faiss.index"
    config.bm25_path = tmp_path / "bm25.pkl"
    store = WikiStore(str(tmp_path / "knowledge.db"))
    vector = VectorIndex(store)
    bm25 = BM25Index(store)
    retriever = HybridRetriever(store, vector, bm25)
    yield store, vector, bm25, retriever
    store.close()


def _ingest(store, vector, bm25, text, space, doc_id):
    from app.ingest.pipeline import chunk_text
    from app.wiki.models import Chunk

    version = store.begin_ingest(space, doc_id, doc_id)
    chunks = [
        Chunk(space=space, doc_id=doc_id, version=version, seq=i, text=p) for i, p in enumerate(chunk_text(text))
    ]
    store.save_chunks(chunks, version)
    vector.rebuild()
    bm25.rebuild()
    return version


TEXT_V1 = """本公司报销时限为 30 天，使用纸质报销单 EXCEL。"""
TEXT_V2 = """本公司报销时限为 60 天，使用线上系统 EXP-CLOUD。"""


def test_new_version_overrides_old(trio):
    store, vector, bm25, _ = trio
    v1 = _ingest(store, vector, bm25, TEXT_V1, "hr", "doc")
    v2 = _ingest(store, vector, bm25, TEXT_V2, "hr", "doc")
    assert v1 == 1 and v2 == 2
    # 旧版本 inactive
    from app.wiki.models import DocVersion
    vs = store.versions("doc")
    assert vs[0].active is False and vs[1].active is True
    # is_latest 过滤：检索命中 v2 而不是 v1
    hits = HybridRetriever(store, vector, bm25).search("报销时限", top_k=5)
    assert hits, "应检索到内容"
    assert all(h.version == 2 for h in hits), "应只命中最新版本 v2"


def test_bm25_hits_exact_proper_noun(trio):
    """BM25 精确命中专有名词/编号（纯向量未必）：演示混合检索的价值。"""
    store, vector, bm25, retriever = trio
    _ingest(store, vector, bm25, "救援专线 7201 已停用，现使用专线 7250。", "hr", "safety")
    # 仅 BM25
    hits = retriever.search("7201 救援专线", top_k=3, weights={"vector": 0, "bm25": 1})
    assert hits, "BM25 应命中专有名词"


def test_rrf_combines_both(trio):
    store, vector, bm25, retriever = trio
    _ingest(store, vector, bm25, "加班补贴 40 元每小时 上限二十小时", "hr", "overtime")
    hits = retriever.search("加班补贴标准", top_k=3)
    assert hits, "混合检索应返回结果"
    assert all("加班" in h.text for h in hits)


def test_no_llm_agent_debug_flow(trio):
    """无 LLM key 时 Agent 仍能返回检索引用（debug 兜底）。"""
    store, vector, bm25, retriever = trio
    _ingest(store, vector, bm25, "公司提供班车", "hr", "commute")
    import os

    os.environ.pop("LLM_API_KEY", None)
    # 重新构建 coordinator，确保 debug 分支
    from app.agent.graph import AgentCoordinator

    config.llm_api_key = ""
    coord = AgentCoordinator(store, retriever)
    result = coord.run("公司提供班车吗")
    assert result["answer"]
    assert isinstance(result["sources"], list)