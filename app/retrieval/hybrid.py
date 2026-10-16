"""混合检索：向量 + BM25 → RRF 融合 → 仅返回当前生效（未覆盖）的 chunk。

技术要点：
- RRF（Reciprocal Rank Fusion）：score(key) = Σ_{retriever} 1/(k + rank)。
  不依赖各 retriever 分数量纲，稳健融合语义召回与精确关键词召回。
- 融合后按 key 回查 SQLite，过滤掉 is_latest=0 / active=0 的旧版本块，
  保证「新覆盖老」后检索只命中最新内容。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from config import config

from .bm25_index import BM25Index
from .vector_index import VectorIndex


@dataclass
class HybridHit:
    text: str
    space: str
    doc_id: str
    title: str
    version: int
    seq: int
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    def citation(self) -> str:
        return f"[{self.space}/{self.doc_id}#v{self.version}-{self.seq}]"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation()
        return d


class HybridRetriever:
    def __init__(self, store, vector: Optional[VectorIndex] = None, bm25: Optional[BM25Index] = None) -> None:
        self._store = store
        self._vector = vector or VectorIndex(store)
        self._bm25 = bm25 or BM25Index(store)

    def search(self, query: str, top_k: Optional[int] = None, weights: Optional[dict] = None) -> List[HybridHit]:
        """weights: {"vector": 1.0, "bm25": 1.0}，可开关某一路便于对比。"""
        top_k = top_k or config.top_k
        weights = weights or {"vector": 1.0, "bm25": 1.0}

        v_top = int(config.retrieval_total_limit * (weights.get("vector", 1.0) or 1.0))
        b_top = int(config.retrieval_total_limit * (weights.get("bm25", 1.0) or 1.0))

        rrf: dict[str, float] = {}
        rank_src: dict[str, dict] = {}

        if self._vector.is_loaded:
            hits = self._vector.search(query, v_top)
            for rank, hit in enumerate(hits):
                key = hit["key"]
                rrf[key] = rrf.get(key, 0.0) + weights["vector"] / (config.rrf_k + rank)
                rank_src.setdefault(key, {})["vec"] = hit["score"]

        if self._bm25.is_loaded:
            hits = self._bm25.search(query, b_top)
            for rank, hit in enumerate(hits):
                key = hit["key"]
                rrf[key] = rrf.get(key, 0.0) + weights["bm25"] / (config.rrf_k + rank)
                rank_src.setdefault(key, {})["bm25"] = hit["score"]

        top_keys = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

        results: List[HybridHit] = []
        for key in top_keys:
            chunk = self._resolve(key)
            if chunk is None:
                continue  # 已被覆盖或不存在 → 过滤
            results.append(
                HybridHit(
                    text=chunk.text,
                    space=chunk.space,
                    doc_id=chunk.doc_id,
                    title=chunk.meta.get("title", chunk.doc_id),
                    version=chunk.version,
                    seq=chunk.seq,
                    score=rrf[key],
                    meta=chunk.meta,
                )
            )
        return results

    def _resolve(self, key: str):
        parts = key.split("::")
        if len(parts) != 4:
            return None
        space, doc_id, version, seq = parts
        try:
            version_i = int(version)
            seq_i = int(seq)
        except ValueError:
            return None
        # 关键过滤：确认块当前生效（未被新版本覆盖）
        if not self._store.is_chunk_active_latest(space, doc_id, version_i, seq_i):
            return None
        return self._store.get_chunk(space, doc_id, version_i, seq_i)