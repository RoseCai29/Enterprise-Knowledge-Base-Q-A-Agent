"""BM25 关键词索引（rank_bm25）。

弥补纯向量对精确关键词/专有名词/版本号的命中不足。
中文先经 jieba 分词再送入 BM25，提升中文关键词召回。
"""
from __future__ import annotations

import pickle
from typing import List, Optional

from rank_bm25 import BM25Okapi

from config import config

# 关闭 jieba 冗余日志
try:
    import jieba

    jieba.setLogLevel(20)
except Exception:  # pragma: no cover
    jieba = None  # type: ignore


def tokenize(text: str) -> List[str]:
    if jieba is not None:
        return list(jieba.cut(text))
    return [t for t in text if t.strip()]


class BM25Index:
    def __init__(self, store=None) -> None:
        self._store = store
        self._bm25: Optional[BM25Okapi] = None
        self._keys: List[str] = []
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None and len(self._keys) > 0

    def rebuild(self) -> int:
        if self._store is None:
            return 0
        chunks = self._store.latest_active_chunks()
        if not chunks:
            self._bm25 = None
            self._keys = []
            self.save()
            return 0
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._keys = [c.key() for c in chunks]
        self.save()
        return len(self._keys)

    def search(self, query: str, top_k: int) -> List[dict]:
        if self._bm25 is None or not self._keys:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self._keys)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"key": self._keys[i], "score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]

    def save(self) -> None:
        config.ensure_dirs()
        blob = {
            "keys": self._keys,
            # BM25Okapi 支持 pickle 序列化（含文档词频）。
            "bm25": self._bm25,
        }
        with open(config.bm25_path, "wb") as f:
            pickle.dump(blob, f)

    def load(self) -> None:
        try:
            if config.bm25_path.exists():
                with open(config.bm25_path, "rb") as f:
                    blob = pickle.load(f)
                self._keys = blob.get("keys", [])
                self._bm25 = blob.get("bm25")
        except Exception:
            self._bm25 = None
            self._keys = []