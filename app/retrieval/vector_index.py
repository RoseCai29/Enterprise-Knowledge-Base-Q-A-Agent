"""FAISS 向量索引封装。

采用「全量轻量重建」策略：local 数据量小时，重建比增量维护更简单可靠。
chunk 对齐键 := f"{space}::{doc_id}::{version}::{seq}"（对应 Chunk.key）。
"""
from __future__ import annotations

import pickle
from typing import List, Optional

from config import config

from ..inference.llm import get_embedder
from ..wiki import models


class VectorIndex:
    def __init__(self, store=None) -> None:
        self._store = store
        self._index = None
        self._keys: List[str] = []  # key 与 FAISS 向量顺序一一对应
        self._vectors = None
        self._embedder = get_embedder()
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and len(self._keys) > 0

    def rebuild(self) -> int:
        """从 store 读取当前生效 chunks 重建 FAISS + 持久化。返回向量数。"""
        if self._store is None:
            return 0
        chunks = self._store.latest_active_chunks()
        if not chunks:
            self._index = None
            self._keys = []
            self._vectors = None
            self.reset()
            return 0

        import faiss
        import numpy as np

        texts = [c.text for c in chunks]
        embeds = self._embedder.embed_documents(texts)  # 全量 embed
        mat = np.asarray(embeds, dtype="float32")
        dim = mat.shape[1]
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        self._index = index
        self._keys = [c.key() for c in chunks]
        self._vectors = mat
        self.save()
        return len(self._keys)

    def search(self, query: str, top_k: int) -> List[dict]:
        """返回 [{key, score, index}]，key 用于回查 store。"""
        if self._index is None or not self._keys:
            return []
        import faiss
        import numpy as np

        q_vec = np.asarray([self._embedder.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q_vec)
        if q_vec.shape[1] != self._index.d:
            # 索引维度与当前 embedder 不一致（如切换了 EMBED_MODE/模型）：
            # 旧索引失效，改用当前 embedder 重建索引，保证可自愈。
            self.rebuild()
            if self._index is None or not self._keys:
                return []
        scores, idxs = self._index.search(q_vec, min(top_k, len(self._keys)))
        return [
            {"key": self._keys[int(i)], "score": float(scores[0][j])}
            for j, i in enumerate(idxs[0])
            if int(i) >= 0
        ]

    # ---------- 持久化 ----------
    # 说明：不用 faiss.write_index —— 它走 C++ 文件流，Windows 上无法处理含
    # 中文的路径（如“项目”）。改用纯 Python 的 np.save（CJK 安全），加载时
    # 在线重建 IndexFlatIP，等价且可移植。
    CONFIG_SUFFIX = ".npy"
    KEYS_SUFFIX = ".keys.pkl"

    def save(self) -> None:
        import numpy as np

        config.ensure_dirs()
        npy_path = str(config.faiss_path) + self.CONFIG_SUFFIX
        if self._vectors is not None:
            np.save(npy_path, self._vectors)
        else:
            import os

            if os.path.exists(npy_path):
                os.remove(npy_path)
        with open(str(config.faiss_path) + self.KEYS_SUFFIX, "wb") as f:
            pickle.dump(self._keys, f)

    def load(self) -> None:
        import numpy as np

        npy_path = str(config.faiss_path) + self.CONFIG_SUFFIX
        keys_path = str(config.faiss_path) + self.KEYS_SUFFIX
        import os

        if not (os.path.exists(npy_path) and os.path.exists(keys_path)):
            self._index = None
            self._keys = []
            self._vectors = None
            return
        try:
            import faiss

            mat = np.load(npy_path).astype("float32")
            with open(keys_path, "rb") as f:
                self._keys = pickle.load(f)
            dim = mat.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(mat)
            self._vectors = mat
        except Exception:
            self._index = None
            self._keys = []
            self._vectors = None

    def reset(self) -> None:
        import os

        for p in (
            str(config.faiss_path) + self.CONFIG_SUFFIX,
            str(config.faiss_path) + self.KEYS_SUFFIX,
        ):
            if os.path.exists(p):
                os.remove(p)