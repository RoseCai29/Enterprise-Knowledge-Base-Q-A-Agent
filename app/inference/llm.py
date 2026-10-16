from __future__ import annotations

import importlib.util
from typing import List, Optional

from config import config


def _has_api(key: Optional[str], base_url: Optional[str]) -> bool:
    return bool(key and base_url)


def suggest_knowledge_missing() -> str:
    return (
        "缺失依赖（天内结束前请 `pip install langchain-openai langchain-core`）。"
        "当前使用 debug 兜底。"
    )


class Embedder:
    """可插拔 Embedding。

    - EMBED_MODE=api  ：调用 OpenAI 兼容 embedding API。
    - EMBED_MODE=local：本地 sentence-transformers（bge 系列）。无需外网 embedding key，
      生产推荐；首次使用自动下载模型到缓存目录。
    - EMBED_MODE=debug：确定性 hash 向量占位，仅 demo/测试用。
    """

    def __init__(self, mode: str | None = None, dim: int | None = None) -> None:
        self.mode = (mode or config.embed_mode).lower()
        self.dim = dim or config.embed_dim
        self._client = None
        self._sentence_model = None
        if self.mode == "api":
            if not _has_api(config.embed_api_key, config.embed_base_url):
                raise RuntimeError(
                    "EMBED_MODE=api 但未配置 EMBED_API_KEY / EMBED_BASE_URL"
                )
            self._build_client()
        elif self.mode == "local":
            self._build_local()

    @property
    def dim(self) -> int:
        return self._dim

    @dim.setter
    def dim(self, v: int) -> None:
        self._dim = v

    def _build_client(self) -> None:
        # 延迟导入，避免缺依赖时拖垮 import
        if importlib.util.find_spec("langchain_openai") is None:
            raise RuntimeError(suggest_knowledge_missing())
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(
            model=config.embed_model,
            base_url=config.embed_base_url,
            api_key=config.embed_api_key,
        )

    def _build_local(self) -> None:
        """懒加载本地 sentence-transformers 模型。首次使用会下载。"""
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "EMBED_MODE=local 需要 sentence-transformers，请 `pip install sentence-transformers`。"
            )
        cache_dir = str(config.embed_local_cache)
        config.embed_local_cache.mkdir(parents=True, exist_ok=True)
        model_name = config.embed_local_model
        try:
            # 走 HuggingFace 镜像（默认 hf-mirror.com），避免直连 huggingface.co 不通
            import os

            os.environ.setdefault("HF_ENDPOINT", config.embed_hf_endpoint)
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer(
                model_name_or_path=model_name,
                cache_folder=cache_dir,
                device=config.embed_local_device,
            )
            dim_fn = getattr(
                self._sentence_model, "get_embedding_dimension",
                self._sentence_model.get_sentence_embedding_dimension,
            )
            self._dim = dim_fn() or self._dim
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"加载本地 embedding 模型失败（{model_name}）：{e}\n"
                "请检查网络，或用 EMBED_HF_ENDPOINT 指定可用的 HuggingFace 镜像，"
                "或在 config 指定 EMBED_LOCAL_CACHE 已有模型的目录。"
            ) from e

    @staticmethod
    def _query_instruction() -> str:
        # bge 中文模型要求：查询句附上前缀以提升检索效果（向量存储不追加）。
        return "为这个句子生成表示以用于检索相关文章："

    @staticmethod
    def _debug_vector(text: str, dim: int) -> List[float]:
        """确定性 hash 向量：字符串特征（bigrams）映射到 [0,1]^dim。仅 debug 占位。"""
        import hashlib

        vec = [0.0] * dim
        tokens = list(text) if len(text) <= 200 else text[:200]
        for i in range(len(tokens) - 1):
            gram = tokens[i] + tokens[i + 1]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16) % max(dim - 1, 1)
            vec[h] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "api" and self._client is not None:
            return self._client.embed_documents(texts)
        if self.mode == "local" and self._sentence_model is not None:
            return self._sentence_model.encode(texts, normalize_embeddings=True).tolist()
        # debug 兜底
        return [self._debug_vector(t, self.dim) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.mode == "api" and self._client is not None:
            return self._client.embed_query(text)
        if self.mode == "local" and self._sentence_model is not None:
            # bge 检索查询需追加指令前缀，文档向量不追加。
            return self._sentence_model.encode(
                self._query_instruction() + text, normalize_embeddings=True
            ).tolist()
        return self._debug_vector(text, self.dim)


_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


class ChatModel:
    """OpenAI 兼容 Chat 模型封装。未配置 key 时返回 mock（DEBUG）。"""

    def __init__(self) -> None:
        self._llm = None
        if _has_api(config.llm_api_key, config.llm_base_url):
            self._build()

    def _build(self) -> None:
        if importlib.util.find_spec("langchain_openai") is None:
            raise RuntimeError(suggest_knowledge_missing())
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            temperature=0.3,
        )

    @property
    def configured(self) -> bool:
        return self._llm is not None

    def llm(self):
        return self._llm

    def invoke(self, messages) -> str:
        """messages: List[dict] role/content。返回文本。
        未配置 key 时使用 mock 示答案（注释明确标注 DEBUG）。"""
        if self._llm is None:
            return (
                "[DEBUG mock 答案] 未配置 LLM_API_KEY，LLM 未连接。"
                "已检索到如下知识库片段，请配置 key 后获取真实答案。"
            )
        # bind 至 BaseChatModel 通用 invoke：dict 消息亦可被 langchain 接受
        resp = self._llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)


_chat: Optional[ChatModel] = None


def get_chat() -> ChatModel:
    global _chat
    if _chat is None:
        _chat = ChatModel()
    return _chat