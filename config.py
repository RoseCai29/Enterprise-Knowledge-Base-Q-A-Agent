"""集中配置：读取 .env，统一提供运行时参数。"""
from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 优先加载项目根目录 .env，其次 .env.example 的默认值仅做参考
load_dotenv(BASE_DIR / ".env")


class Config:
    def __init__(self) -> None:
        # ---- LLM ----
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")

        # ---- Embeddings ----
        self.embed_mode: str = os.getenv("EMBED_MODE", "local")  # api | local | debug
        self.embed_base_url: str = os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
        self.embed_api_key: str = os.getenv("EMBED_API_KEY", "")
        self.embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
        self.embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))
        # 本地 embedding（EMBED_MODE=local）
        self.embed_local_model: str = os.getenv("EMBED_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
        self.embed_local_cache: Path = BASE_DIR / os.getenv("EMBED_LOCAL_CACHE", "./.cache/models")
        self.embed_local_device: str = os.getenv("EMBED_LOCAL_DEVICE", "cpu")  # cpu | cuda
        # HuggingFace 镜像端点：直连 huggingface.co 不可达时（如国内网络）用 hf-mirror.com
        self.embed_hf_endpoint: str = os.getenv("EMBED_HF_ENDPOINT", "https://hf-mirror.com")

        # ---- Service / Auth ----
        self.api_key: str = os.getenv("API_KEY", "")  # 空则 /api 不校验 X-API-Key
        self.host: str = os.getenv("SERVICE_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("SERVICE_PORT", "8000"))

        # ---- Storage ----
        self.data_dir: Path = BASE_DIR / os.getenv("DATA_DIR", "./data").strip("./")
        self.db_path: Path = self.data_dir / "knowledge.db"
        self.faiss_path: Path = self.data_dir / "faiss.index"
        self.bm25_path: Path = self.data_dir / "bm25.pkl"
        self.chunks_path: Path = self.data_dir / "chunks.pkl"

        # ---- Retrieval ----
        self.top_k: int = int(os.getenv("TOP_K", "8"))
        self.vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "20"))
        self.bm25_top_k: int = int(os.getenv("BM25_TOP_K", "20"))
        self.rrf_k: int = int(os.getenv("RRF_K", "60"))
        self.retrieval_total_limit: int = int(os.getenv("RETRIEVAL_TOTAL_LIMIT", "30"))

        # ---- Chunking ----
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

        # ---- Agent ----
        self.max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "4"))
        self.max_reflection: int = int(os.getenv("MAX_REFLECTION", "2"))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.joinpath("spaces").mkdir(parents=True, exist_ok=True)


config = Config()