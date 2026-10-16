"""FastAPI 服务：摄入 / 问答 / Wiki 查询 / 检索调试。"""
from __future__ import annotations

import time

from fastapi import Depends, FastAPI, HTTPException, UploadFile, Request
from pydantic import BaseModel, Field

from config import config

from ..agent.graph import AgentCoordinator
from ..core.log import get_logger
from ..ingest.pipeline import IngestError, ingest_file
from ..retrieval.bm25_index import BM25Index
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.vector_index import VectorIndex
from ..wiki.store import WikiStore
from .security import verify_api_key

log = get_logger("api")

app = FastAPI(
    title="企业知识库问答 Agent",
    version="0.2.0",
    description="LangGraph Agent + Wiki 层 + 向量/BM25 混合检索（Dify 自定义工具端点：/api/ask、/api/retrieve）",
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    resp = await call_next(request)
    cost = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %s (%.0fms)", request.method, request.url.path, resp.status_code, cost)
    return resp


config.ensure_dirs()

store = WikiStore()
vector = VectorIndex(store)
bm25 = BM25Index(store)
retriever = HybridRetriever(store, vector, bm25)
coordinator = AgentCoordinator(store, retriever)

log.info(
    "服务就绪 embed_mode=%s llm_configured=%s api_auth=%s docs=%d vec=%s bm25=%s",
    config.embed_mode,
    coordinator.llm_ok,
    bool(config.api_key),
    len(store.list_documents()),
    vector.is_loaded,
    bm25.is_loaded,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    top_k: int = Field(
        default=6, ge=1, le=20, description="返回给 Agent 的引用片段数（来源 top-K）"
    )


class IngestRequest(BaseModel):
    space: str = Field(min_length=1, description="知识空间")
    doc_id: str = Field(min_length=1, description="文档唯一 id")
    title: str | None = Field(default=None, description="文档标题（默认取文件名）")


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, description="检索 query")
    top_k: int = Field(default=6, ge=1, le=20, description="返回片段数")
    use_vector: bool = Field(default=True, description="是否使用向量检索")
    use_bm25: bool = Field(default=True, description="是否使用 BM25 关键词检索")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": coordinator.llm_ok,
        "embed_mode": config.embed_mode,
        "docs": len(store.list_documents()),
        "vector_loaded": vector.is_loaded,
        "bm25_loaded": bm25.is_loaded,
    }


@app.post("/api/ingest", dependencies=[Depends(verify_api_key)])
def api_ingest(req: IngestRequest) -> dict:
    """以文本内容摄入（供 API 直接传入，简化测试）。"""
    if not req.title and not req.doc_id:
        raise HTTPException(400, "需要 doc_id")
    return {"ok": False, "detail": "请使用 multipart 上传文件：POST /api/ingest/file"}


@app.post("/api/ingest/file", dependencies=[Depends(verify_api_key)])
async def api_ingest_file(
    file: UploadFile,
    space: str,
    doc_id: str,
    title: str | None = None,
) -> dict:
    tmp = config.data_dir / "upload" / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())
    try:
        result = ingest_file(store, vector, bm25, tmp, space=space, doc_id=doc_id, title=title)
    except (IngestError, Exception) as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)
    return result


@app.post("/api/ask", dependencies=[Depends(verify_api_key)])
def ask(req: AskRequest) -> dict:
    result = coordinator.run(req.question)
    result["sources"] = result.get("sources", [])[: req.top_k]
    return result


@app.post("/api/retrieve", dependencies=[Depends(verify_api_key)])
def retrieve(req: RetrieveRequest) -> dict:
    """原始混合检索接口（调试/对比向量 vs BM25）。"""
    hits = retriever.search(
        req.query,
        top_k=req.top_k,
        weights={"vector": 1.0 if req.use_vector else 0.0, "bm25": 1.0 if req.use_bm25 else 0.0},
    )
    return {"query": req.query, "hits": [h.to_dict() for h in hits]}


@app.get("/api/wiki/spaces")
def list_spaces():
    return {"spaces": store.list_spaces()}


@app.get("/api/wiki/docs")
def list_docs(space: str | None = None):
    return {"documents": [d.__dict__ for d in store.list_documents(space)]}


@app.get("/api/wiki/docs/{doc_id}/versions")
def doc_versions(doc_id: str):
    return {"doc_id": doc_id, "versions": [v.__dict__ for v in store.versions(doc_id)]}


def run():
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)