"""Agent 工具集。

- search_knowledge_base：混合检索（向量 + BM25 + RRF）核心工具。
- get_doc_versions / get_latest_doc：Wiki 层查询工具，用于追问历史版本/当前内容。

通过 registry() 提供给 LangGraph 的 ToolNode / bind_tools 复用。
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from ..retrieval.hybrid import HybridRetriever
from ..wiki.store import WikiStore

_retriever: Optional[HybridRetriever] = None
_store: Optional[WikiStore] = None


def _get_store() -> WikiStore:
    global _store
    if _store is None:
        _store = WikiStore()
    return _store


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(_get_store())
    return _retriever


def _render_context(hits) -> str:
    lines = []
    for i, h in enumerate(hits):
        lines.append(
            f"◆ 片段{i+1}  {h.citation()}（space={h.space}, doc={h.doc_id}, "
            f"v{h.version}, 标题={h.title}）\n{h.text}"
        )
    return "\n\n".join(lines)


@tool
def search_knowledge_base(query: str, top_k: int = 6) -> str:
    """在企业知识库中做混合检索（向量语义 + BM25 关键词）并返回相关片段。

    用于回答用户问题时检索依据。返回带来源标注的文本片段列表。
    """
    retriever = _get_retriever()
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        return "未检索到相关知识库内容。"
    return _render_context(hits)


@tool
def get_doc_versions(doc_id: str) -> str:
    """查看某篇文档的版本历史（用于某问题涉及文档新旧变更时的追问）。"""
    store = _get_store()
    versions = store.versions(doc_id)
    if not versions:
        return f"文档 {doc_id} 无版本记录。"
    return "\n".join(
        f"v{v.version}  active={v.active}  created_at={v.created_at}  source={v.source}"
        for v in versions
    )


@tool
def get_latest_doc(space: str, doc_id: str) -> str:
    """获取某篇文档最新生效版本的完整内容（分快列出）。"""
    store = _get_store()
    chunks = store.latest_active_chunks(space=space, doc_id=doc_id)
    if not chunks:
        return f"未找到 {space}/{doc_id} 的当前生效内容。"
    chunks_sorted = sorted(chunks, key=lambda c: (c.version, c.seq))
    lines = [f"共 {len(chunks_sorted)} 块："]
    for c in chunks_sorted:
        lines.append(f"--- v{c.version} 块{c.seq} ---\n{c.text}")
    return "\n".join(lines)


def all_tools() -> List:
    return [search_knowledge_base, get_doc_versions, get_latest_doc]


def tool_names() -> List[str]:
    return [t.name for t in all_tools()]