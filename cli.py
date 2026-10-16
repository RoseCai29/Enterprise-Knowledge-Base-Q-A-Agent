"""CLI：摄入 / 问答 / 列表 / 版本 / 检索对比 / 启动 API。

用法：
  python cli.py ingest scripts/seed_docs/员工手册_v1.md --space hr --doc-id emp-handbook
  python cli.py ask "报销流程是什么"
  python cli.py list
  python cli.py versions emp-handbook
  python cli.py retrieve "电梯逃生编号" --compare
  python cli.py serve
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from config import config

from app.ingest.pipeline import ingest_file
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_index import VectorIndex
from app.wiki.store import WikiStore

cli = typer.Typer(help="企业知识库问答 Agent CLI")
console = Console()


def _components():
    config.ensure_dirs()
    store = WikiStore()
    vector = VectorIndex(store)
    bm25 = BM25Index(store)
    retriever = HybridRetriever(store, vector, bm25)
    return store, vector, bm25, retriever


@cli.command()
def ingest(
    file: Path = typer.Argument(..., help="知识文档路径"),
    space: str = typer.Option(..., "--space", "-s", help="所属知识空间"),
    doc_id: str = typer.Option(..., "--doc-id", "-d", help="文档唯一 id"),
    title: str = typer.Option(None, "--title", "-t", help="文档标题（默认取文件名）"),
):
    store, vector, bm25, _ = _components()
    result = ingest_file(store, vector, bm25, file, space=space, doc_id=doc_id, title=title)
    version = result["version"]
    latest = store.get_document(space, doc_id)
    note = ""
    if version and latest and version > 1:
        note = f"\n新版本 v{version} 已覆盖旧版本（旧块已软删除，history 保留）。"
    console.print(
        Panel(
            f"已摄入：space={space}, doc={doc_id}, 标题={result['title']}\n"
            f"版本 v{result['version']}，共 {result['chunks']} 块{note}",
            title="[green]ingest", style="green",
        )
    )


@cli.command()
def ask(
    question: str = typer.Argument(..., help="问题"),
    top_k: int = typer.Option(6, "--top-k"),
    compare: bool = typer.Option(False, "--compare", help="同时打印向量-only / BM25-only 对比"),
):
    from app.agent.graph import AgentCoordinator

    store, vector, bm25, retriever = _components()
    coordinator = AgentCoordinator(store, retriever)
    result = coordinator.run(question)
    console.print(Panel(f"[bold]{result['question']}[/bold]", title="问题"))
    console.print(Panel(result["answer"], title=f"[cyan]答案（steps={result['steps']}）[/cyan]"))
    if result.get("sources"):
        table = Table(title="引用来源")
        table.add_column("来源", style="bold")
        table.add_column("标题")
        table.add_column("版本")
        for s in result["sources"]:
            table.add_row(escape(s["citation"]), escape(s["title"]), f"v{s['version']}")
        console.print(table)
    if compare:
        _print_compare(retriever, question, top_k)


def _print_compare(retriever, query, top_k):
    console.print(Panel("检索方式对比", title="retrieve"))
    rows = {
        "混合(向量+BM25)": {"vector": 1, "bm25": 1},
        "仅向量": {"vector": 1, "bm25": 0},
        "仅BM25": {"vector": 0, "bm25": 1},
    }
    for label, w in rows.items():
        hits = retriever.search(query, top_k=top_k, weights=w)
        citations = " | ".join(h.citation() for h in hits) or "(空)"
        console.print(escape(f"[{label}] → {citations}"))


@cli.command()
def list(space: str = typer.Option(None, "--space", "-s")):
    store, *_ = _components()
    table = Table(title="知识库文档（当前生效）")
    table.add_column("space", style="bold")
    table.add_column("doc_id")
    table.add_column("标题")
    table.add_column("最新版本")
    table.add_column("更新时间")
    for d in store.list_documents(space):
        table.add_row(d.space, d.doc_id, d.title, f"v{d.latest_version}", d.updated_at)
    console.print(table)


@cli.command()
def versions(doc_id: str = typer.Argument(...)):
    store, *_ = _components()
    versions_ = store.versions(doc_id)
    if not versions_:
        console.print(f"文档 {doc_id} 无版本记录。")
        return
    table = Table(title=f"{doc_id} 版本历史")
    table.add_column("版本")
    table.add_column("active")
    table.add_column("创建时间")
    for v in versions_:
        table.add_row(f"v{v.version}", "是" if v.active else "否(被覆盖)", v.created_at)
    console.print(table)


@cli.command()
def retrieve(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(6, "--top-k"),
    vector: bool = typer.Option(True, "--vector/--no-vector"),
    bm25: bool = typer.Option(True, "--bm25/--no-bm25"),
):
    store, vec, b, retriever = _components()
    hits = retriever.search(
        query, top_k=top_k, weights={"vector": 1 if vector else 0, "bm25": 1 if bm25 else 0}
    )
    table = Table(title=f"混合检索：{query}")
    table.add_column("来源")
    table.add_column("分数")
    table.add_column("text")
    for h in hits:
        table.add_row(escape(h.citation()), f"{h.score:.4f}", h.text[:60] + ("..." if len(h.text) > 60 else ""))
    console.print(table)


@cli.command()
def serve(port: int = typer.Option(8000, "--port", "-p")):
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=port, reload=False)


def main():
    cli()


if __name__ == "__main__":
    main()