"""LangGraph 全链路 Agent。

节点：
  rewrite   → 改写/澄清用户问题
  tools     → REACT 风格多轮工具调用（混合检索/追问版本），支持多跳检索
  generate  → 基于上下文生成带引用的答案
  条件边    → 基于反思（reflect）判断是否补充检索，达到上限后收尾

状态驱动力取自 ChatModel：未配置 key（debug）时自动退化为
「直接混合检索 → mock 答案+真实引用」，保证无 key 也能端到端演示。
本图用 StateGraph 显式建模，便于面试讲解。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from config import config

from ..inference.llm import get_chat
from ..retrieval.hybrid import HybridRetriever
from ..wiki.store import WikiStore
from .tools import all_tools, get_doc_versions, get_latest_doc, search_knowledge_base

REWRITE_PROMPT = (
    "请把用户问题改写为适合检索企业知识库的 query。只输出改写后的 query 本身，"
    "不要解释。若无需改写，原样返回。\n用户问题：{question}"
)

SYSTEM_GEN = (
    "你是企业知识库问答助手。请严格依据提供的知识库片段作答，在答案中对每个事实"
    "增加来源标注（例如 [hr/emp-handbook#v2-3]）。若片段不足以回答，明确说明"
    "「知识库中未找到相关内容」。禁止编造。\n\n知识库片段：\n{context}\n\n"
    "请回答用户问题如下。"
)

REFLECT_PROMPT = (
    "判断当前检索到的知识库片段是否足够回答下列问题。只需回答 yes 或 no。\n"
    "问题：{question}\n片段：\n{context}"
)


class AgentState(TypedDict, total=False):
    question: str
    rewritten: str
    context: str
    answer: str
    sources: List[dict]
    steps: int
    reflections: int
    logs: List[str]


class AgentCoordinator:
    def __init__(
        self,
        store: WikiStore | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.store = store or WikiStore()
        self.retriever = retriever or HybridRetriever(self.store)
        self.chat = get_chat()
        self.llm_ok = self.chat.configured
        self.tool_map = {t.name: t for t in all_tools()}
        self.graph = self._build_graph()

    # ---------- 节点 ----------

    def _node_rewrite(self, state: AgentState) -> AgentState:
        if not self.llm_ok:
            return {"rewritten": state["question"]}
        resp = self.chat.invoke(
            [
                {"role": "system", "content": REWRITE_PROMPT.format(question=state["question"])},
                {"role": "user", "content": "请根据上述要求改写。"},
            ]
        ).strip()
        return {"rewritten": resp or state["question"]}

    def _node_retrieve(self, state: AgentState) -> AgentState:
        hits = self.retriever.search(state["rewritten"], top_k=config.top_k)
        sources = [h.to_dict() for h in hits]
        context = _render(hits)
        return {"context": context, "sources": sources, "steps": state.get("steps", 0) + 1}

    def _node_generate(self, state: AgentState) -> AgentState:
        ctx = state.get("context", "")
        if not self.llm_ok:
            answer = (
                "[DEBUG] 未配置 LLM_API_KEY，以下为基于混合检索的 mock 答案。\n"
                "已命中的知识库片段摘要如下，可直接用于演示引用溯源：\n\n" + ctx
            )
        else:
            answer = self.chat.invoke(
                [
                    {"role": "system", "content": SYSTEM_GEN.format(context=ctx, question=state["rewritten"])},
                    {"role": "user", "content": state["question"]},
                ]
            ).strip()
        return {"answer": answer}

    def _should_reflect(self, state: AgentState) -> str:
        """条件边：反思是否需补检。返回 'tools' | 'generate'。"""
        reflections = state.get("reflections", 0)
        if not self.llm_ok or config.max_reflection <= 0:
            return "generate"
        if state.get("steps", 0) >= config.max_agent_steps:
            return "generate"
        if reflections >= config.max_reflection:
            return "generate"
        # 上下文为空 → 可再试一轮
        if not state.get("context", "").strip():
            return "tools"
        resp = self.chat.invoke(
            [
                {"role": "system", "content": REFLECT_PROMPT.format(
                    question=state["rewritten"], context=state.get("context", ""))},
                {"role": "user", "content": "yes 或 no？"},
            ]
        ).strip().lower()
        if "yes" in resp:
            return "generate"
        return "tools"

    def _node_tools(self, state: AgentState) -> AgentState:
        """REACT 风格多轮工具调用：由 LLM 自主决定检索/追问，结果并入 context。"""
        if not self.llm_ok:
            # 无 LLM 时退回单轮混合检索，等价于 _node_retrieve
            return self._node_retrieve(state)
        try:
            llm = self.chat.llm().bind_tools(all_tools())
        except Exception:
            return self._node_retrieve(state)

        context = state.get("context", "") or ""
        for _step in range(max(1, config.max_agent_steps)):
            messages: List[dict] = [
                {"role": "system", "content": "你是知识库检索代理。用工具搜索知识库。已知信息：" + context},
                {"role": "user", "content": state["rewritten"]},
            ]
            resp = llm.invoke(messages)
            content = getattr(resp, "content", "") or ""
            tool_calls = getattr(resp, "tool_calls", []) or []
            if not tool_calls:
                # 无工具调用 → 视为已得到上下文，结束
                break
            context += "\n\n[工具执行]\n"
            for tc in tool_calls:
                name = tc.get("name") or (tc.get("function", {}) or {}).get("name")
                args = tc.get("args") or {}
                if callable(name):
                    pass
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    args = {}
                if isinstance(name, str) and name in self.tool_map:
                    result = self.tool_map[name].invoke(args)
                    context += f"\n◆ {name}({json.dumps(args, ensure_ascii=False)}):\n{result}"
        return {"context": context, "steps": state.get("steps", 0) + 1}

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("rewrite", self._node_rewrite)
        g.add_node("tools", self._node_tools)
        g.add_node("generate", self._node_generate)

        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "tools")
        g.add_conditional_edges(
            "tools",
            self._should_reflect,
            {"tools": "tools", "generate": "generate"},
        )
        g.add_edge("generate", END)
        return g.compile()

    # ---------- 执行 ----------

    def run(self, question: str) -> dict:
        result = self.graph.invoke(
            {"question": question, "rewritten": question, "steps": 0, "reflections": 0}
        )
        return {
            "question": question,
            "rewritten": result.get("rewritten", question),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "steps": result.get("steps", 0),
        }


def _render(hits) -> str:
    from ..retrieval.hybrid import HybridHit

    lines = []
    for i, h in enumerate(hits):
        lines.append(
            f"◆ 片段{i+1}  {h.citation()}（space={h.space}, doc={h.doc_id}, "
            f"v{h.version}, 标题={h.title}）\n{h.text}"
        )
    return "\n\n".join(lines) if lines else "(无相关片段)"