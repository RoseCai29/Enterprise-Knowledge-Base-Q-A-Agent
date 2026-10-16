# 企业知识库问答 Agent（Enterprise Knowledge Base Q&A Agent）

从零搭建的企业级知识库问答系统：**全链路 Agent + Wiki 层 + 混合检索（向量 + BM25）**。
一句话：把企业文档按 Wiki 组织管理，摄入时「新覆盖老」，问答时由 LangGraph Agent 混合检索后给出带引用的答案。

---

## 1. 能做什么

- **Wiki 层**：文档按 `space / doc_id` 组织，保持结构化；同一文档可多版本演进。
- **知识更新“新覆盖老”**：重新摄入同一文档 → 版本号 +1，旧块软删除（`active=0, is_latest=0`），检索只命中最新内容；旧版本完整留档，可审计、可回滚。
- **混合检索**：向量检索（FAISS）捕捉语义 + BM25 关键词检索（rank_bm25 + jieba）保证精确关键词/专有名词/版本号命中，用 **RRF（Reciprocal Rank Fusion）** 融合。
- **全链路 Agent**：LangGraph 多节点编排（改写 → 工具检索 → 反思补检 → 带引用生成）。
- **两种入口**：CLI + FastAPI REST。

## 2. 技术栈

Python 3.12 · LangChain / **LangGraph** · FAISS · rank-bm25 · jieba · FastAPI · Typer
LLM / Embedding 均走 **OpenAI 兼容接口**（DeepSeek / SiliconFlow / OpenAI 等），`.env` 配置。

## 3. 快速开始

```bash
# 1) 安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2) 配置（可选：不配 key 也能用，检索+版本全通，仅 LLM 答案走 debug mock）
copy .env.example .env     # 填入 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

# 3) 一键摄入演示数据（含员工手册 v1→v2 覆盖演示）
python scripts/seed.py

# 4) 问答（未配 key 时返回带引用的 mock 答案）
python cli.py ask "报销时限是多久？发票有效期几天？"

# 5) 对比检索方式（验证 BM25 对精确关键词的价值）
python cli.py ask "火灾逃生救援专线号码？" --compare

# 6) 启动 REST 服务
python cli.py serve --port 8000
```

> Embedding 默认 `EMBED_MODE=local`（本地 bge 中文模型，首次运行自动下载，
> 经 HuggingFace 镜像 hf-mirror.com）。只需配好 DeepSeek 的 `LLM_API_KEY` 即可
> 获得真实语义检索 + 真实 LLM 答案；`EMBED_MODE=debug/api` 详见 `.env.example`。

## 4. CLI 命令

| 命令 | 说明 |
|---|---|
| `python cli.py ingest <file> -s <space> -d <doc_id> [-t <title>]` | 摄入文档（重复摄入即触发“新覆盖老”） |
| `python cli.py ask "问题" [--compare]` | 问答；`--compare` 打印 混合/仅向量/仅BM25 对比 |
| `python cli.py list [-s space]` | 列出当前生效文档 |
| `python cli.py versions <doc_id>` | 查看某文档版本历史 |
| `python cli.py retrieve "query" [--no-vector/--no-bm25]` | 原始混合检索调试 |
| `python cli.py serve` | 启动 FastAPI |

## 5. REST API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 + 索引状态 |
| `POST` | `/api/ask`  `{"question","top_k"}` | 问答（返回 answer + sources 引用） |
| `POST` | `/api/retrieve` `{"query","top_k","use_vector","use_bm25"}` | 混合检索调试 |
| `POST` | `/api/ingest/file`（multipart） | 上传文档摄入（`space/doc_id` 表单参数） |
| `GET` | `/api/wiki/spaces` / `/api/wiki/docs` | 空间 / 文档列表 |
| `GET` | `/api/wiki/docs/{doc_id}/versions` | 版本历史 |

```bash
curl -X POST http://127.0.0.1:8000/api/ask -H "Content-Type: application/json" -d '{"question":"IT 故障 P0 响应时间要求？"}'
```

## 6. 目录结构

```
├── config.py                 # 集中配置（读 .env）
├── cli.py                    # Typer CLI
├── app/
│   ├── wiki/
│   │   ├── models.py         # Space/Document/Version/Chunk 模型
│   │   └── store.py          # SQLite 存储 + 版本「新覆盖老」逻辑
│   ├── ingest/pipeline.py    # 解析→切块→向量化→入库→重建索引
│   ├── retrieval/
│   │   ├── vector_index.py   # FAISS（np 持久化，CJK 路径安全）
│   │   ├── bm25_index.py     # rank_bm25 + jieba
│   │   └── hybrid.py         # RRF 融合 + is_latest 过滤
│   ├── inference/llm.py      # OpenAI 兼容 chat + embedding（含 debug 兜底）
│   ├── agent/
│   │   ├── tools.py          # 知识库检索/Wiki 查询工具
│   │   └── graph.py          # LangGraph 全链路编排
│   └── api/main.py           # FastAPI
├── scripts/seed.py           # 演示数据一键摄入
├── scripts/seed_docs/        # 示例：员工手册 v1/v2、IT 运维手册
└── tests/test_smoke.py       # 版本覆盖 / RRF / is_latest 过滤 smoke
```

## 7. 核心设计 & 面试讲点

### 7.1 “新覆盖老”的版本体系
文档身份 = `(space, doc_id)`。重摄入 → `version+1`，SQLite 中把旧版本的 chunks 和 version 记录置为 **软删除**（`active=0, is_latest=0`），写入新版本。检索**只读 `is_latest=1 AND active=1`**，所以旧内容“被覆盖”；但 `doc_versions` 保留全历史，可 `versions` 命令审计，可回滚（`rollback_ingest`）。**讲点：软删 + is_latest + 版本号，而非物理删除**。

### 7.2 混合检索与 RRF 融合
纯向量对**精确关键词/专有名词/编号**（如“专线 7250”）常召回不足。故叠加 BM25。融合用 **RRF**：`score(key)=Σ 1/(k+rank)`，对「命中排名」而非原始分数求和，规避两个检索器分数量纲不可比的问题。
用 `python cli.py ask "..." --compare` 可直观对比三者差异。

### 7.3 全链路 Agent（LangGraph）
- **rewrite**：改写/澄清 query。
- **tools**：REACT 风格多轮工具调用（`search_knowledge_base`、`get_doc_versions`、`get_latest_doc`）。
- **reflect（条件边）**：自评上下文是否足够，不足则补检，受 `MAX_AGENT_STEPS / MAX_REFLECTION` 约束。
- **generate**：基于片段生成**带引用**答案（`[space/doc_id#vX-Y]`）。
- 未配置 LLM key 时自动退化为「直接混合检索 + mock 答案」，保证无 key 也能端到端演示。

### 7.4 关键坑位（踩过的真实问题）
- **FAISS 在 Windows 上写索引路径含中文会失败**（`faiss::FileIOWriter` 用 C++ fopen，不认编码字）→ 改为 `np.save` 纯 Python 持久化，加载时在线重建 `IndexFlatIP`，CJK 路径安全且更可移植。
- **Rich 控制台会把方括号当 markup**：引用串 `[it/it-ops#v2-0]` 被吞成空 → 打印前 `rich.markup.escape(...)`。
- **DeepSeek 没有 embedding API** → 引入本地 embedding（sentence-transformers bge 中文），并可插拔 api/local/debug；`EMBED_MODE=local` 只要一个 DeepSeek key 就能跑真实语义检索。
- **huggingface.co 直连不通（国内）** → 默认走 HuggingFace 镜像 `hf-mirror.com`（`EMBED_HF_ENDPOINT`）。
- **切换 embedding 模式/模型后旧 FAISS 维度不一致导致 500**（`assert d==self.d`）→ VectorIndex.search 检测维度不匹配时**自动用当前 embedder 重建索引**，自愈。
- FastAPI 上传接口需 `python-multipart`；Docker 首次启动本地模型会下载模型，用命名卷缓存避免重启重拉。

## 8. 测试

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

覆盖：版本覆盖后只命中最新块、BM25 精确命中专有名词、RRF 混合可用、无 key 下 Agent debug 流程。

## 9. 后续可扩展（Roadmap）

- 增量索引（当前为小库全量重建）。
- 重排（rerank）模型对融合结果精排。
- PDF/DOCX/网页抓取拟定解析接入。
- 权限隔离（按 space/用户级裁剪）+ 引用溯源到版本。
- 评估集（golden QA）与 RAG 指标（Recall@k、Faithfulness）。