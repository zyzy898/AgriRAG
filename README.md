# AgriRAG — 农业领域知识问答系统

基于 **RAG（Retrieval-Augmented Generation）** 的垂直领域智能问答系统。LLM 分批摘要 + 双字段向量存储 + 多路召回融合 + 分数融合重排序。

## 技术架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  知识文档     │────▶│  LLM 分批    │────▶│  双字段存储   │
│  (.txt)      │     │  摘要生成     │     │ text1+text2  │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                    │
                    短摘要(text1) ──▶ embedding  │
                    长摘要(text2) ──▶ 存入不嵌入  │
                                                  │
                    ┌─────────────────────────────┘
                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   用户问题    │────▶│  多路召回     │────▶│  RRF 融合     │
│              │     │ 稠密+BM25    │     │  去重排序     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                         相似度阈值过滤(≥0.30)
                                                  │
                                         Cross-Encoder 重排序
                                                  │
                                         分数融合: α·余弦 + (1-α)·sigmoid(logits)
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  LLM 生成    │◀────│  Prompt 组装  │◀────│  TOP 30      │
│  (API)       │     │  上下文拼接   │     │  最终结果     │
└──────────────┘     └──────────────┘     └──────────────┘
```

## 核心特性

- **LLM 分批摘要** — 长文档使用 LangChain RecursiveCharacterTextSplitter 按 3000 字/批在句子边界分割（重叠 200 字），每批生成 20 字短摘要 + markdown 长摘要
- **双字段向量存储** — text1（短摘要）生成 embedding 用于检索，text2（长摘要）命中后直接返回给 LLM
- **多路召回 + RRF 融合** — 稠密向量召回 (Milvus COSINE, 60 条) + BM25 稀疏关键词召回 (30 条) 双路并行，Reciprocal Rank Fusion 融合去重
- **分数融合重排序** — 融合后的候选集经 bge-reranker-v2-m3 Cross-Encoder 精排，Milvus 余弦相似度(α=0.4) + sigmoid(Reranker logits)(1-α=0.6) 加权融合，初检候选 → 精排 → TOP 30
- **相似度阈值过滤** — 低于阈值(默认 0.30)的检索结果自动丢弃，减少噪声
- **Qwen3-Embedding-0.6B** — 本地 Embedding 模型，1024 维稠密向量
- **内容哈希去重** — MD5 确定性 ID，支持增量入库不重复
- **SSE 流式输出** — API 模式逐 token 输出
- **短文档直通** — < 3000 字文档跳过 LLM 摘要，原文直接入库

## 项目结构

```
AgriRAG/
├── config.py              # 全局配置（环境变量可覆盖）
├── models.py              # 模型层：Embedding + LLM API + Reranker + RRF融合
├── milvus_client.py       # 数据层：Milvus 双字段 CRUD
├── ingest.py              # 入库管线：LLM 摘要 → 双字段存储
├── query.py               # 查询管线：多路召回 → RRF融合 → 重排序 → 生成
├── bm25_retriever.py      # BM25 稀疏关键词检索引擎（jieba 分词）
├── requirements.txt       # Python 依赖
├── models/                # 本地模型文件（.gitignore 忽略）
│   ├── Qwen3-Embedding-0.6B/   # Embedding 模型（1024维）
│   ├── bge-reranker-v2-m3/     # Cross-Encoder 重排序模型
│   └── Qwen3.5-0.8B/           # 备用模型
└── database_dir/农业/
    ├── txt/               # 原始知识文件
    └── summary/           # LLM 生成的摘要存档
```

## 快速开始

```bash
cd AgriRAG
pip install -r requirements.txt

# 知识入库（LLM 摘要 + 向量存储）
python ingest.py --reset

# 交互式问答
python query.py

# 单次提问
python query.py 樱桃叶斑病如何防治？
```

## 配置说明

编辑 `config.py` 或通过环境变量覆盖：

| 配置项 | 配置键 | 默认值 | 说明 |
|--------|--------|--------|------|
| Milvus 地址 | `MILVUS_HOST` | 127.0.0.1 | 向量数据库地址 |
| Milvus 端口 | `MILVUS_PORT` | 19530 | 向量数据库端口 |
| API 地址 | `LLM_API_URL` | — | OpenAI 兼容接口（环境变量） |
| API Key | `LLM_API_KEY` | — | API 密钥（环境变量） |
| API 模型 | `LLM_API_MODEL` | — | 模型名称（环境变量） |
| 分块大小 | `CHUNK_SIZE` | 800 | LangChain 文本切割块大小 |
| 分块重叠 | `CHUNK_OVERLAP` | 120 | 相邻块重叠字符数 |
| 检索数量 | `TOP_K` | 30 | 最终返回 top K |
| 相似度阈值 | `SIMILARITY_THRESHOLD` | 0.30 | 低于此值的结果过滤 |
| 多路召回 | `MULTI_RECALL_ENABLED` | True | 启用稠密+BM25 双路召回 |
| 稠密召回数 | `DENSE_RECALL_K` | 60 | 稠密向量召回候选数量 |
| BM25 召回数 | `BM25_RECALL_K` | 30 | BM25 稀疏关键词召回数量 |
| RRF 平滑参数 | `RRF_K` | 60 | 融合平滑参数，越大排名靠后影响越小 |
| 稠密权重 | `DENSE_WEIGHT` | 1.0 | 稠密路径在 RRF 中的权重 |
| 稀疏权重 | `SPARSE_WEIGHT` | 1.0 | 稀疏路径在 RRF 中的权重 |
| Reranker 开关 | `RERANK_ENABLED` | True | 是否启用 Cross-Encoder 重排序 |
| Reranker 候选 | `RERANK_RETRIEVAL_K` | 60 | 输入 reranker 的候选数量 |
| Reranker 批大小 | `RERANK_BATCH_SIZE` | 16 | reranker 推理批次大小 |
| 分数融合 | `FUSION_ENABLED` | True | 是否启用 Milvus+Reranker 加权融合 |
| 融合权重 α | `FUSION_ALPHA` | 0.4 | Milvus 余弦相似度权重，剩余给 Reranker |
| Reranker 归一化 | `RERANKER_APPLY_SIGMOID` | True | 对 Reranker logits 做 sigmoid 归一化 |

## 入库流程

```
文档 > 3000 字 ──▶ RecursiveCharacterTextSplitter 分句切割(3000字/批, 重叠200字)
                       │
                       ├─ 第1批 ──▶ 短摘要(text1) + 长摘要(text2)
                       ├─ 第2批 ──▶ 短摘要(text1) + 长摘要(text2)
                       └─ ...
                             │
                             ▼
                       text1 → embedding → Milvus
                       text2 → 存入库中，不嵌入

文档 ≤ 3000 字 ──▶ 原文直接作为 text1 + text2 入库
```

## 检索流程

```
用户问题 ──▶ ┌─ 路径1: Qwen3 Embedding → Milvus COSINE 稠密召回(60条)
              │
              └─ 路径2: jieba 分词 → BM25 稀疏关键词召回(30条)
                     │
                     ▼
              RRF 融合去重排序
                     │
              相似度过滤(≥0.30)
                     │
              分数融合: α·余弦 + (1-α)·sigmoid(logits)
                     │
              bge-reranker 精排 → TOP 30
                     │
              组装上下文(text2) → LLM API 生成答案
```

## 打分机制

检索结果通过**三阶段评分 + 加权融合**决定排序：

1. **多路召回** — 稠密向量 (Milvus COSINE) + 稀疏关键词 (BM25) 双路并行检索
2. **RRF 融合** — Reciprocal Rank Fusion: `score(doc) = Σ weight_r / (k + rank_i(doc, r))`，按排名位置融合两路结果
3. **Reranker 精评** — bge-reranker-v2-m3 Cross-Encoder 对每对 (问题, 文档) 打分，sigmoid 归一化到 [0,1]
4. **分数融合** — `综合分数 = α × Milvus余弦 + (1-α) × sigmoid(Reranker)`

每条上下文标注多维分数：`(综合: 0.7234, Milvus: 0.6500, Reranker: 0.7723, 稠密#5/稀疏#3)`

`FUSION_ALPHA`（默认 0.4）控制 Milvus 的权重：调大更偏语义相似，调小更偏深度语义匹配。

## 依赖

- **torch / transformers** — 本地模型推理
- **sentence-transformers** — Cross-Encoder 重排序
- **pymilvus** — 向量数据库客户端
- **langchain** — RecursiveCharacterTextSplitter 文本切割
- **jieba** — 中文分词（BM25）
- **rank-bm25** — BM25 稀疏关键词检索
