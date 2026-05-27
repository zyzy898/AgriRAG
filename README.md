# AgriRAG — 农业领域知识问答系统

基于 **RAG（Retrieval-Augmented Generation）** 的垂直领域智能问答系统。LLM 分批摘要 + 双字段向量存储 + 分数融合重排序。

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
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  LLM 生成    │◀────│  Prompt 组装  │◀────│  Milvus      │
│  (API)       │     │  上下文拼接   │     │  向量检索     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                        相似度阈值过滤(≥0.30)
                                                  │
                                        分数融合 + Cross-Encoder 重排序
                                          α·余弦 + (1-α)·sigmoid(logits)
```

## 核心特性

- **LLM 分批摘要** — 长文档按 10000 字/批分割，每批生成 20 字短摘要 + markdown 长摘要
- **双字段向量存储** — text1（短摘要）生成 embedding 用于检索，text2（长摘要）命中后直接返回给 LLM
- **分数融合重排序** — Milvus 余弦相似度(α=0.4) + bge-reranker-v2-m3 sigmoid 分数(1-α=0.6) 加权融合，初检 60 条 → 精排 → TOP 30
- **相似度阈值过滤** — 低于阈值(默认 0.30)的检索结果自动丢弃，减少噪声
- **Qwen3-Embedding-0.6B** — 本地 Embedding 模型，1024 维稠密向量
- **内容哈希去重** — MD5 确定性 ID，支持增量入库不重复
- **SSE 流式输出** — API 模式逐 token 输出
- **短文档直通** — < 3000 字文档跳过 LLM 摘要，原文直接入库

## 项目结构

```
AgriRAG/
├── config.py              # 全局配置（环境变量可覆盖）
├── models.py              # 模型层：Embedding + LLM API + Reranker
├── milvus_client.py       # 数据层：Milvus 双字段 CRUD
├── ingest.py              # 入库管线：LLM 摘要 → 双字段存储
├── query.py               # 查询管线：检索 → 重排序 → 生成
├── requirements.txt       # Python 依赖
├── database_dir/农业/
│   ├── txt/               # 原始知识文件
│   └── summary/           # LLM 生成的摘要存档
└── Qwen3-Embedding-0.6B/  # Embedding 模型（本地部署）
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

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| Milvus 地址 | `MILVUS_HOST` | 127.0.0.1 | 向量数据库地址 |
| Milvus 端口 | `MILVUS_PORT` | 19530 | 向量数据库端口 |
| API 地址 | `LLM_API_URL` | volces.com | OpenAI 兼容接口 |
| API Key | `LLM_API_KEY` | — | API 密钥 |
| API 模型 | `LLM_API_MODEL` | doubao-seed | 模型名称 |
| 检索数量 | — | 30 | Milvus 返回 top K |
| 相似度阈值 | — | 0.30 | 低于此值的结果过滤 |
| 初检候选 | — | 60 | 输入 reranker 的数量 |
| 分数融合 | — | True | 是否启用 Milvus+Reranker 加权融合 |
| 融合权重 α | — | 0.4 | Milvus 余弦相似度权重，剩余给 Reranker |
| Reranker 归一化 | — | True | 对 Reranker logits 做 sigmoid 归一化 |

## 入库流程

```
文档 > 3000 字 ──▶ 分批(10000字/批, 重叠300字)
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
用户问题 ──▶ Qwen3 Embedding ──▶ Milvus COSINE 初检(60条)
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

检索结果通过**两阶段评分 + 加权融合**决定排序：

1. **Milvus 初检** — 余弦相似度（0~1），过滤 < 0.30 的结果
2. **Reranker 精评** — bge-reranker-v2-m3 Cross-Encoder 对每对 (问题, 文档) 打分，sigmoid 归一化到 [0,1]
3. **分数融合** — `综合分数 = α × Milvus余弦 + (1-α) × sigmoid(Reranker)`

每条上下文标注三维分数：`(综合: 0.7234, Milvus: 0.6500, Reranker: 0.7723)`

`FUSION_ALPHA`（默认 0.4）控制 Milvus 的权重：调大更偏语义相似，调小更偏深度语义匹配。
