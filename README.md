# AgriRAG — 农业领域知识问答系统

基于 **RAG（Retrieval-Augmented Generation）** 的垂直领域智能问答系统。LLM 分批摘要 + 双字段向量存储 + **三路召回融合** (稠密向量 + BM25稀疏 + Neo4j图检索) + text1 分层去重 + 分数融合重排序 + **Redis 答案缓存**。

## 性能优化（v2）

| 优化项 | 方案 | 效果 |
|--------|------|------|
| 批量子图查询 | `UNWIND $disease_names` 一次性拉取全部病害子图 | N+1 次 Cypher → 1 次 |
| 批量反向查询 | 药剂 `IN $list` + 症状 `UNWIND` 用 `UNION` 合并 | N+M 次 Cypher → 1 次 |
| 症状反查病害 | `症状.描述 ← CONTAINS ← 子串匹配 → :有症状 → 病害` | 支持通过症状描述查找对应病害 |
| GPU 共享 | Embedding 用完自动卸载，释放 GPU 给 Reranker | Reranker CPU(60s) → GPU(3-5s) |
| BM25 启动预加载 | 系统启动时建好 BM25 索引 | 消除首次查询冷启动 |
| Reranker 切 GPU | `RERANKER_DEVICE = "cuda:0"` | 560M 模型 GPU 推理替代 CPU |
| text1 分层去重 | jieba + text1短摘要 Jaccard 分层阈值 | 精排候选缩减 25-40%，总延迟降低 15-30% |
| Redis 答案缓存 | 同一问题被问5次后进入缓存，命中直返 | 重复查询 <0.01s，节省 API 调用 |
| 缓存 CSV 导出 | 高频问答自动同步到 `cache/faq_answers.csv` | 离线分析常用问题 |

## 技术架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  用户问题     │────▶│  Redis 缓存   │────▶│  三路召回     │────▶│  RRF 融合     │
│              │     │  命中→直返    │ 未命中│稠密+BM25+图谱│     │  去重排序     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┐
                     │  LLM实体抽取 → Neo4j图数据库│
                     │  病害/症状/药剂/传播途径    │
                     └────────────────────────────┘
                                                  │
                                         相似度阈值过滤(≥0.30)
                                                  │
                                  text1 分层去重: 同源≤3 + text1 Jaccard分层阈值
                                                  │
                                         Cross-Encoder 重排序 (GPU)
                                                  │
                                         分数融合: α·余弦 + (1-α)·sigmoid(logits)
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  LLM 生成    │◀────│ Prompt 组装  │◀────│  TOP 30      │
│  (API)       │     │  上下文拼接   │     │  最终结果     │
└──────────────┘     └──────────────┘     └──────────────┘
```

## 核心特性

- **LLM 分批摘要** — 长文档使用 LangChain RecursiveCharacterTextSplitter 按 3000 字/批在句子边界分割（重叠 200 字），每批生成 20 字短摘要 + markdown 长摘要
- **双字段向量存储** — text1（短摘要）生成 embedding 用于检索，text2（长摘要）命中后直接返回给 LLM
- **三路召回 + RRF 融合** — 稠密向量 (Milvus COSINE, 60条) + BM25 稀疏关键词 (30条) + Neo4j 图检索 (15条) 三路并行，Reciprocal Rank Fusion 融合去重
- **text1 分层去重** — RRF 融合后、Cross-Encoder 精排前，仅对 text1（LLM 生成的 20 字短摘要）做 Jaccard 去重，分层阈值（同源 0.35 / 跨源 0.85）。图谱候选硬保留。不对 text2（800 字长文）做去重，避免将同一病害的不同角度（症状/用药/时机）误判为重复
- **Neo4j 知识图谱** — 从农业知识文本中自动抽取病害实体和关系（症状、传播途径、防治药剂、关联病害），LLM 辅助三元组抽取，存入 Neo4j 图数据库
- **实体匹配检索** — 支持病害名、药剂名、症状描述三类实体匹配；药剂可反查所治病害，症状可反查对应病害
- **批量 Cypher 优化** — 子图序列化使用 `UNWIND` 一次性拉取，反向查询使用 `UNION` 合并，消除 N+1 问题
- **分数融合重排序** — bge-reranker-v2-m3 Cross-Encoder 精排，Milvus 余弦相似度(α=0.4) + sigmoid(Reranker logits)(1-α=0.6) 加权融合，初检候选 → 精排 → TOP 30
- **GPU 模型互斥** — Embedding 模型用完自动卸载，释放 GPU 给 Reranker；两个模型不会同时占用显存，适配单 GPU 小显存场景
- **相似度阈值过滤** — 低于阈值(默认 0.30)的检索结果自动丢弃，减少噪声
- **Qwen3-Embedding-0.6B** — 本地 Embedding 模型，1024 维稠密向量
- **BM25 启动预加载** — 系统启动时自动构建 BM25 索引，消除首次查询冷启动
- **内容哈希去重** — MD5 确定性 ID，支持增量入库不重复
- **SSE 流式输出** — API 模式逐 token 输出
- **短文档直通** — < 3000 字文档跳过 LLM 摘要，原文直接入库
- **Redis 答案缓存** — 问题 MD5 哈希 → Redis 计数，同一问题被问 5 次后自动缓存答案。缓存命中直接返回（<0.01s）。所有高频问答同步导出到 `cache/faq_answers.csv`，Redis 不可用时静默降级

## 项目结构

```
AgriRAG/
├── config.py                    # 全局配置（环境变量可覆盖）
├── models.py                    # 模型层：Embedding + LLM API + Reranker + RRF融合 + GPU互斥管理
├── milvus_client.py             # 数据层：Milvus 双字段 CRUD
├── ingest.py                    # 入库管线：LLM 摘要 → 双字段存储 → (可选)知识图谱
├── query.py                     # 查询管线：三路召回 → GPU卸载 → RRF融合 → 重排序 → 生成
├── cache.py                     # Redis 缓存层：答案缓存读写 + 优雅降级 + 批量清除
├── bm25_retriever.py            # BM25 稀疏关键词检索引擎（jieba 分词，启动预加载）
├── entity_relation_extractor.py # 知识图谱抽取：规则+LLM提取病害实体和关系 → Cypher入库
├── graph_retriever.py           # 图检索引擎：Neo4j连接、实体链接(病害/药剂/症状)、批量Cypher检索
├── requirements.txt             # Python 依赖
├── models/                      # 本地模型文件（.gitignore 忽略）
│   ├── Qwen3-Embedding-0.6B/    # Embedding 模型（1024维）
│   └── bge-reranker-v2-m3/      # Cross-Encoder 重排序模型
│    
└── database_dir/农业/
    ├── txt/                     # 原始知识文件
    └── summary/                 # LLM 生成的摘要存档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Milvus（向量数据库）

```bash
# Docker 方式
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

### 3. 启动 Neo4j（图数据库，图谱检索需要）

```bash
# Docker 方式
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j123 \
  neo4j:5
```

### 4. 启动 Redis（答案缓存，可选）

```bash
# Docker 方式
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 验证连接
python test_redis.py
```

> Redis 不可用时会自动降级，不影响系统正常运行。

### 5. 知识入库 + 图谱构建

```bash
# 仅向量入库（不含图谱）
python ingest.py --reset

# 向量入库 + Neo4j 知识图谱构建
python ingest.py --reset --build-graph

# 单独构建知识图谱（不重新入库）
python entity_relation_extractor.py --clear

# 测试模式：仅抽取前 5 个病害查看效果
python entity_relation_extractor.py --test
```

### 6. 交互式问答

```bash
# 交互模式
python query.py

# 单次提问
python query.py 樱桃叶斑病如何防治？

# 症状反查病害
python query.py 叶片上出现黄色斑点是什么病害？

# 药剂反查病害
python query.py 戊唑醇能治哪些病害？
```

## 配置说明

编辑 `config.py` 或通过环境变量覆盖：

| 配置项 | 配置键 | 默认值 | 说明 |
|--------|--------|--------|------|
| Milvus 地址 | `MILVUS_HOST` | 127.0.0.1 | 向量数据库地址 |
| Milvus 端口 | `MILVUS_PORT` | 19530 | 向量数据库端口 |
| Neo4j URI | `NEO4J_URI` | bolt://localhost:7687 | 图数据库连接地址 |
| Neo4j 用户 | `NEO4J_USER` | neo4j | 图数据库用户名 |
| Neo4j 密码 | `NEO4J_PASSWORD` | neo4j123 | 图数据库密码 |
| 图检索开关 | `GRAPH_RECALL_ENABLED` | True | 启用 Neo4j 图谱检索 |
| 图检索数量 | `GRAPH_RECALL_K` | 15 | 图检索召回数量 |
| 图检索权重 | `GRAPH_WEIGHT` | 0.8 | 图检索在 RRF 融合中的权重 |
| API 地址 | `LLM_API_URL` | — | OpenAI 兼容接口（环境变量） |
| API Key | `LLM_API_KEY` | — | API 密钥（环境变量） |
| API 模型 | `LLM_API_MODEL` | — | 模型名称（环境变量） |
| 分块大小 | `CHUNK_SIZE` | 800 | LangChain 文本切割块大小 |
| 分块重叠 | `CHUNK_OVERLAP` | 120 | 相邻块重叠字符数 |
| 检索数量 | `TOP_K` | 30 | 最终返回 top K |
| 相似度阈值 | `SIMILARITY_THRESHOLD` | 0.30 | 低于此值的结果过滤 |
| 多路召回 | `MULTI_RECALL_ENABLED` | True | 启用稠密+BM25+图谱多路召回 |
| 稠密召回数 | `DENSE_RECALL_K` | 60 | 稠密向量召回候选数量 |
| BM25 召回数 | `BM25_RECALL_K` | 30 | BM25 稀疏关键词召回数量 |
| RRF 平滑参数 | `RRF_K` | 60 | 融合平滑参数，越大排名靠后影响越小 |
| 稠密权重 | `DENSE_WEIGHT` | 1.0 | 稠密路径在 RRF 中的权重 |
| 稀疏权重 | `SPARSE_WEIGHT` | 1.0 | 稀疏路径在 RRF 中的权重 |
| 去重开关 | `DEDUP_BEFORE_RERANK_ENABLED` | True | 启用精排前 text1 分层去重 |
| 去重同源限制 | `DEDUP_MAX_PER_SOURCE` | 3 | 同源文档最多保留条数 |
| 去重图谱保留 | `DEDUP_SKIP_GRAPH` | True | 图谱候选不参与内容去重 |
| 去重跨源阈值 | `DEDUP_TEXT1_JACCARD_THRESHOLD` | 0.85 | 跨源 text1 Jaccard 阈值（保守） |
| 去重同源阈值 | `DEDUP_SAME_SOURCE_THRESHOLD` | 0.35 | 同源 text1 Jaccard 阈值（区分症状/防治/药剂） |
| Reranker 开关 | `RERANK_ENABLED` | True | 是否启用 Cross-Encoder 重排序 |
| Reranker 设备 | `RERANKER_DEVICE` | cuda:0 | Reranker 推理设备 (cpu / cuda:0) |
| Reranker 候选 | `RERANK_RETRIEVAL_K` | 60 | 输入 reranker 的候选数量 |
| Reranker 批大小 | `RERANK_BATCH_SIZE` | 3 | reranker 推理批次大小 |
| 分数融合 | `FUSION_ENABLED` | True | 是否启用 Milvus+Reranker 加权融合 |
| 融合权重 α | `FUSION_ALPHA` | 0.4 | Milvus 余弦相似度权重，剩余给 Reranker |
| Reranker 归一化 | `RERANKER_APPLY_SIGMOID` | True | 对 Reranker logits 做 sigmoid 归一化 |
| Redis 地址 | `REDIS_HOST` | 127.0.0.1 | 缓存 Redis 地址 |
| Redis 端口 | `REDIS_PORT` | 6379 | 缓存 Redis 端口 |
| Redis DB | `REDIS_DB` | 0 | 缓存 Redis 数据库编号 |
| Redis 密码 | `REDIS_PASSWORD` | — | 缓存 Redis 密码（环境变量） |
| 缓存 TTL | `REDIS_TTL` | 86400 | 答案缓存过期时间（秒），默认 24 小时 |
| 缓存阈值 | `REDIS_CACHE_THRESHOLD` | 5 | 同一问题被问多少次后才进入缓存 |
| 缓存 CSV 目录 | `CACHE_CSV_DIR` | `./cache` | 缓存 Q&A CSV 文件输出目录 |
| 缓存开关 | `REDIS_CACHE_ENABLED` | 1 | 设为 0 关闭缓存 |

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

(可选) --build-graph ──▶ LLM实体关系抽取 → Neo4j 知识图谱
                           病害/症状/药剂/传播途径/关联病害
```

## 检索流程

```
用户问题 ──▶ Redis 缓存查询 (MD5)
               │ 命中 → 直接返回答案 (<0.01s)
               │ 未命中 ↓
           Embedding(GPU) → Milvus COSINE 稠密召回(60条)
                │
                ├──▶ Embedding 卸载 → 释放 GPU
                │
                ├─ 路径2: jieba 分词 → BM25 稀疏关键词召回(30条)
                │
                └─ 路径3: 实体链接(病害/药剂/症状) → 批量 Neo4j 图检索(15条)
                       │
                       ▼
                RRF 三路融合去重排序
                       │
                相似度过滤(≥0.30)
                       │
                text1 分层去重 (同源≤3, text1 Jaccard 0.35/0.85, 图谱保留)
                       │
                Reranker(GPU) 精排
                       │
                分数融合: α·余弦 + (1-α)·sigmoid(logits)
                       │
                Cross-Encoder 精排 → TOP 30
                       │
                组装上下文(text2) → LLM API 生成答案
                       │
                Redis 计数器 +1 → 达到5次后缓存 → 同步 cache/faq_answers.csv
```

### 召回路径决策

| 条件 | 实际召回路径 |
|------|-------------|
| `MULTI_RECALL_ENABLED=False` | 仅稠密向量（一路） |
| `MULTI_RECALL_ENABLED=True`, `GRAPH_RECALL_ENABLED=False` | 稠密 + BM25（二路） |
| 全开且匹配到病害/药剂/症状实体 | 稠密 + BM25 + 图谱（三路） |
| 全开但未匹配到任何实体 | 稠密 + BM25（实际二路，图谱为空） |

## Neo4j 知识图谱

### 图谱模型

```
节点类型:
  病害              — 属性: 名称、病原类型、来源
  症状              — 属性: 描述
  药剂              — 属性: 名称、浓度、用法
  传播途径          — 属性: 方式
  防治方法          — 属性: 描述、类型(agricultural/chemical)

关系类型:
  (病害)-[:有症状]->(症状)
  (病害)-[:用药]->(药剂)
  (病害)-[:传播方式]->(传播途径)
  (病害)-[:防治措施]->(防治方法)
  (病害)-[:关联 {原因: '同药剂/同病原类型/同治法'}]->(病害)
```

### Cypher 查询示例

```cypher
-- 查看某个病害的完整子图
MATCH (d:病害 {名称: '叶斑病'})-[r]-(n) RETURN d, r, n

-- 药剂反查：戊唑醇能治哪些病害
MATCH (d:病害)-[:用药]->(p:药剂 {名称: '戊唑醇'}) RETURN d.名称

-- 症状反查：哪些病害有特定症状
MATCH (d:病害)-[:有症状]->(s:症状)
WHERE s.描述 CONTAINS '叶片黄化'
RETURN d.名称

-- 查找同病原类型的病害
MATCH (d1:病害)-[:关联 {原因: '同病原类型'}]->(d2:病害) RETURN d1.名称, d2.名称
```

### 图检索实体匹配

图检索引擎从用户问题中识别三种实体，并支持反向查找：

| 实体类型 | 正向匹配 | 反向查找 |
|----------|----------|----------|
| 病害名 | 问题中含病害名 → 直接检索子图 | — |
| 药剂名 | — | 药剂 → `[:用药]` → 反查所治病害 |
| 症状描述 | — | 症状 `CONTAINS` → `[:有症状]` → 反查对应病害 |

匹配方式为最长子串优先匹配（长名称优先，短名称去重）。

### 图检索擅长的问题类型

| 问题类型 | 示例 |
|---------|------|
| 精确属性查询 | "流胶病危害什么部位？" |
| 药剂反向查询 | "戊唑醇能治哪些病害？" |
| 症状反向查询 | "叶片上出现黄色斑点是什么病害？" |
| 关联查询 | "哪些病害与叶斑病有相同的传播途径？" |
| 跨病害对比 | "灰霉病和褐腐病的防治方法有什么异同？" |

### 图谱管理

```bash
# 查看图谱中的病害数量（Neo4j Browser: http://localhost:7474）
MATCH (d:病害) RETURN count(d)

# 查看某个病害的所有关系
MATCH (d:病害 {名称: '叶斑病'})-[r]-(n) RETURN d, r, n

# 查找使用相同药剂的病害
MATCH (d1:病害)-[:用药]->(p:药剂)<-[:用药]-(d2:病害)
WHERE d1.名称 < d2.名称
RETURN d1.名称, d2.名称, p.名称
```

## 打分机制

检索结果通过**三阶段评分 + 加权融合**决定排序：

1. **三路召回** — 稠密向量 (Milvus COSINE) + 稀疏关键词 (BM25) + Neo4j 图检索三路并行检索
2. **RRF 融合** — Reciprocal Rank Fusion: `score(doc) = Σ weight_r / (k + rank_i(doc, r))`，按排名位置融合三路结果
3. **text1 分层去重** — 仅对 text1（LLM 生成的 20 字短摘要）做 Jaccard 去重：同源阈值 0.35（区分症状/防治/药剂），跨源阈值 0.85（保守）。图谱候选硬保留，不对 text2 长文做去重
4. **Reranker 精评** — bge-reranker-v2-m3 Cross-Encoder (GPU) 对每对 (问题, 文档) 打分，sigmoid 归一化到 [0,1]
5. **分数融合** — `综合分数 = α × Milvus余弦 + (1-α) × sigmoid(Reranker)`

每条上下文标注多维分数：`(综合: 0.7234, Milvus: 0.6500, Reranker: 0.7723, 稠密#5/稀疏#3/图谱#1)`

`FUSION_ALPHA`（默认 0.4）控制 Milvus 的权重：调大更偏语义相似，调小更偏深度语义匹配。

## GPU 管理

由于 RTX 2050 仅 4GB 显存，无法同时容纳 Embedding 模型 (~1.2GB) 和 Reranker 模型 (~1.1GB)。系统采用自动互斥策略：

```
查询开始 → Embedding 加载到 GPU → encode → 自动卸载
         → BM25 / 图谱 / RRF / text1 去重 (CPU)
         → Reranker 加载到 GPU → 精排 → 返回结果
```

| 函数 | 作用 |
|------|------|
| `unload_embedding_model()` | Embedding → CPU，清空引用，释放 GPU |
| `unload_reranker()` | Reranker → CPU，清空引用，释放 GPU |

`load_embedding_model()` 和 `load_reranker()` 加载前会**自动卸载对方**，确保不会 OOM。

若要回退到 CPU Reranker（无 GPU 环境），修改 `config.py`：

```python
RERANKER_DEVICE = "cpu"
```

## 依赖

- **torch / transformers** — 本地模型推理
- **sentence-transformers** — Cross-Encoder 重排序
- **pymilvus** — 向量数据库客户端
- **langchain** — RecursiveCharacterTextSplitter 文本切割
- **jieba** — 中文分词（BM25）
- **rank-bm25** — BM25 稀疏关键词检索
- **neo4j** — Neo4j 图数据库 Python 驱动
- **redis** — Redis 答案缓存
