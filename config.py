import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Milvus 配置（支持环境变量覆盖）
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", 19530))
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

COLLECTION_NAME = "agriculture_knowledge"

# Embedding 模型
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models/Qwen3-Embedding-0.6B")
EMBEDDING_DIM = 1024

# LLM API 配置（兼容 OpenAI 接口格式，唯一模式）
LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_MODEL = os.environ.get("LLM_API_MODEL", "")
LLM_API_STREAM = True  # API 模式是否启用流式输出
LLM_MAX_TOKENS = 4096   # 生成最大 token 数
LLM_TEMPERATURE = 0.7

# LangChain 文本切割配置
CHUNK_SIZE = 800          # 每个文本块最大字符数（增大以保持病害描述完整）
CHUNK_OVERLAP = 120       # 相邻块之间的重叠字符数

# 知识库
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "database_dir", "农业", "txt")

# 检索参数
TOP_K = 30
SIMILARITY_THRESHOLD = 0.30  # 低于此分数的检索结果不纳入上下文

# 多路召回配置（稠密向量 + 稀疏关键词 + 图检索 → RRF融合）
MULTI_RECALL_ENABLED = True       # 是否启用多路召回（False 则仅用稠密向量召回）
BM25_RECALL_K = 30                # BM25 稀疏关键词召回数量
DENSE_RECALL_K = 60               # 稠密向量召回数量（多路模式下替代 RERANK_RETRIEVAL_K 作为首路召回量）
RRF_K = 60                        # RRF 融合平滑参数，越大排名靠后的文档影响越小
DENSE_WEIGHT = 1.0                # 稠密路径在 RRF 融合中的权重
SPARSE_WEIGHT = 1.0               # 稀疏路径在 RRF 融合中的权重
GRAPH_WEIGHT = 0.8                # 图检索路径在 RRF 融合中的权重

# Neo4j 图数据库配置（支持环境变量覆盖）
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# 图检索配置
GRAPH_RECALL_ENABLED = True       # 是否启用图检索（False 则回退到二路召回）
GRAPH_RECALL_K = 15               # 图检索召回数量

# Reranker 配置
RERANK_ENABLED = True             # 是否启用 Cross-Encoder 重排序
RERANK_MODEL_NAME = "models/bge-reranker-v2-m3"  # reranker 模型名称或本地路径
RERANK_RETRIEVAL_K = 60           # 初检时从 Milvus 检索的候选数量（应 > TOP_K）
RERANK_BATCH_SIZE = 10            # reranker 推理批次大小
RERANKER_DEVICE = "cuda:0"           # Reranker 推理设备: "cpu" / "cuda:0" (显存不足时选 cpu)

# 分数融合配置（Milvus 余弦相似度 + Reranker Cross-Encoder 分数）
FUSION_ENABLED = True              # 是否启用分数融合（False 则仅用 Reranker 分数）
FUSION_ALPHA = 0.4                # Milvus 余弦相似度权重 (0~1)，剩余权重给 Reranker
RERANKER_APPLY_SIGMOID = True     # 对 Reranker 原始 logits 做 sigmoid 归一化到 [0,1]

# Redis 缓存配置（支持环境变量覆盖）
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("None") 
REDIS_TTL = int(os.environ.get("REDIS_TTL", 86400))   # 答案缓存过期时间（秒），默认 24 小时
REDIS_CACHE_ENABLED = os.environ.get("REDIS_CACHE_ENABLED", "1") != "0"

# System Prompt（角色设定和要求，用于 chat template 的 system 消息）
SYSTEM_PROMPT = """你是一位专业的农业技术顾问，擅长樱桃种植、番茄栽培、温室管理等领域。请严格根据下方提供的参考知识来回答用户问题。

回答要求：
1. 充分利用参考知识中的所有相关内容，尽量给出完整回答
2. 涉及具体数值（温度、浓度、用量等）时务必准确引用原文数据
3. 回答要条理清晰，适当使用分点或分段，便于阅读
4. 如有多个知识来源涉及同一问题，综合归纳后给出完整答案
5. 只有参考知识完全不涉及的内容，才说明无法回答

注意：不要在回答中写类似"参考来源:[1],[2]"的编号引用，这些编号对用户没有意义。"""

# User 消息模板（context 和 question 由 query.py 填入）
USER_TEMPLATE = """参考知识：
{context}

用户问题：{question}"""

# 兼容旧代码的完整模板（仅用于不需要 chat template 的场景）
PROMPT_TEMPLATE = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE
