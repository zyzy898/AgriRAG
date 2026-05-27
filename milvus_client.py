import hashlib
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
import config

_client = None


def get_client():
    """获取 Milvus 客户端，带连接检测"""
    global _client
    if _client is None:
        try:
            _client = MilvusClient(uri=config.MILVUS_URI)
            _client.list_collections()  # 验证连接
        except Exception as e:
            _client = None
            print(f"Milvus 连接失败 ({config.MILVUS_URI}): {e}")
            raise
    return _client


def create_collection():
    """创建双字段 collection（text1=短摘要嵌入用，text2=长摘要返回LLM用）"""
    client = get_client()
    if client.has_collection(config.COLLECTION_NAME):
        print(f"Collection '{config.COLLECTION_NAME}' 已存在，跳过创建")
        return True

    schema = CollectionSchema([
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema("text1", DataType.VARCHAR, max_length=4096),
        FieldSchema("text2", DataType.VARCHAR, max_length=18196),
        FieldSchema("source", DataType.VARCHAR, max_length=256),
        FieldSchema("emb", DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    ])

    client.create_collection(collection_name=config.COLLECTION_NAME, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="emb",
        metric_type="COSINE",
        index_type="AUTOINDEX",
        index_name="vector_index"
    )
    client.create_index(config.COLLECTION_NAME, index_params)
    client.load_collection(config.COLLECTION_NAME)
    print(f"Collection '{config.COLLECTION_NAME}' 创建成功 (双字段 text1+text2)")
    return True


def reset_collection():
    """强制重建 collection（清空所有数据）"""
    client = get_client()
    if client.has_collection(config.COLLECTION_NAME):
        client.drop_collection(config.COLLECTION_NAME)
        print(f"已删除旧 Collection '{config.COLLECTION_NAME}'")
    return create_collection()


def _make_doc_id(text1, source):
    """根据 text1 和来源生成确定性 ID，用于去重"""
    content = f"{source}:{text1}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def insert_documents(documents):
    """插入文档（text1=短摘要嵌入，text2=长摘要内容），基于内容哈希去重"""
    client = get_client()
    if not client.has_collection(config.COLLECTION_NAME):
        create_collection()

    data = []
    skipped = 0
    for doc in documents:
        text1 = doc.get("text1", doc.get("text", ""))
        doc_id = _make_doc_id(text1, doc.get("source", "unknown"))
        existing = client.get(config.COLLECTION_NAME, ids=[doc_id])
        if existing:
            skipped += 1
            continue
        data.append({
            "id": doc_id,
            "text1": text1,
            "text2": doc.get("text2", doc.get("text", "")),
            "source": doc.get("source", "unknown"),
            "emb": doc["embedding"]
        })

    if data:
        client.insert(config.COLLECTION_NAME, data)
        client.flush(config.COLLECTION_NAME)

    if skipped > 0:
        print(f"跳过 {skipped} 个重复文档块")

    return len(data)


def search_vectors(query_vector, top_k=None):
    """检索相似向量，返回 text1（短摘要）、text2（长内容）和 source"""
    if top_k is None:
        top_k = config.TOP_K

    client = get_client()
    if not client.has_collection(config.COLLECTION_NAME):
        return []

    client.load_collection(config.COLLECTION_NAME)
    results = client.search(
        config.COLLECTION_NAME,
        [query_vector],
        output_fields=["id", "text1", "text2", "source"],
        limit=top_k
    )

    if not results:
        return []

    formatted = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        text1 = entity.get("text1", "")
        text2 = entity.get("text2", "")
        # text1 太短时用 text2 作为展示内容
        display_text = text2 if len(text1) < 50 else text1
        formatted.append({
            "id": entity.get("id", ""),
            "text1": text1,
            "text2": text2,
            "text": display_text,
            "source": entity.get("source", ""),
            "score": hit.get("distance", 0.0),
        })
    return formatted
