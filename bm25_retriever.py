import jieba
from rank_bm25 import BM25Okapi
import milvus_client

_bm25_index = None
_doc_list = []


def _tokenize(text):
    return list(jieba.cut(text))


def build_index():
    global _bm25_index, _doc_list

    client = milvus_client.get_client()
    if not client.has_collection(milvus_client.config.COLLECTION_NAME):
        _bm25_index = None
        _doc_list = []
        return 0

    all_results = []
    offset = 0
    page_size = 1000
    while True:
        try:
            page = client.query(
                collection_name=milvus_client.config.COLLECTION_NAME,
                filter="id != ''",
                output_fields=["id", "text1", "text2", "source"],
                limit=page_size,
                offset=offset
            )
        except Exception:
            page = client.query(
                collection_name=milvus_client.config.COLLECTION_NAME,
                expr="id != ''",
                output_fields=["id", "text1", "text2", "source"],
                limit=page_size,
                offset=offset
            )
        if not page:
            break
        all_results.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    _doc_list = []
    corpus = []
    for r in all_results:
        text2 = r.get("text2", "")
        if text2:
            _doc_list.append(r)
            corpus.append(_tokenize(text2))

    _bm25_index = BM25Okapi(corpus) if corpus else None
    count = len(corpus)
    print(f"BM25 索引构建完成，共 {count} 条文档")
    return count


def search(query, top_k=30):
    global _bm25_index, _doc_list

    if _bm25_index is None:
        build_index()

    if not _bm25_index or not _doc_list:
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        doc = _doc_list[idx]
        results.append({
            "id": doc.get("id", ""),
            "text1": doc.get("text1", ""),
            "text2": doc.get("text2", ""),
            "text": doc.get("text2", ""),
            "source": doc.get("source", ""),
            "score": score,
        })

    return results


def rebuild():
    global _bm25_index, _doc_list
    _bm25_index = None
    _doc_list = []
    return build_index()
