import sys
import os
import time
import config
import models
import milvus_client

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def retrieve_context(question, top_k=None, show_score=True):
    """多路召回：稠密向量 + BM25稀疏 → RRF融合 → Cross-Encoder重排序 → 组装上下文"""

    # === 路径1: 稠密向量召回 (Milvus COSINE) ===
    if config.MULTI_RECALL_ENABLED:
        dense_k = config.DENSE_RECALL_K
    else:
        dense_k = config.RERANK_RETRIEVAL_K if config.RERANK_ENABLED else (top_k or config.TOP_K)

    query_embedding = models.encode_texts([question])
    dense_results = milvus_client.search_vectors(query_embedding[0].tolist(), dense_k)

    # 稠密路阈值过滤
    dense_filtered = [r for r in dense_results if r.get("score", 0) >= config.SIMILARITY_THRESHOLD]

    # === 路径2: BM25 稀疏关键词召回 ===
    sparse_filtered = []
    if config.MULTI_RECALL_ENABLED:
        try:
            import bm25_retriever
            sparse_results = bm25_retriever.search(question, config.BM25_RECALL_K)
            sparse_filtered = [r for r in sparse_results if r.get("score", 0) > 0]
        except ImportError:
            print("  (jieba/rank-bm25 未安装，回退到单路召回)")
        except Exception as e:
            print(f"  (BM25 检索异常: {e})")

    # === RRF 融合去重 ===
    if config.MULTI_RECALL_ENABLED and sparse_filtered:
        merged = models.rrf_fusion(
            dense_filtered,
            sparse_filtered,
            k=config.RRF_K,
            dense_weight=config.DENSE_WEIGHT,
            sparse_weight=config.SPARSE_WEIGHT
        )
        rrf_top_k = config.RERANK_RETRIEVAL_K if config.RERANK_ENABLED else (top_k or config.TOP_K)
        filtered = merged[:rrf_top_k]
    else:
        filtered = dense_filtered

    if not filtered and dense_results:
        print(f"  (所有结果相似度低于阈值 {config.SIMILARITY_THRESHOLD}，已过滤)")

    # === Reranker 重排序 ===
    if config.RERANK_ENABLED and filtered:
        try:
            rerank_texts = [r.get("text2", r.get("text", "")) for r in filtered]
            filtered_for_rerank = [
                {"text": t, "source": r.get("source", ""), "score": r.get("score", 0)}
                for r, t in zip(filtered, rerank_texts)
            ]
            filtered = models.rerank(question, filtered_for_rerank, top_k or config.TOP_K)
        except Exception as e:
            print(f"  (Reranker 不可用，使用融合结果: {e})")

    # === 组装上下文 ===
    context_parts = []
    for i, result in enumerate(filtered):
        text = result.get("text2", "") or result.get("text", "")
        source = result.get("source", "")
        score = result.get("score", 0.0)
        milvus_s = result.get("milvus_score")
        reranker_s = result.get("reranker_score")
        rrf_s = result.get("rrf_score")
        dense_rank = result.get("_dense_rank")
        sparse_rank = result.get("_sparse_rank")
        if not text:
            continue
        header = f"[{i+1}] {source}"
        if show_score:
            if milvus_s is not None and reranker_s is not None:
                rank_info = ""
                if dense_rank or sparse_rank:
                    parts = []
                    if dense_rank:
                        parts.append(f"稠密#{dense_rank}")
                    if sparse_rank:
                        parts.append(f"稀疏#{sparse_rank}")
                    rank_info = f", {'/'.join(parts)}"
                header += f" (综合: {score:.4f}, Milvus: {milvus_s:.4f}, Reranker: {reranker_s:.4f}{rank_info})"
            elif rrf_s is not None:
                header += f" (RRF: {rrf_s:.4f})"
            else:
                header += f" (相似度: {score:.4f})"
        context_parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(context_parts) if context_parts else "未找到相关知识"


def answer_question(question):
    t0 = time.time()
    print("检索相关知识...")
    context = retrieve_context(question)
    t_retrieve = time.time() - t0

    t1 = time.time()
    print("生成答案...")
    prompt = config.USER_TEMPLATE.format(context=context, question=question)

    if config.LLM_API_STREAM:
        print()

    answer = models.generate_answer(prompt)
    t_generate = time.time() - t1

    print(f"\n检索: {t_retrieve:.2f}s | 生成: {t_generate:.2f}s | 总计: {t_retrieve + t_generate:.2f}s")

    return answer


def main():
    print("=" * 50)
    print("农业知识问答系统 (RAG)")
    recall_mode = "多路召回 (稠密+稀疏)" if config.MULTI_RECALL_ENABLED else "单路稠密召回"
    print(f"召回: {recall_mode} | 阈值: {config.SIMILARITY_THRESHOLD}")
    if config.RERANK_ENABLED:
        rerank_input = config.RERANK_RETRIEVAL_K if config.MULTI_RECALL_ENABLED else config.RERANK_RETRIEVAL_K
        print(f"Reranker: {config.RERANK_MODEL_NAME} (候选{rerank_input}→TOP{config.TOP_K})")
    if config.MULTI_RECALL_ENABLED:
        print(f"RRF: k={config.RRF_K} | 稠密权重={config.DENSE_WEIGHT} | 稀疏权重={config.SPARSE_WEIGHT}")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = answer_question(question)
        if not config.LLM_API_STREAM:
            print(f"\n答案:\n{answer}")
        return

    while True:
        try:
            question = input("\n请输入问题: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("再见!")
                break
            if not question:
                continue

            print()
            answer = answer_question(question)
            if not config.LLM_API_STREAM:
                print(f"\n答案:\n{'-' * 50}")
                print(answer)
                print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
