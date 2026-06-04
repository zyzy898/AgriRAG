import sys
import os
import re
import time
import config
import models
import milvus_client
import cache

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def retrieve_context(question, top_k=None, show_score=True):
    """多路召回：稠密向量 + BM25稀疏 + 图检索 → RRF融合 → Cross-Encoder重排序 → 组装上下文"""

    # === 路径1: 稠密向量召回 (Milvus COSINE) ===
    if config.MULTI_RECALL_ENABLED:
        dense_k = config.DENSE_RECALL_K
    else:
        dense_k = config.RERANK_RETRIEVAL_K if config.RERANK_ENABLED else (top_k or config.TOP_K)

    query_embedding = models.encode_texts([question])
    dense_results = milvus_client.search_vectors(query_embedding[0].tolist(), dense_k)

    models.unload_embedding_model()

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

    # === 路径3: Neo4j 图检索 ===
    graph_filtered = []
    if config.MULTI_RECALL_ENABLED and config.GRAPH_RECALL_ENABLED:
        try:
            import graph_retriever
            graph_results = graph_retriever.search(question, config.GRAPH_RECALL_K)
            graph_filtered = [r for r in graph_results if r.get("score", 0) > 0]
        except ImportError:
            pass
        except Exception as e:
            print(f"  (图检索异常: {e})")

    # === RRF 融合去重 ===
    if config.MULTI_RECALL_ENABLED and (sparse_filtered or graph_filtered):
        rrf_paths = [{"results": dense_filtered, "weight": config.DENSE_WEIGHT, "name": "dense"}]
        if sparse_filtered:
            rrf_paths.append({"results": sparse_filtered, "weight": config.SPARSE_WEIGHT, "name": "sparse"})
        if graph_filtered:
            rrf_paths.append({"results": graph_filtered, "weight": config.GRAPH_WEIGHT, "name": "graph"})
        merged = models.rrf_fusion(paths=rrf_paths, k=config.RRF_K)
        rrf_top_k = config.RERANK_RETRIEVAL_K if config.RERANK_ENABLED else (top_k or config.TOP_K)
        filtered = merged[:rrf_top_k]
    else:
        filtered = dense_filtered

    if not filtered and dense_results:
        print(f"  (所有结果相似度低于阈值 {config.SIMILARITY_THRESHOLD}，已过滤)")

    # === 精排前内容去重 ===
    if config.DEDUP_BEFORE_RERANK_ENABLED and config.RERANK_ENABLED and len(filtered) > 1:
        filtered = models.deduplicate_candidates(
            filtered,
            max_per_source=config.DEDUP_MAX_PER_SOURCE,
            skip_graph=config.DEDUP_SKIP_GRAPH,
            text1_jaccard_threshold=config.DEDUP_TEXT1_JACCARD_THRESHOLD,
            same_source_threshold=config.DEDUP_SAME_SOURCE_THRESHOLD,
        )

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
    num_to_file = {}
    source_map = []
    for i, result in enumerate(filtered):
        text = result.get("text2", "") or result.get("text", "")
        source = result.get("source", "")
        score = result.get("score", 0.0)
        milvus_s = result.get("milvus_score")
        reranker_s = result.get("reranker_score")
        rrf_s = result.get("rrf_score")
        dense_rank = result.get("_dense_rank")
        sparse_rank = result.get("_sparse_rank")
        graph_rank = result.get("_graph_rank")
        if not text:
            continue
        label = source.replace("知识图谱/", "")
        num_to_file[i + 1] = label
        header = f"[{i+1}] {label}"
        if show_score:
            if milvus_s is not None and reranker_s is not None:
                rank_info = ""
                if dense_rank or sparse_rank or graph_rank:
                    parts = []
                    if dense_rank:
                        parts.append(f"稠密#{dense_rank}")
                    if sparse_rank:
                        parts.append(f"稀疏#{sparse_rank}")
                    if graph_rank:
                        parts.append(f"图谱#{graph_rank}")
                    rank_info = f", {'/'.join(parts)}"
                header += f" (综合: {score:.4f}, Milvus: {milvus_s:.4f}, Reranker: {reranker_s:.4f}{rank_info})"
            elif rrf_s is not None:
                header += f" (RRF: {rrf_s:.4f})"
            else:
                header += f" (相似度: {score:.4f})"
        context_parts.append(f"{header}\n{text}")
        if label not in source_map:
            source_map.append(label)

    context = "\n\n---\n\n".join(context_parts) if context_parts else "未找到相关知识"
    source_legend = "\n".join(f"[{j+1}] {s}" for j, s in enumerate(source_map)) if source_map else ""
    return context, num_to_file, source_legend


def _replace_source_numbers(answer, num_to_file):
    def replacer(match):
        inner = match.group(1)
        nums = [int(n.strip()) for n in re.split(r"[,，、]", inner) if n.strip().isdigit()]
        parts = [num_to_file.get(n, f"[{n}]") for n in nums]
        return "、".join(parts)

    answer = re.sub(r"\[(\d+(?:[,，、]\s*\d+)*)\]", replacer, answer)
    return answer


def answer_question(question):
    cached = cache.get_cached_answer(question)
    if cached is not None:
        answer = cached["answer"]
        source_legend = cached.get("source_legend", "")
        cache.increment_counter(question)
        if config.LLM_API_STREAM:
            print(f"\n  [缓存命中] {answer}")
        else:
            print(f"\n  [缓存命中]")
        print(f"缓存命中 | 耗时 <0.01s")
        return answer, source_legend

    t0 = time.time()
    print("检索相关知识...")
    context, num_to_file, source_legend = retrieve_context(question)
    t_retrieve = time.time() - t0

    t1 = time.time()
    print("生成答案...")
    prompt = config.USER_TEMPLATE.format(context=context, question=question)

    if config.LLM_API_STREAM:
        print()

    answer = models.generate_answer(prompt)
    t_generate = time.time() - t1

    if num_to_file:
        answer = _replace_source_numbers(answer, num_to_file)

    if source_legend:
        legend_text = f"\n\n---\n参考文件:\n{source_legend}"
        if config.LLM_API_STREAM:
            print(legend_text)
        answer += legend_text

    print(f"\n检索: {t_retrieve:.2f}s | 生成: {t_generate:.2f}s | 总计: {t_retrieve + t_generate:.2f}s")

    if answer and not answer.startswith("[错误]"):
        count = cache.increment_counter(question)
        threshold = config.REDIS_CACHE_THRESHOLD
        if count == threshold:
            cache.set_cached_answer(question, answer, source_legend)
            print(f"  [缓存] 该问题已达到 {threshold} 次，已缓存")
        elif count > threshold:
            cache.set_cached_answer(question, answer, source_legend)
            print(f"  [缓存] 第 {count} 次，更新缓存答案")

    return answer, source_legend


def main():
    print("=" * 50)
    print("农业知识问答系统 (RAG)")
    recall_parts = ["稠密"]
    if config.MULTI_RECALL_ENABLED:
        recall_parts.append("BM25稀疏")
    if config.MULTI_RECALL_ENABLED and config.GRAPH_RECALL_ENABLED:
        recall_parts.append("Neo4j图谱")
    recall_mode = "多路召回 (" + "+".join(recall_parts) + ")"
    print(f"召回: {recall_mode} | 阈值: {config.SIMILARITY_THRESHOLD}")
    if config.RERANK_ENABLED:
        rerank_input = config.RERANK_RETRIEVAL_K if config.MULTI_RECALL_ENABLED else config.RERANK_RETRIEVAL_K
        print(f"Reranker: {config.RERANK_MODEL_NAME} (候选{rerank_input}→TOP{config.TOP_K})")
    if config.MULTI_RECALL_ENABLED:
        weight_parts = [f"稠密={config.DENSE_WEIGHT}", f"稀疏={config.SPARSE_WEIGHT}"]
        if config.GRAPH_RECALL_ENABLED:
            weight_parts.append(f"图谱={config.GRAPH_WEIGHT}")
        print(f"RRF: k={config.RRF_K} | 权重: {' / '.join(weight_parts)}")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)

    if config.MULTI_RECALL_ENABLED:
        try:
            import bm25_retriever
            print("预加载 BM25 索引...")
            bm25_retriever.build_index()
        except ImportError:
            pass

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer, source_legend = answer_question(question)
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
            answer, source_legend = answer_question(question)
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
