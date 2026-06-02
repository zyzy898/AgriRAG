import sys
import json
import requests
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import config

embedding_model = None
embedding_tokenizer = None
reranker_model = None


def _free_gpu_memory():
    """释放 PyTorch 缓存的 GPU 显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_reranker():
    """将 Reranker 模型移到 CPU 并释放其 GPU 显存"""
    global reranker_model
    if reranker_model is not None:
        if hasattr(reranker_model, "model"):
            reranker_model.model.to("cpu")
        reranker_model = None
        _free_gpu_memory()


def unload_embedding_model():
    """将 Embedding 模型移到 CPU 并释放其 GPU 显存"""
    global embedding_model, embedding_tokenizer
    if embedding_model is not None:
        embedding_model.cpu()
        embedding_model = None
        embedding_tokenizer = None
        _free_gpu_memory()


def load_embedding_model():
    global embedding_model, embedding_tokenizer
    if embedding_model is None:
        unload_reranker()
        model_path = config.EMBEDDING_MODEL_PATH
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            embedding_model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True
            ).to("cuda:0")
            embedding_model.eval()
        except OSError as e:
            print(f"\n[错误] Embedding 模型加载失败")
            print(f"  路径: {model_path}")
            print(f"  原因: {e}")
            print(f"  请检查模型文件是否完整，或路径是否正确")
            raise SystemExit(1)
        except Exception as e:
            print(f"\n[错误] Embedding 模型加载异常: {e}")
            raise SystemExit(1)
    return embedding_model, embedding_tokenizer


def encode_texts(texts, batch_size=32):
    """分批编码文本，带进度提示，避免大量文本时 OOM"""
    model, tokenizer = load_embedding_model()
    device = next(model.parameters()).device
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_idx = i // batch_size + 1

        if total_batches > 1:
            progress = f"[{batch_idx}/{total_batches}] Embedding 生成中... ({min(i + batch_size, len(texts))}/{len(texts)})"
            print(f"\r{progress}", end="", flush=True)

        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())

    if total_batches > 1:
        print()

    return np.concatenate(all_embeddings, axis=0)


def call_llm_api(system_prompt, user_prompt, temperature=None, max_tokens=None, stream=False):
    """通用 LLM API 调用，支持自定义 system/user prompt，失败时抛出异常"""
    url = config.LLM_API_URL
    if "/chat/completions" not in url:
        url = url.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

    if temperature is None:
        temperature = config.LLM_TEMPERATURE
    if max_tokens is None:
        max_tokens = config.LLM_MAX_TOKENS

    payload = {
        "model": config.LLM_API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=300, stream=stream)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    if stream:
        content = _handle_stream_response(resp)
        if not content:
            raise RuntimeError("API 返回空内容，请检查 API_KEY 或模型配置")
        return content
    else:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def generate_answer(prompt):
    """使用 LLM API 生成答案（供 query.py 调用，流式输出并带有友好错误提示）"""
    try:
        return call_llm_api(
            system_prompt=config.SYSTEM_PROMPT,
            user_prompt=prompt,
            stream=config.LLM_API_STREAM
        )
    except requests.exceptions.ConnectionError:
        return "[错误] 无法连接到 LLM API，请检查 LLM_API_URL 配置"
    except requests.exceptions.Timeout:
        return "[错误] LLM API 请求超时"
    except Exception as e:
        return f"[错误] API 调用失败: {e}"


def _handle_stream_response(resp):
    """处理 SSE 流式响应，逐字打印并返回完整文本"""
    full_content = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_content += content
            except json.JSONDecodeError:
                continue
        elif line.startswith("{"):
            try:
                data = json.loads(line)
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_content += content
            except json.JSONDecodeError:
                continue
    if full_content:
        print()
    return full_content


def rrf_fusion(*args, k=60, dense_weight=None, sparse_weight=None, graph_weight=None, paths=None):
    """RRF (Reciprocal Rank Fusion) 多路召回融合 (支持2路或3路)

    公式: score(doc) = Σ weight_r / (k + rank_i(doc, r))

    支持两种调用方式:
      1. 旧式: rrf_fusion(dense_results, sparse_results, k=60, dense_weight=1.0, sparse_weight=1.0)
      2. 新式: rrf_fusion(paths=[{"results": ..., "weight": ..., "name": ...}, ...], k=60)

    Args:
        k: RRF 平滑参数
        paths: 路径列表，每项为 {"results": list, "weight": float, "name": str}
        旧式参数 (dense_results, sparse_results, dense_weight, sparse_weight) 仍支持

    Returns:
        按 RRF 分数降序的合并结果列表
    """
    if paths is None:
        paths = []
        if len(args) >= 1 and args[0]:
            paths.append({"results": args[0], "weight": dense_weight or 1.0, "name": "dense"})
        if len(args) >= 2 and args[1]:
            paths.append({"results": args[1], "weight": sparse_weight or 1.0, "name": "sparse"})

    scores = {}
    doc_map = {}

    rank_prefix = {"dense": "_dense_rank", "sparse": "_sparse_rank", "graph": "_graph_rank"}

    for path in paths:
        results = path.get("results", [])
        weight = path.get("weight", 1.0)
        name = path.get("name", "unknown")
        rank_key = rank_prefix.get(name, f"_{name}_rank")

        for rank, doc in enumerate(results):
            doc_id = doc.get("id") or f"{doc.get('source', '')}|{doc.get('text1', '')}"
            scores[doc_id] = scores.get(doc_id, 0) + weight / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = dict(doc)
                for pk in rank_prefix.values():
                    doc_map[doc_id][pk] = None
                doc_map[doc_id][rank_key] = rank + 1
            else:
                doc_map[doc_id][rank_key] = rank + 1

    merged = [
        {
            **doc_map[doc_id],
            "score": round(float(scores[doc_id]), 6),
            "rrf_score": round(float(scores[doc_id]), 6),
        }
        for doc_id in scores
    ]
    merged.sort(key=lambda x: x["score"], reverse=True)

    path_names = [p.get("name", "?") for p in paths]
    path_hits_parts = []
    for p in paths:
        rk = rank_prefix.get(p.get("name", ""), f"_{p.get('name', '')}_rank")
        hits = sum(1 for d in merged if d.get(rk) is not None)
        path_hits_parts.append(f"{p.get('name', '?')}{hits}条")
    print(f"  RRF融合: {' + '.join(path_hits_parts)} → 去重{len(merged)}条")

    return merged


def load_reranker():
    """加载 Cross-Encoder reranker 模型"""
    global reranker_model
    if reranker_model is None:
        unload_embedding_model()
        from sentence_transformers import CrossEncoder
        device = config.RERANKER_DEVICE
        reranker_model = CrossEncoder(
            config.RERANK_MODEL_NAME,
            max_length=512,
            device=device
        )
    return reranker_model


def rerank(query, documents, top_k=None):
    """使用 Cross-Encoder 对检索结果重新排序（支持分数融合）

    融合公式: fused = α * Milvus余弦相似度 + (1-α) * sigmoid(Reranker_logits)

    Args:
        query: 用户查询字符串
        documents: 初检结果列表，每项为 {"text": str, "source": str, "score": float}
        top_k: 重排序后保留的数量，默认使用 config.TOP_K

    Returns:
        重排序后的结果列表，score 字段为融合分数（或 Reranker 分数）
    """
    if not documents:
        return documents

    if top_k is None:
        top_k = config.TOP_K

    model = load_reranker()
    texts = [doc["text"] for doc in documents]
    pairs = [[query, text] for text in texts]

    raw_scores = model.predict(pairs, batch_size=config.RERANK_BATCH_SIZE, show_progress_bar=False)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)

    if config.FUSION_ENABLED:
        reranker_scores = 1.0 / (1.0 + np.exp(-raw_scores)) if config.RERANKER_APPLY_SIGMOID else raw_scores

        results = []
        for doc, rerank_score in zip(documents, reranker_scores):
            milvus_score = float(doc.get("score", 0))
            fused = config.FUSION_ALPHA * milvus_score + (1 - config.FUSION_ALPHA) * float(rerank_score)
            doc["score"] = round(fused, 4)
            doc["milvus_score"] = round(milvus_score, 4)
            doc["reranker_score"] = round(float(rerank_score), 4)
            results.append(doc)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    else:
        if config.RERANKER_APPLY_SIGMOID:
            scores = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            scores = raw_scores

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(ranked))

        result = []
        for doc, score in ranked[:top_k]:
            doc["score"] = round(float(score), 4)
            result.append(doc)

        return result
